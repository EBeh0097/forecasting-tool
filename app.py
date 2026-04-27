import os
import re
import tempfile
import subprocess
from datetime import datetime
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Appeals + PDF RAG + Health Plan Formatter", layout="wide")
st.title("Appeals + PDF RAG + Health Plan Formatter")

@st.cache_resource
def install_playwright_browser() -> bool:
    try:
        subprocess.run(["playwright", "install", "chromium"], check=True)
        return True
    except Exception as e:
        st.warning(f"Playwright browser install had an issue: {e}")
        return False

PLAYWRIGHT_READY = install_playwright_browser()

def get_openai_api_key() -> Optional[str]:
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.environ.get("OPENAI_API_KEY")

def get_llm() -> ChatOpenAI:
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in Streamlit secrets.")
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)

def get_embeddings() -> OpenAIEmbeddings:
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in Streamlit secrets.")
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

def validate_plan_contract(value: str) -> str:
    value = value.strip().upper()
    if not re.fullmatch(r"[A-Z]\d{4}", value):
        raise ValueError("Plan Contract # must look like H5215.")
    return value

def validate_short_date(value: str) -> str:
    value = value.strip()
    datetime.strptime(value, "%m/%d/%Y")
    return value

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if not df.empty:
        first_row = [str(x).strip().lower() for x in df.iloc[0].tolist()]
        cols = [str(x).strip().lower() for x in df.columns.tolist()]
        if first_row == cols:
            df = df.iloc[1:].reset_index(drop=True)
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df.reset_index(drop=True)

def dataframe_from_html(html: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(StringIO(html))
        tables = [t for t in tables if t.shape[0] > 0 and t.shape[1] > 1]
        if tables:
            return clean_dataframe(max(tables, key=lambda x: x.shape[0] * x.shape[1]))
    except Exception:
        pass
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        headers = [th.get_text(" ", strip=True) for th in rows[0].find_all(["th", "td"])]
        body = []
        for row in rows[1:]:
            vals = [td.get_text(" ", strip=True) for td in row.find_all(["th", "td"])]
            if vals:
                body.append(vals)
        if headers and body:
            max_len = max(len(headers), max(len(r) for r in body))
            headers = headers + [f"col_{i}" for i in range(len(headers), max_len)]
            body = [r + [""] * (max_len - len(r)) for r in body]
            return clean_dataframe(pd.DataFrame(body, columns=headers))
    return pd.DataFrame()

def try_set_max_page_size(page) -> Optional[str]:
    for select_group in [page.locator("select"), page.locator("select[name*='PageSize'], select[id*='PageSize']")]:
        for i in range(select_group.count()):
            select = select_group.nth(i)
            try:
                options = select.locator("option").all_text_contents()
                normalized = [o.strip() for o in options if o.strip()]
                if not normalized:
                    continue
                preferred = None
                if any(o.lower() == "all" for o in normalized):
                    preferred = next(o for o in normalized if o.lower() == "all")
                else:
                    numeric_options = []
                    for o in normalized:
                        m = re.search(r"\d+", o)
                        if m:
                            numeric_options.append((int(m.group()), o))
                    if numeric_options:
                        preferred = max(numeric_options, key=lambda x: x[0])[1]
                if preferred:
                    select.select_option(label=preferred)
                    try:
                        page.wait_for_load_state("networkidle", timeout=15000)
                    except Exception:
                        pass
                    page.wait_for_timeout(1500)
                    return preferred
            except Exception:
                continue
    return None

def try_click_next(page) -> bool:
    candidates = [
        page.get_by_role("link", name=re.compile(r"^next$", re.I)),
        page.get_by_role("button", name=re.compile(r"^next$", re.I)),
        page.get_by_role("link", name=re.compile(r"^>$")),
        page.get_by_role("button", name=re.compile(r"^>$")),
        page.locator("a[aria-label*='Next'], button[aria-label*='Next']"),
        page.locator("a:has-text('Next'), button:has-text('Next')"),
        page.locator(".paginate_button.next, .pagination-next, li.next a"),
    ]
    for candidate in candidates:
        try:
            if candidate.count() == 0:
                continue
            btn = candidate.first
            disabled = (btn.get_attribute("disabled")) or ""
            class_name = (btn.get_attribute("class")) or ""
            aria_disabled = (btn.get_attribute("aria-disabled")) or ""
            if "disabled" in disabled.lower() or "disabled" in class_name.lower() or aria_disabled.lower() == "true":
                continue
            btn.click(timeout=5000)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            page.wait_for_timeout(1500)
            return True
        except Exception:
            continue
    return False

def get_results_table(page) -> pd.DataFrame:
    return dataframe_from_html(page.content())

def collect_all_pages(page, max_pages: int = 200) -> pd.DataFrame:
    page_size_choice = try_set_max_page_size(page)
    collected_frames = []
    seen_signatures = set()
    for _ in range(max_pages):
        df_page = get_results_table(page)
        if df_page.empty:
            break
        signature = (
            tuple(df_page.columns.tolist()),
            tuple(map(tuple, df_page.head(5).astype(str).fillna("").values.tolist())),
            tuple(map(tuple, df_page.tail(5).astype(str).fillna("").values.tolist())),
            df_page.shape,
        )
        if signature in seen_signatures:
            break
        seen_signatures.add(signature)
        collected_frames.append(df_page)
        if page_size_choice and str(page_size_choice).strip().lower() == "all":
            break
        if not try_click_next(page):
            break
    if not collected_frames:
        return pd.DataFrame()
    return clean_dataframe(pd.concat(collected_frames, ignore_index=True).drop_duplicates().reset_index(drop=True))

def scrape_medicare_appeals(plan_contract: str, start_date: str, end_date: str):
    plan_contract = validate_plan_contract(plan_contract)
    start_date = validate_short_date(start_date)
    end_date = validate_short_date(end_date)
    url = "https://medicareappeals.com/AppealSearch"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(2000)
            plan_locators = [
                page.get_by_label("Plan Contract #", exact=False),
                page.locator("input[name*='Plan'][name*='Contract'], input[id*='Plan'][id*='Contract']"),
                page.locator("input").nth(1),
            ]
            start_locators = [
                page.get_by_label("Start Date", exact=False),
                page.locator("input[name*='Start'][name*='Date'], input[id*='Start'][id*='Date']"),
            ]
            end_locators = [
                page.get_by_label("End Date", exact=False),
                page.locator("input[name*='End'][name*='Date'], input[id*='End'][id*='Date']"),
            ]
            def fill_first_working(locator_list, value: str):
                last_error = None
                for loc in locator_list:
                    try:
                        loc.first.wait_for(timeout=5000)
                        loc.first.fill("")
                        loc.first.fill(value)
                        return
                    except Exception as e:
                        last_error = e
                raise last_error
            fill_first_working(plan_locators, plan_contract)
            fill_first_working(start_locators, start_date)
            fill_first_working(end_locators, end_date)
            clicked = False
            for btn in [
                page.get_by_role("button", name="Search"),
                page.get_by_text("Search", exact=True),
                page.locator("input[type='submit'][value*='Search'], button:has-text('Search')")
            ]:
                try:
                    btn.first.click(timeout=5000)
                    clicked = True
                    break
                except Exception:
                    pass
            if not clicked:
                end_locators[0].first.press("Enter")
            page.wait_for_load_state("domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)
            df = collect_all_pages(page)
            return df, {"plan_contract": plan_contract, "start_date": start_date, "end_date": end_date, "row_count": int(len(df))}
        except PlaywrightTimeoutError as e:
            raise RuntimeError(f"Playwright timeout: {str(e)}")
        finally:
            context.close()
            browser.close()

def analyze_medicare_appeals_df(df_results: pd.DataFrame) -> Dict[str, Any]:
    out = {"row_count_used_for_analysis": int(len(df_results))}
    if "Plan Timely" in df_results.columns:
        counts = df_results["Plan Timely"].astype(str).str.strip().value_counts()
        num_yes = int(counts.get("Yes", 0))
        denom = int(counts.get("Yes", 0) + counts.get("No", 0))
        pct = round((num_yes / denom) * 100, 2) if denom > 0 else 0.0
        out["plan_timely_analysis"] = {"num_yes": num_yes, "denom_yes_no": denom, "percentage": pct}
    else:
        out["plan_timely_analysis"] = {"error": "'Plan Timely' column not found."}
    if "IRE Recon Decision" in df_results.columns:
        counts = df_results["IRE Recon Decision"].astype(str).str.strip().value_counts()
        num_unf = int(counts.get("Unfavorable", 0))
        denom = int(counts.get("Favorable", 0) + counts.get("Unfavorable", 0) + counts.get("Partially Favorable", 0))
        pct = round((num_unf / denom) * 100, 2) if denom > 0 else 0.0
        out["ire_recon_decision_analysis"] = {"num_unfavorable": num_unf, "denom_favorable_unfavorable_partially": denom, "percentage": pct}
    else:
        out["ire_recon_decision_analysis"] = {"error": "'IRE Recon Decision' column not found."}
    return out

def build_pdf_vectorstore(uploaded_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.getbuffer())
        temp_path = tmp.name
    docs = PyPDFLoader(temp_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "]).split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    try:
        os.remove(temp_path)
    except Exception:
        pass
    return vectorstore, chunks

def answer_pdf_question(question: str, vectorstore: FAISS):
    llm = get_llm()
    docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(question)
    context = "\n\n".join([f"[Chunk {i} | page {doc.metadata.get('page', 'unknown')}]\n{doc.page_content}" for i, doc in enumerate(docs, start=1)])
    prompt = f"""Answer the question using only the PDF context below.
If the answer is not in the context, say you do not see it in the uploaded PDF.

Question:
{question}

PDF Context:
{context}
"""
    return llm.invoke(prompt).content, docs

MEASURE_ID_MAP = {"N-CHP": 21, "N-CDP": 30, "N-MLC": 23, "N-MLD": 32, "MTM": 36, "N-API": 35, "SNP": 43}

def read_health_plan_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return pd.read_csv(uploaded_file, sep="|")
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type.")

def format_measurement_period(value: Any) -> Any:
    if pd.isna(value):
        return value
    text = str(value).strip()
    parts = re.split(r"\s*-\s*", text)
    if len(parts) != 2:
        return text
    def parse_piece(piece: str):
        piece = piece.strip()
        for fmt in ("%b %Y", "%B %Y", "%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(piece, fmt)
            except Exception:
                pass
        try:
            dt = pd.to_datetime(piece)
            return None if pd.isna(dt) else dt.to_pydatetime()
        except Exception:
            return None
    start = parse_piece(parts[0]); end = parse_piece(parts[1])
    if start and end:
        return f"{start.strftime('%b %Y')} - {end.strftime('%b %Y')}"
    return text

def format_report_date(value: Any) -> Any:
    if pd.isna(value):
        return value
    try:
        dt = pd.to_datetime(value)
        return f"{dt.month}/{dt.day}/{dt.year}"
    except Exception:
        return value

def backfill_numerator_denominator(row: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(row.get("Numerator"), errors="coerce")
    denominator = pd.to_numeric(row.get("Denominator"), errors="coerce")
    rate = pd.to_numeric(row.get("Rate"), errors="coerce")
    acronym = str(row.get("MeasureAcronym", "")).strip()
    if (not pd.isna(numerator) and not pd.isna(denominator)) or pd.isna(rate):
        return row
    if acronym in {"N-CHP", "N-CDP"}:
        inferred_den = 365
        inferred_num = (rate * inferred_den) / (1000 * 30 / 365)
    else:
        inferred_den = 100
        inferred_num = (rate / 100.0) * inferred_den
    if pd.isna(denominator):
        row["Denominator"] = round(inferred_den, 2)
    if pd.isna(numerator):
        row["Numerator"] = round(inferred_num, 2)
    return row

def calculate_rate(row: pd.Series) -> float:
    numerator = pd.to_numeric(row.get("Numerator"), errors="coerce")
    denominator = pd.to_numeric(row.get("Denominator"), errors="coerce")
    acronym = str(row.get("MeasureAcronym", "")).strip()
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    if acronym in {"N-CHP", "N-CDP"}:
        return (numerator / denominator) * 1000 * 30 / 365
    return (numerator / denominator) * 100

def transform_health_plan_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_dataframe(df)
    rename_map = {}
    for col in df.columns:
        normalized = col.strip().lower()
        if normalized == "measurename":
            rename_map[col] = "MeasureName"
        elif normalized == "measurementperiod":
            rename_map[col] = "MeasurementPeriod"
        elif normalized == "reportdate":
            rename_map[col] = "ReportDate"
        elif normalized == "measureacronym":
            rename_map[col] = "MeasureAcronym"
    df = df.rename(columns=rename_map)
    required = ["Contract", "MeasureAcronym", "Source", "MeasurementPeriod", "ReportDate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {', '.join(missing)}")
    if "MeasureName" not in df.columns: df["MeasureName"] = ""
    if "Numerator" not in df.columns: df["Numerator"] = np.nan
    if "Denominator" not in df.columns: df["Denominator"] = np.nan
    if "Rate" not in df.columns: df["Rate"] = np.nan
    df["MeasureAcronym"] = df["MeasureAcronym"].astype(str).str.strip()
    df["Comments"] = df["MeasureName"]
    df["MeasurementPeriod"] = df["MeasurementPeriod"].apply(format_measurement_period)
    df["ReportDate"] = df["ReportDate"].apply(format_report_date)
    base_df = df.copy()
    extras = []
    if "N-MLC" in base_df["MeasureAcronym"].values:
        mld = base_df[base_df["MeasureAcronym"] == "N-MLC"].copy()
        mld["MeasureAcronym"] = "N-MLD"
        extras.append(mld)
    if "N-CHP" in base_df["MeasureAcronym"].values:
        cdp = base_df[base_df["MeasureAcronym"] == "N-CHP"].copy()
        cdp["MeasureAcronym"] = "N-CDP"
        extras.append(cdp)
    df = pd.concat([base_df] + extras, ignore_index=True) if extras else base_df.copy()
    df = df.apply(backfill_numerator_denominator, axis=1)
    df["Rate"] = df.apply(calculate_rate, axis=1)
    df["MeasureID"] = df["MeasureAcronym"].map(MEASURE_ID_MAP)
    return df[["Contract","MeasureAcronym","Rate","Source","Numerator","Denominator","Comments","MeasurementPeriod","ReportDate","MeasureID"]].reset_index(drop=True)

def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="output")
    return output.getvalue()

if not get_openai_api_key():
    st.warning("Add OPENAI_API_KEY to Streamlit secrets before using PDF Q&A.")

tab1, tab2, tab3 = st.tabs(["Appeals Search + Analysis", "PDF Q&A (Grounded)", "Health Plan Formatter"])

with tab1:
    st.subheader("Medicare Appeals Search")
    if PLAYWRIGHT_READY:
        st.success("Playwright browser is ready.")
    c1, c2, c3 = st.columns(3)
    with c1:
        plan_contract = st.text_input("Plan Contract #", value="H5215")
    with c2:
        start_date = st.text_input("Start Date (mm/dd/yyyy)", value="01/01/2025")
    with c3:
        end_date = st.text_input("End Date (mm/dd/yyyy)", value="12/01/2025")
    if st.button("Run Appeals Search", type="primary"):
        try:
            with st.spinner("Scraping Medicare Appeals results..."):
                df_results, metadata = scrape_medicare_appeals(plan_contract, start_date, end_date)
            analysis_output = analyze_medicare_appeals_df(df_results)
            st.success("Appeals search completed.")
            st.subheader("Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Rows", metadata["row_count"])
            pt = analysis_output.get("plan_timely_analysis", {})
            ire = analysis_output.get("ire_recon_decision_analysis", {})
            m2.metric("Plan Timely %", f"{pt['percentage']}%" if "percentage" in pt else "N/A")
            m3.metric("Unfavorable IRE %", f"{ire['percentage']}%" if "percentage" in ire else "N/A")
            rows = []
            rows.append({"Metric": "Plan Timely", "Numerator": pt.get("num_yes"), "Denominator": pt.get("denom_yes_no"), "Percentage": pt.get("percentage", "N/A")})
            rows.append({"Metric": "IRE Recon Decision - Unfavorable", "Numerator": ire.get("num_unfavorable"), "Denominator": ire.get("denom_favorable_unfavorable_partially"), "Percentage": ire.get("percentage", "N/A")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.markdown("### How the calculations are generated")
            st.markdown(
                """
- **Total Rows** is the total number of appeal records scraped from the results table after combining all available pages and removing duplicate rows.
- **Plan Timely %** is calculated from the **Plan Timely** column as:

  `number of Yes records / (number of Yes + number of No records) × 100`

  Records with other values or blanks are not included in that denominator.
- **Unfavorable IRE %** is calculated from the **IRE Recon Decision** column as:

  `number of Unfavorable records / (Favorable + Unfavorable + Partially Favorable records) × 100`

  Records outside those categories are excluded from that denominator.
- The table shown under **Final analysis numbers** displays the exact numerator, denominator, and resulting percentage used for each metric.
                """
            )
        except Exception as e:
            st.error(f"Search failed: {e}")

with tab2:
    st.subheader("Upload a PDF and ask grounded questions")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")
    if pdf_file is not None:
        cache_key = f"{pdf_file.name}_{pdf_file.size}"
        if st.session_state.get("pdf_cache_key") != cache_key:
            try:
                with st.spinner("Indexing PDF for grounded Q&A..."):
                    vectorstore, chunks = build_pdf_vectorstore(pdf_file)
                st.session_state["pdf_vectorstore"] = vectorstore
                st.session_state["pdf_chunks"] = chunks
                st.session_state["pdf_cache_key"] = cache_key
                st.success(f"PDF indexed. Chunks created: {len(chunks)}")
            except Exception as e:
                st.error(f"Failed to index PDF: {e}")
    question = st.text_input("Ask a question about the uploaded PDF", key="pdf_question")
    if st.button("Ask PDF Question"):
        if "pdf_vectorstore" not in st.session_state:
            st.error("Upload and index a PDF first.")
        elif not question.strip():
            st.error("Enter a question.")
        elif not get_openai_api_key():
            st.error("OPENAI_API_KEY is required for PDF Q&A.")
        else:
            try:
                with st.spinner("Retrieving relevant chunks and answering..."):
                    answer, docs = answer_pdf_question(question, st.session_state["pdf_vectorstore"])
                st.subheader("Answer")
                st.write(answer)
                st.subheader("Retrieved Support")
                for i, doc in enumerate(docs, start=1):
                    with st.expander(f"Chunk {i} | page {doc.metadata.get('page', 'unknown')}"):
                        st.write(doc.page_content)
            except Exception as e:
                st.error(f"PDF Q&A failed: {e}")

with tab3:
    st.subheader("Health Plan Formatter")
    st.caption("Accepts .txt, .csv, .xlsx, or .xls. Pipe-delimited .txt files are supported.")
    hp_file = st.file_uploader("Upload health plan input file", type=["txt", "csv", "xlsx", "xls"], key="hp_uploader")
    if hp_file is not None:
        try:
            df_input = read_health_plan_file(hp_file)
            st.subheader("Input Preview")
            st.dataframe(df_input.head(20), use_container_width=True)
            if st.button("Generate Formatted Output"):
                df_output = transform_health_plan_dataframe(df_input)
                st.subheader("Formatted Output")
                st.dataframe(df_output, use_container_width=True)
                st.download_button("Download Formatted Excel", data=dataframe_to_excel_bytes(df_output), file_name="health_plan_formatted_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Formatting failed: {e}")
    st.markdown("**MeasureID mapping used**")
    st.json(MEASURE_ID_MAP)
