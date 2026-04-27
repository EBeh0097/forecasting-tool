# Full Merged Streamlit App v3

This app includes:
1. Medicare Appeals Search + Analysis
2. PDF Q&A with grounded retrieval
3. Health Plan Formatter

## Updates in this version
- Appeals tab shows summarized output only
- Appeals tab no longer includes LLM Q&A on the analysis output
- Appeals tab now includes a plain-language summary explaining how each appeals metric is calculated
- MeasureID updates:
  - MTM = 36
  - N-API = 35
  - SNP = 43
- If numerator or denominator is missing in the input file, the formatter backfills them from Rate
  - standard measures use denominator 100
  - N-CHP and N-CDP use a default inferred denominator of 365

## Local run
pip install -r requirements.txt
playwright install chromium
streamlit run app.py

## Streamlit secrets
OPENAI_API_KEY = "your-key-here"
