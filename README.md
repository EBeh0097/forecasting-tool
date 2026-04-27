# AI-Powered Healthcare Analytics Tool
Developed a multi-functional Streamlit application that integrates web scraping, data transformation, and retrieval-augmented generation
(RAG). The tool automates Medicare appeals analysis, enables intelligent querying of large documents, and standardizes healthcare datasets
with domain-specific logic, improving efficiency and decision-making workflows.

# Full Merged Streamlit App

This app includes:
1. Medicare Appeals Search + Analysis
2. PDF Q&A with grounded retrieval
3. Health Plan Formatter

# Key Capabilities

## Medicare Appeals Analysis
Automatically scrapes and aggregates appeals data from external sources
Performs structured analysis to compute key performance metrics (e.g., timeliness and unfavorable decision rates)
Presents clean, summarized insights for decision-making

## Document Intelligence (RAG)
Allows users to upload large PDF documents
Uses embeddings and vector search to enable grounded, context-aware question answering

## Healthcare Data Formatter
Transforms raw health plan data into standardized formats
Applies domain-specific logic for measure calculations, ID mapping, and data completion
Handles missing values and recalculates metrics accurately

# Technical Highlights
Built with Python, Streamlit, and LangChain
Uses Playwright for dynamic web scraping
Implements FAISS vector database for document retrieval
Designed with a modular, multi-tool architecture for scalability
