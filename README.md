# Intelligent Academic Research Assistant — Gemini-capable RAG Demo

Quick start:
1. Create virtual env and install:
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt

2. Copy .env.example to .env and edit if you want Gemini:
   cp .env.example .env
   # set USE_GEMINI=true and GEMINI_API_KEY=your_key if you will use Gemini

3. Run:
   streamlit run app.py

Files:
- app.py         -> Streamlit UI (upload PDF, index, QA, summarize)
- rag_pipeline.py-> pipeline: PDF->chunks->embeddings->FAISS + generator helper
- sample_papers/ -> put PDFs here to test (not committed to repo for copyright issues)
- listModels.py -> Lists available llm models
- requirements.txt -> Needed to be installed for system
- test_paper.py -> benchmark, questions are volatile for distinct papers
- benchmark_plots/ -> Visual outputs of benchmark execution
