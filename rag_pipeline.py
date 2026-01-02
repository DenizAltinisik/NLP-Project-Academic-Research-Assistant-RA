# rag_pipeline.py

import os
from dotenv import load_dotenv
load_dotenv()

USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() in ("1", "true", "yes")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
import pickle
import google.generativeai as genai
import time
from google.api_core.exceptions import ResourceExhausted
from PyPDF2 import PdfReader
# ----------------------------- PDF OKUMA -----------------------------
def load_pdf_documents(file_obj):
    """
    PDF'i okur ve sayfa numaralarıyla birlikte bir sözlük listesi döndürür.
    Return: [{'page_content': '...', 'metadata': {'page': 1}}, ...]
    """
    reader = PdfReader(file_obj)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # Sayfa numarası 1'den başlayacak
            documents.append({"page_content": text, "metadata": {"page": i + 1}})
    return documents

# ----------------------------- CHUNKING (METADATA KORUYARAK) -----------------------------
def chunk_documents(documents, chunk_size=2000, chunk_overlap=100):
    """
    Sayfa bazlı metinleri chunklara ayırma
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    
    chunked_docs = []
    for doc in documents:
        # Her pagei kendi içinde bölüyoruz ki page no exact olsun
        chunks = splitter.split_text(doc["page_content"])
        for chunk in chunks:
            chunked_docs.append({
                "text": chunk,
                "metadata": doc["metadata"] # page no'yu buraya taşı
            })
    return chunked_docs

# ----------------------------- FAISS STORE (GÜNCELLENDİ) -----------------------------
class FaissStore:
    def __init__(self, dim, index_path=None):
        self.dim = dim
        self.doc_map = [] # sadece text değil text + metadata saklar

        if index_path and os.path.exists(index_path + ".faiss"):
            self.index = faiss.read_index(index_path + ".faiss")
            with open(index_path + ".pkl", "rb") as f:
                self.doc_map = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors, docs_with_metadata):
        """
        vectors: numpy array
        docs_with_metadata: [{'text': '...', 'metadata': {'page': 1}}, ...]
        """
        vectors = np.array(vectors).astype("float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.doc_map.extend(docs_with_metadata)

    def search(self, vector, k=3):
        v = np.array([vector]).astype("float32")
        faiss.normalize_L2(v)
        _, I = self.index.search(v, k)
        results = []

        for idx in I[0]:
            if idx < len(self.doc_map):
                results.append(self.doc_map[idx]) 

        return results

# ----------------------------- EMBEDDINGS -----------------------------
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def embed_texts(text_list):
    model = get_embed_model()
    vectors = model.encode(text_list, convert_to_numpy=True)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors

# ----------------------------- RE-RANKING (METADATA İLE) -----------------------------
_rerank_model = None

def get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _rerank_model

def rerank_chunks(query, chunks_with_metadata, top_k=3):
    """
    chunks_with_metadata: [{'text': '...', 'metadata': {...}}, ...]
    """
    if not chunks_with_metadata:
        return []
    
    model = get_rerank_model()
    
    # Model sadece textleri puanlar
    chunk_texts = [c['text'] for c in chunks_with_metadata]
    pairs = [[query, txt] for txt in chunk_texts]
    
    scores = model.predict(pairs)
    sorted_indices = np.argsort(scores)[::-1]
    
    results = []
    for i in sorted_indices[:top_k]:
        original_score = float(scores[i])
        normalized_score = 1 / (1 + np.exp(-original_score))
        
        # Orijinal metadatalı objeyi alıp içine skor ekleme
        result_item = chunks_with_metadata[i].copy()
        result_item["score"] = original_score
        result_item["norm_score"] = normalized_score
        
        results.append(result_item)
    
    return results

# ----------------------------- GEMINI API -----------------------------
def call_gemini_api(prompt):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except ResourceExhausted:
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            return f"Error: {str(e)}"
    return "Error: Kota aşıldı."

def local_generate(prompt):
    # Fallback (şimdilik)
    return "Local model not loaded."

def generate_answer(context_texts, question):
    context = "\n\n".join(context_texts)
    prompt = f"""
You are an academic research assistant.
Use ONLY the information in the context.

Context:
{context}

Question: {question}

Answer concisely.
"""
    if USE_GEMINI:
        return call_gemini_api(prompt)
    return local_generate(prompt)


def load_pdf_documents(file_obj, start_page=1, end_page=None):
    """
    PDF'i okur ve belirtilen sayfa aralığını işler.
    start_page: İşlemeye başlanacak sayfa (1'den başlar)
    end_page: Bitiş sayfası (None ise sonuna kadar)
    """
    reader = PdfReader(file_obj)
    total_pages = len(reader.pages)
    
    # Eğer end_page belirtilmemişse veya hatalıysa eof yap
    if end_page is None or end_page > total_pages:
        end_page = total_pages
        
    documents = []
    
    # range fonksiyonu son değeri dahil etmediği için end_page aynen kalacak
    for i in range(start_page - 1, end_page):
        # Index sınırı kontrolü
        if i >= total_pages or i < 0:
            continue
            
        page = reader.pages[i]
        text = page.extract_text()
        if text:
            documents.append({"page_content": text, "metadata": {"page": i + 1}})
            
    return documents

# gereksiz noise avoidi için referans detection func
def detect_reference_page(file_obj):
    """
    PDF içindeki 'References', 'Bibliography' veya 'Kaynaklar' başlıklarını arar.
    Ancak hata yapmamak için sadece makalenin SON YARISINDA arar.
    Bulduğu sayfa numarasını döndürür. Bulamazsa None döner.
    """
    reader = PdfReader(file_obj)
    total_pages = len(reader.pages)
    
    keywords = ["references", "bibliography", "kaynaklar", "citations"]
    
    # Aramaya page no'ın yarısından sonra başlar, pek güvenli değil ama şimdilik işe yarıyor
    start_search_index = int(total_pages * 0.5) 
    
    for i in range(start_search_index, total_pages):
        page = reader.pages[i]
        text = page.extract_text()
        if not text:
            continue
            
        # Metnin sadece ilk 500 karakterine bak
        header_text = text[:500].lower()
        
        lines = header_text.split('\n')
        for line in lines:
            clean_line = line.strip()
            
            is_keyword = any(keyword == clean_line for keyword in keywords)
            is_short_header = any(f"{keyword}" in clean_line and len(clean_line) < 30 for keyword in keywords)
            
            if is_keyword or is_short_header:
                return i + 1 
                
    return None
