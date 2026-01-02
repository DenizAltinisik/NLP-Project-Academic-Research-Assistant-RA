import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import numpy as np
import PyPDF2
import time

# Arka plan modüllerini import et
from rag_pipeline import (
    load_pdf_documents, 
    chunk_documents,    
    embed_texts,
    FaissStore,
    get_embed_model,
    generate_answer,
    rerank_chunks,
    detect_reference_page
)

# -------------------- SAYFA AYARLARI & STİL --------------------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS ile arayüzü güzelleştirme
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .chunk-card {
        background-color: white;
        color: #333333; /* <-- KRİTİK EKLEME: Yazı rengini koyu griye sabitledik */
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .chunk-card h4 {
        color: #1b5e20; /* Başlıkları da koyu yeşil yaptık ki daha şık dursun */
        margin-bottom: 5px;
    }
    .chunk-card p {
        color: #333333; /* Paragraflar kesinlikle koyu renk olsun */
    }
    .highlight {
        background-color: #e8f5e9;
        padding: 5px;
        border-radius: 5px;
        font-weight: bold;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- YAN PANEL (SIDEBAR) --------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3048/3048122.png", width=80)
    st.title("Araştırma Asistanı")
    st.markdown("---")
    
    st.subheader("1. Doküman Yükle")
    uploaded_file = st.file_uploader("PDF Dosyası Seçin", type=["pdf"])
    
    if "db" not in st.session_state:
        st.session_state["db"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Ayarlar kısmı
    if uploaded_file:
        uploaded_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        total_pages = len(pdf_reader.pages)
        
        st.info(f"📊 Toplam Sayfa: {total_pages}")
        
        # Otomatik referans algılama
        uploaded_file.seek(0)
        detected_ref = detect_reference_page(uploaded_file)
        default_end = detected_ref - 1 if detected_ref else total_pages
        
        if detected_ref:
            st.success(f"🤖 Referanslar {detected_ref}. sayfada tespit edildi.")
            
        page_range = st.slider(
            "İşlenecek Sayfa Aralığı",
            1, total_pages, (1, default_end)
        )
        
        start_page, end_page = page_range
        
        if st.button("🚀 Dokümanı Analiz Et", type="primary"):
            with st.spinner("Yapay zeka dokümanı okuyor ve indeksliyor..."):
                uploaded_file.seek(0)
                documents = load_pdf_documents(uploaded_file, start_page=start_page, end_page=end_page)
                
                if documents:
                    chunked_docs = chunk_documents(documents)
                    texts_only = [d["text"] for d in chunked_docs]
                    embeddings = embed_texts(texts_only)
                    
                    dim = embeddings.shape[1]
                    db = FaissStore(dim)
                    db.add(embeddings, chunked_docs)
                    
                    st.session_state["db"] = db
                    st.toast(f"Analiz Tamamlandı! {len(chunked_docs)} parça indekslendi.", icon="✅")
                else:
                    st.error("Seçilen aralıkta metin bulunamadı.")

    st.markdown("---")
    st.subheader("2. Arama Ayarları")
    k_val = st.slider("Okunacak Parça Sayısı (K)", 1, 10, 3)
    st.caption("Daha yüksek K değeri daha fazla bağlam sağlar ama hızı düşürebilir.")

# -------------------- ANA EKRAN (MAIN) --------------------

st.header("🎓 Akıllı Akademik Asistan")
st.markdown("Bu asistan, yüklediğiniz makaleleri **RAG teknolojisi** ile analiz eder, referans göstererek cevaplar.")

# db hazır değilse karşılama ekranı göster
if not st.session_state.get("db"):
    st.info("👈 Lütfen sol panelden bir PDF yükleyin ve 'Dokümanı Analiz Et' butonuna basın.")
    
    # Demo/Placeholder görsel
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: gray; padding: 50px;">
            <h3>Bekleniyor...</h3>
            <p>Sistem analize hazır olduğunda burada sohbet ekranı açılacaktır.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # --- CHAT ARAYÜZÜ ---
    
    # Geçmiş mesajları göster
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcıdan soru al
    if prompt := st.chat_input("Makale hakkında bir soru sorun... (Örn: Bu çalışmanın temel amacı nedir?)"):
        
        # Kullanıcı mesajını ekle ve göster
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Asistan cevabını üret
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Makale taranıyor..."):
                db = st.session_state["db"]
                INITIAL_RETRIEVAL_K = max(k_val * 5, 20)
                
                # 1. Retrieval
                q_emb = get_embed_model().encode([prompt], convert_to_numpy=True)
                q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
                initial_results = db.search(q_emb[0], k=INITIAL_RETRIEVAL_K)
                
                # 2. Re-ranking
                ranked_results = rerank_chunks(prompt, initial_results, top_k=k_val)
                final_text_chunks = [res["text"] for res in ranked_results]
                
                # 3. Generation
                answer = generate_answer(final_text_chunks, prompt)
                
                # Cevabı göster
                message_placeholder.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})

            # --- KANITLARI GÖSTER (EXPANDER İLE) ---
            with st.expander("🔍 Kaynaklar ve Kanıtlar (Dokümandan Kesitler)"):
                tabs = st.tabs([f"Kanıt {i+1}" for i in range(len(ranked_results))])
                
                for i, (tab, res) in enumerate(zip(tabs, ranked_results)):
                    page_num = res.get("metadata", {}).get("page", "?")
                    score = res["score"]
                    text = res["text"]
                    
                    with tab:
                        # Özel HTML tasarımı ile kart görünümü
                        st.markdown(f"""
                        <div class="chunk-card">
                            <h4>📄 Sayfa: {page_num}</h4>
                            <p style="font-size: 0.9em; color: gray;">Güven Skoru: <b>{score:.4f}</b></p>
                            <hr>
                            <p>{text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Görselleştirmek için Progress Bar
                        st.progress(res["norm_score"], text="Alaka Düzeyi")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Powered by Gemini 2.0 Flash & FAISS | Developed for Academic Research</div>", unsafe_allow_html=True)