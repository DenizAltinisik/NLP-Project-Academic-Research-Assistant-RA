import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from rag_pipeline import (
    load_pdf_documents, 
    chunk_documents, 
    embed_texts, 
    FaissStore, 
    generate_answer, 
    rerank_chunks
)
from sentence_transformers import SentenceTransformer
# rag_pipeline içinde _embed_model'e erişim gerekebilir, yoksa her seferinde yeniden yükleriz
import rag_pipeline
from google.api_core.exceptions import ResourceExhausted

# ------------------- AYARLAR -------------------
PDF_PATH = r"sample_papers\gogus_hastaliklari.pdf"  # Test edilecek PDF
OUTPUT_CSV = "benchmark_results.csv"
PLOT_DIR = "benchmark_plots"

# Test Senaryoları (Bunları değiştirerek deney yapabilirsiniz)
CHUNK_SIZES = [500, 1000, 2000]       # Küçük, Orta, Büyük parçalar
K_VALUES = [3, 5, 8]                  # Az, Orta, Çok bağlam
EMBEDDING_MODELS = ["all-MiniLM-L6-v2"]

# --- SORU HAVUZU (Şablon) ---
# Buraya makalenizle ilgili soruları ve (varsa) beklenen cevapları yazın.
# Ground Truth boş bırakılırsa accuracy hesaplanmaz, sadece süre ve token ölçülür.
TEST_QUESTIONS = [
    # --- BÖLÜM 1: ANATOMİ VE EMBRİYOLOJİ ---
    {
        "question": "Diyafram embriyolojik gelişim sürecinde hangi dört temel yapıdan köken alır?",
        "ground_truth": "Septum transversum, plevraperitonial membranlar, özefagusun dorsal mezenteri ve vücut kasları."
    },
    {
        "question": "Diyaframın innervasyonunu sağlayan frenik sinir hangi spinal segmentlerden köken alır?",
        "ground_truth": "C3, C4 ve C5 spinal segmentlerinden köken alır."
    },
    {
        "question": "Vena kava, özefagus ve aort; diyaframdaki açıklıklardan sırasıyla hangi vertebra seviyelerinde geçerler?",
        "ground_truth": "Vena kava T8, Özefagus (ve vagus) T10, Aort (ve duktus torasikus) T12 seviyesinde geçer."
    },
    {
        "question": "Solunum fonksiyonu açısından diyafram, vital kapasitenin yaklaşık yüzde kaçından tek başına sorumludur?",
        "ground_truth": "Yaklaşık %65-80'inden sorumludur."
    },

    # --- BÖLÜM 2: DİYAFRAM PARALİZİLERİ (FELÇLERİ) ---
    {
        "question": "Tek taraflı diyafram paralizisinin tanısında floroskopik olarak kullanılan doğrulayıcı test nedir?",
        "ground_truth": "Sniff testi (Hastaya sertçe burnundan nefes çektirilerek yapılan test)."
    },
    {
        "question": "Bilateral diyafram paralizisi olan hastalarda supin (sırtüstü) pozisyona geçildiğinde ortaya çıkan tipik klinik tablo nedir?",
        "ground_truth": "Dakikalar içerisinde takipne (hızlı solunum) ve yüzeyel solunumun ortaya çıkmasıdır (Kalp yetmezliğindeki ortopneden bu özelliği ile ayrılır)."
    },
    {
        "question": "Tek taraflı diyafram paralizisinin en sık görülen nedeni nedir?",
        "ground_truth": "Pulmoner kökenli neoplazmın (tümörün) frenik sinir invazyonudur."
    },

    # --- BÖLÜM 3: HERNİLER (FITIKLAR) VE EVENTRASYON ---
    {
        "question": "Morgagni hernisi ve Larrey hernisi diyaframın hangi bölgesinde, neye göre isimlendirilir?",
        "ground_truth": "Sternokostal hiyatusun sağ tarafındaysa Morgagni, sol tarafındaysa Larrey hernisi olarak adlandırılır."
    },
    {
        "question": "En sık görülen hiatal herni tipi hangisidir ve görülme oranı nedir?",
        "ground_truth": "Tip I (Sliding/Kayma) herni olup, tüm hiatal hernilerin yaklaşık %95'ini oluşturur."
    },
    {
        "question": "Bochdalek hernisi anatomik olarak nerede yerleşir ve hangi yaş grubunda daha sık görülür?",
        "ground_truth": "Posterolateral defekttir (Plöroperitoneal kanalın kapanma kusuru) ve sıklıkla çocukluk çağında görülür."
    }
]

def estimate_tokens(text):
    """Basit bir token tahmin fonksiyonu (1 token ~= 4 karakter)"""
    if not text: return 0
    return len(text) // 4

def run_benchmark():
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        
    results = []
    
    print(f"📄 PDF Yükleniyor: {PDF_PATH}...")
    try:
        with open(PDF_PATH, "rb") as f:
            documents = load_pdf_documents(f)
        print(f"✅ PDF Yüklendi. {len(documents)} sayfa.")
    except FileNotFoundError:
        print(f"❌ HATA: Dosya bulunamadı: {PDF_PATH}")
        return

    # --- TEST DÖNGÜSÜ ---
    for model_name in EMBEDDING_MODELS:
        print(f"\n🔄 Model: {model_name}")
        # Modeli pipeline'a yükle (Global değişkeni güncelleyerek)
        rag_pipeline._embed_model = SentenceTransformer(model_name)
        
        for chunk_size in CHUNK_SIZES:
            print(f"  ✂️ Chunk Size: {chunk_size} ...")
            
            # 1. INDEXING SÜRESİ ÖLÇÜMÜ
            t_start_index = time.time()
            
            chunked_docs = chunk_documents(documents, chunk_size=chunk_size)
            texts_only = [d["text"] for d in chunked_docs]
            embeddings = embed_texts(texts_only)
            dim = embeddings.shape[1]
            db = FaissStore(dim)
            db.add(embeddings, chunked_docs)
            
            indexing_time = time.time() - t_start_index
            print(f"     ↳ Indexing Time: {indexing_time:.2f}s ({len(chunked_docs)} chunks)")
            
            for k in K_VALUES:
                print(f"    🔍 K Value: {k} ...")
                
                for q_idx, q_data in enumerate(TEST_QUESTIONS):
                    question = q_data["question"]
                    
                    # 2. SORGULAMA SÜRESİ (LATENCY) ÖLÇÜMÜ
                    t_start_query = time.time()
                    
                    # Retrieval + Re-ranking
                    initial_k = k * 5
                    q_emb = rag_pipeline.get_embed_model().encode([question], convert_to_numpy=True)
                    import numpy as np
                    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
                    
                    initial_chunks = db.search(q_emb[0], k=initial_k)
                    ranked_results = rerank_chunks(question, initial_chunks, top_k=k)
                    retrieved_chunks = [res["text"] for res in ranked_results]
                    
                    # Generation (Retry mekanizmalı)
                    answer = "ERROR"
                    for attempt in range(3):
                        try:
                            time.sleep(2) # API rate limit koruması
                            answer = generate_answer(retrieved_chunks, question)
                            break
                        except Exception as e:
                            print(f"       ⚠️ API Hatası: {e}")
                            time.sleep(10)
                    
                    latency = time.time() - t_start_query
                    
                    # 3. METRİKLERİ KAYDET
                    total_input_tokens = sum([estimate_tokens(c) for c in retrieved_chunks]) + estimate_tokens(question)
                    output_tokens = estimate_tokens(answer)
                    
                    results.append({
                        "Model": model_name,
                        "Chunk_Size": chunk_size,
                        "K": k,
                        "Indexing_Time": indexing_time,
                        "Num_Chunks": len(chunked_docs),
                        "Question_ID": q_idx + 1,
                        "Latency": latency,
                        "Input_Tokens_Est": total_input_tokens,
                        "Output_Tokens_Est": output_tokens,
                        "Total_Tokens_Est": total_input_tokens + output_tokens,
                        "Answer": answer
                    })
                    print(f"       ✅ Soru {q_idx+1} tamamlandı. ({latency:.2f}s)")

    # Sonuçları Kaydet
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig', sep=';')
    print(f"\n💾 Sonuçlar '{OUTPUT_CSV}' dosyasına kaydedildi.")
    
    generate_plots(df)

def generate_plots(df):
    """Elde edilen verilerden otomatik grafikler çizer"""
    print("📊 Grafikler oluşturuluyor...")
    sns.set_theme(style="whitegrid")
    
    # Grafik 1: Chunk Size vs Indexing Time
    plt.figure(figsize=(10, 6))
    # Her Chunk Size için tek bir indexing time var (K'dan bağımsız), duplicates'i siliyoruz
    df_index = df[["Chunk_Size", "Indexing_Time", "Num_Chunks"]].drop_duplicates()
    
    sns.barplot(x="Chunk_Size", y="Indexing_Time", data=df_index, palette="viridis")
    plt.title("Chunk Boyutunun İndeksleme Süresine Etkisi")
    plt.ylabel("Süre (Saniye)")
    plt.xlabel("Chunk Size (Karakter)")
    for i, row in df_index.iterrows():
        # Çubukların üstüne parça sayısını yaz
        plt.text(i, row.Indexing_Time, f"{row.Num_Chunks} Chunks", color='black', ha="center", va="bottom")
    plt.savefig(f"{PLOT_DIR}/indexing_time.png")
    plt.close()
    
    # Grafik 2: K Değeri vs Ortalama Cevap Süresi (Latency)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="K", y="Latency", hue="Chunk_Size", data=df, marker="o", palette="tab10")
    plt.title("K Değeri ve Chunk Boyutunun Cevap Süresine (Latency) Etkisi")
    plt.ylabel("Ortalama Süre (Saniye)")
    plt.xlabel("K (LLM'e Giden Parça Sayısı)")
    plt.savefig(f"{PLOT_DIR}/latency_analysis.png")
    plt.close()
    
    # Grafik 3: Token Kullanımı (Maliyet Analizi)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="K", y="Total_Tokens_Est", hue="Chunk_Size", data=df, palette="magma")
    plt.title("Tahmini Token Tüketimi (Maliyet Analizi)")
    plt.ylabel("Toplam Token (Girdi + Çıktı)")
    plt.xlabel("K Değeri")
    plt.savefig(f"{PLOT_DIR}/token_usage.png")
    plt.close()
    
    print(f"✅ Grafikler '{PLOT_DIR}' klasörüne kaydedildi!")

if __name__ == "__main__":
    run_benchmark()