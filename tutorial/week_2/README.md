# 語意檢索與模型評估實作教學 (Week 2)

本筆記為教學型實作範例，示範如何從古典詩文本建立語意檢索系統、使用向量資料庫（FAISS / Qdrant）、以 LangChain LCEL 建構可組合的流水線，並介紹以 LLM 驅動的程式生成、執行與模型評估方法。

---

## 📅 專案資訊

- 初版日期：2026.01.31  
- 筆記版本：v1.0.0  
- Notebook：`notebook.ipynb`（範例程式碼分章節演示）

---

## 🚀 本 Notebook 核心學習目標

完成本 Notebook 你將能：

- 理解語意檢索（semantic retrieval）與傳統字串比對的差異與優勢。  
- 熟悉詞向量（Word2Vec / GloVe）與上下文詞向量（BERT / Transformer）的基本概念。  
- 實作文字嵌入（embeddings），並用於建立 FAISS 與 Qdrant 向量資料庫。  
- 了解 FAISS 常見索引類型（IndexFlatL2、IVF、PQ、IVF+PQ）與其取捨。  
- 掌握三種檢索策略：similarity、similarity_score_threshold、MMR（Maximum Marginal Relevance）。  
- 使用 LangChain Expression Language (LCEL) 建構可重組、支援同步/異步與平行化的 Pipeline。  
- 結合 LLM 做「生成 → 評估 → 修正（Reflection）」的自動化流程。  
- 使用 LLM 生成程式碼並安全執行取得結果（code generation + exec pattern）。  
- 建立模型韌性（Model Resilience）測試：輸入擾動（typos、paraphrase、noise）、Cosine similarity、Perplexity 評估。  

---

## 🧠 重點內容概要

1. 語意檢索基礎  
   - 為何單詞比對（keyword matching）會失效（同義、變形、一字多義等問題）。  
   - Word2Vec / GloVe 的向量語義直觀示例（例：king - man + woman ≈ queen）。  
   - Contextual Embeddings 與 Transformer 的優勢：能針對同一詞在不同上下文產生不同向量表示。  

2. Embedding 與向量資料庫實作流程  
   - 選擇適合中文的 embedding（範例使用 BAAI/bge-m3）。  
   - 將文本（本範例為「唐詩三百首」切分後的詩文）轉為 Document，建立向量與 metadata（作者、詩體、詩名）。  
   - 使用 FAISS 建立向量庫（示範保存與載入 local index）。  
   - 進階：將 FAISS 向量取出並建立 IVF+PQ 索引以節省記憶體並提高大規模檢索效率。  

3. FAISS：IndexFlatL2、IVF、PQ 與 IVF+PQ  
   - IndexFlatL2：精確但對大規模資料不實用（慢、耗記憶體）。  
   - IVF（倒排索引）：先用 k-means 分群，查詢時只搜尋部分 cluster（nprobe）。  
   - PQ（產品量化）：把向量切段並用 codebook 編碼，極大壓縮記憶體但引入量化誤差。  
   - IVF+PQ：兼顧速度與記憶體效率，適用大規模相似度搜尋。  

4. 三種檢索策略比較與使用場景  
   - similarity（預設）：找最相似文件，適合「找最直接相關結果」。  
   - similarity_score_threshold：設定相似度閾值以嚴格過濾結果（提升精準度）。  
   - MMR（最大邊際相關性）：在相似度與多樣性間折衷，避免多筆結果互相冗餘。  

5. Qdrant 範例（Server + Filter）  
   - 使用 Qdrant 作為向量服務（上傳 points、payload），並示範透過 metadata filter 做精細化檢索（如詩體、作者過濾）。  
   - 比較 FAISS（local）與 Qdrant（server）在過濾、擴展性上的差異與優勢。  

6. LangChain Expression Language (LCEL) 與 Runnable 概念  
   - 用管線符（|）把 Prompt、LLM、Parser、自訂轉換串成可重用的流水線。  
   - RunnableParallel 平行化、多階段 chain 與異步 ainvoke 的基本用法。  
   - 範例：retriever → context 組合 → prompt → model → output parser 的完整流程（含可配置的 Runtime Configuration）。  

7. LLM 驅動的程式生成與執行（Code Generation + Exec）  
   - 用 LLM 產生 Python 程式碼（嚴格只輸出程式），再用受控的 exec 執行並擷取結果（answer 變數）。  
   - 示範如何處理 exec 的命名空間以降低副作用（使用獨立 dict 作為 env）。  
   - 範例題目：GCD、三角面積、單擺週期、撲克牌機率等。  

8. Model Evaluation 與 Model Resilience（模型穩健性測試）  
   - 識別 Model Drift、Data Drift、Model Resilience 三類問題。  
   - 輸入擾動（Text Corruption、Paraphrasing、Noise Injection、Adversarial Perturbations）自動化生成流程。  
   - 無人力評分策略：輸出語意相似度（embeddings cosine similarity）、自我一致性、多次抽樣比較、往返驗證（round-trip）。  
   - Perplexity（以 GPT-2 或類似 LM 衡量語言流暢性）搭配相似度進行雙軸評估（語意穩定但語言品質下降的情況）。  

---

## 🛠️ 技術棧（Tech Stack）

- Python 生態：pandas、numpy、transformers、torch、gensim  
- Embeddings：HuggingFace / BAAI/bge-m3、Sentence-Transformers（all-MiniLM-L6-v2 等）  
- Vector Stores：FAISS、Qdrant（langchain_qdrant / qdrant-client）  
- LangChain / LangChain Core（LCEL、Runnables、OutputParsers）  
- LLM：OpenAI-like client（ChatOpenAI 範例使用 gpt-4o / gpt-4o-mini）、或本地 Ollama 類型  
- 評估工具：sklearn.metrics.cosine_similarity、GPT2 perplexity（transformers）  

---

## 📖 常見操作速查

- 建立 embedding 與向量庫（FAISS）  
- 以 metadata 過濾：FAISS 採後處理（post-filter），需配合 fetch_k 與 filter 參數  
- 切換 search_type：similarity / similarity_score_threshold / mmr  
- Qdrant 上傳：client.upsert(points=...) 並以 payload 做 filter 查詢  
- LCEL pipeline：prompt | model | parser，並可用 RunnablePassthrough.assign 插入中繼資訊  
- 平行化：RunnableParallel 並行產生多種詩體或多個查詢  
- 異步化：將 I/O 密集步驟改寫為 ainvoke/abatch 以提升吞吐  

---

## ✅ 教學建議與實作提示

- 開發流程建議：先以同步 pipeline 驗證邏輯，確認無誤後再改寫為異步版本以提升效能。  
- FAISS 對中文文本需要穩定的 embedding；若要部署大規模系統，建議先在小資料集上測試 IVF/PQ 參數（nlist, nprobe, m, bits）。  
- 當使用 LLM 生成程式並 exec 時，務必採取 sandbox 或限制可用 builtins 的方式，避免任意執行危險指令。  
- 評估系統應同時考量語意相似度與語言流暢性（Perplexity），兩者合併能更全面地檢測模型在擾動下的表現。  

---

## 參考資源與延伸閱讀

- LangChain 官方文件 / LCEL 教學  
- FAISS 官方 repo 與 IVF/PQ 教學文章  
- Qdrant 文件（collection、filter、payload 用法）  
- HuggingFace / Sentence-Transformers embedding model 列表  
- Transformers（GPT2 用於 Perplexity 計算）  

---

如需範例程式的快速上手或針對某一章節（例如 IVF+PQ 調參、LCEL 串接範例、或 LLM 程式執行 sandbox）提供更詳盡的步驟與範例，歡迎告訴我你想要深化的主題，我會提供對應的 README 範例腳本與可複製的程式片段。