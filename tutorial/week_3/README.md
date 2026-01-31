# RAG N-Shot 與應用實作 (Week 3)

本專案為 RAG（Retrieval-Augmented Generation）系列教學的 **Week 3 教學筆記**，內容聚焦於「N-Shot 提示與風格學習」、各類分類應用、資料擷取（網頁 / Word / PDF）、非同步工作流設計，以及以 MLflow 進行模型追蹤的基礎實作。

---

## 📅 專案資訊

- **週次**：week_3  
- **建立日期**：2026.01.31  
- **筆記版本**：v1.0.0  
- **Notebook**：`notebook.ipynb`（請參考內文程式碼與範例）

---

## 🚀 本週核心學習目標

- 理解並實作 RAG + N-Shot Prompting，學會讓 LLM 從範例中「學風格」並產生一致性回應。  
- 建構向量索引與檢索（Qdrant / HuggingFace embeddings），並實作 style-extraction + generate pipeline（可複用於文風模仿）。  
- 掌握分類任務在 LLM 世界的三種策略：Zero-Shot、Few-Shot、LLM-as-a-Judge（使用 Pydantic 輸出解析由模型產生結構化回應）。  
- 完整實作多個範例：飛安報告分類、唐詩/宋詞詩體分類、風格化生成（模仿「掄語」示例）。  
- 網頁爬取與文本清洗（同步與非同步版本）、以及從 Word / PDF 提取文字的方法與非同步封裝。  
- 實作履歷與職缺的自動匹配流程（包含抓取、抽取、格式化、模型判定）。  
- 使用 MLflow 追蹤模型訓練流程（本地 Tracking server、模型註冊、版本與 alias 管理）。

---

## 🧩 重要內容摘要（章節導覽）

1. 範例工具與 pipeline 工廠函式
   - build_standard_chat_prompt_template：用於快速建立 system/human 的 Chat Prompt Template。
   - build_pipeline：把 PromptTemplate 與 model 串成可直接呼叫的 runnable pipeline。

2. 風格學習（N-Shot）
   - 範例：以幽默 / 誇張的「掄語」示例做 embedding → 建立 Qdrant 集合 → 檢索相似例句 → 提取風格（style extraction pipeline）→ 以該風格生成回應。
   - 核心做法：檢索（retriever）→ 把檢索到的 examples 合併成 context → 用 LLM 抽取 style（Pydantic parser）→ 再以 style 作為 system prompt 生成答案。

3. 向量資料庫與檢索（Qdrant）
   - 使用 HuggingFaceEmbeddings（BAAI/bge-m3）建立向量、Qdrant 建立 collection，並示範 add_documents / as_retriever 的流程。
   - 說明 collection 建立時的 dimension、distance 等基本設定。

4. 分類任務（實作範例）
   - 飛安報告分類（HFACS taxonomy）：示範 system prompt 列出候選類別、以 LLM 判斷並回傳分類結果與理由（PydanticOutputParser 範例）。
   - 詩詞分類：處理唐詩 & 宋詞檔案、解析格式、建立 Pydantic schema 做詩體 / 詞牌判別，並討論詞牌識別的挑戰（平仄、聲律等）。
   - 比較 Zero-Shot / Few-Shot / LLM-based 的優缺點。

5. HR：職缺抓取、工作描述抽取與履歷自動配對
   - 網頁抓取 → BeautifulSoup 清洗 → LLM 提取 job description 的 pipeline。
   - 非同步版本：用 aiohttp + LCEL 將網頁擷取封裝為 async runnable，加速大量 URL 抓取。
   - 履歷提取：示範 python-docx 與 pypdf 的同步與非同步封裝（DocxExtractor、PdfExtractor），並說明在 async 環境用 asyncio.to_thread 的必要性。
   - 匹配流程：格式化 job description 與 resume，使用 Pydantic parser 回傳 Yes/No + reason。

6. 內容一致化與輸出格式化
   - Template-driven output：示範如何強制 LLM 回傳符合特定模板（例如履歷欄位輸出），確保下游系統能穩定解析與接收。

7. 第三方 LLM API 整合
   - Perplexity 與 DeepSeek 範例：說明如何把 Chat Prompt 轉換成各家 API 所需的 messages 格式，並示範 pipeline 包裝方法（messages_2_perplexity、DeepSeek client 示例）。
   - 注意：實際整合時需自行處理 base_url、api_key 與 responses 格式差異。

8. MLflow 基礎（Part 1）
   - 用 Iris + LogisticRegression 做範例訓練、在 local MLflow Tracking server 紀錄參數、metrics、artifact（模型）。
   - 示範 model registry / 版本查詢 / alias（別名）／tags 的使用與載入模型的範例。

---

## 🛠️ 技術棧（Tech Stack）

- Python 3.10+  
- LangChain / langchain_core / langchain_community  
- OpenAI-compatible client (ChatOpenAI) 或其它模型提供者封裝  
- Embeddings：HuggingFaceEmbeddings（BAAI/bge-m3 範例）  
- Vector DB：Qdrant (langchain_qdrant)  
- 非同步 HTTP：aiohttp  
- 文檔處理：python-docx、pypdf  
- 解析器：PydanticOutputParser / StrOutputParser  
- 追蹤系統：MLflow (local tracking server)  
- 其他：BeautifulSoup、requests、pandas、scikit-learn、qdrant-client

---

## 📦 先決條件與環境設置（快速上手）

1. 需要的環境變數（視範例使用而定）：
   - OPENAI_API_KEY / PERPLEXITY_API_KEY / DEEPSEEK_API_KEY
2. 建議安裝（部分範例）：
   - pip install langchain langchain-core langchain-community openai qdrant-client langchain-qdrant python-docx pypdf aiohttp mlflow scikit-learn pandas bs4 requests
3. 如果要跑 MLflow Tracking server（本機模擬）：
   - mlflow server --host 127.0.0.1 --port 8080
   - notebook 中需設定 mlflow.set_tracking_uri("http://127.0.0.1:8080")

---

## ▶️ 快速執行建議（Run Tips）

- 先準備好 API keys 與必要模型權限（若使用本地或雲端大型模型亦然）。
- 若只想跑「風格提取 + 生成」範例：建立 Qdrant collection → 加入示例 documents → 執行 style_pipeline 即可。
- 若要大量抓取網站，建議使用 notebook 提供的 async_parsing_process + LCEL pipeline（可同時併發多個 URL）。
- 在做文件解析（.docx/.pdf）時，若在 async 環境請使用 notebook 中的 Runnable 封裝（DocxExtractor / PdfExtractor），以 avoid blocking event loop。

---

## ⚠️ 注意事項與最佳實務

- LLM 作為分類器/評審（LLM-as-a-Judge）方便但並非完美：對於關鍵安全或法務決策，仍建議加入人工覆核與明確的標準答案（Ground Truth）驗證。  
- 非同步 I/O 與同步檔案讀取混用時要特別小心：python-docx 與 pypdf 為同步函式庫，請以 asyncio.to_thread 包裝。  
- 使用 embeddings +檢索時，index 的品質（分片大小、chunking、embedding model）會強烈影響 downstream 的 Faithfulness 與 Relevancy。  
- 權限與隱私：抓取外部網站或處理履歷資料時，請注意法律與隱私合規（個資保護）。

---

## 🧪 練習題與延伸挑戰

- 練習一：把「掄語」範例改成模仿李白風格，透過 few-shot 範例與 style extraction pipeline 驗證結果一致性。  
- 練習二：嘗試將詞牌辨識改為基於音韻（平仄）特徵，實作一個把漢字轉為拼音 / 聲調的 preprocessor，再以相似度比對詞牌樣式。  
- 練習三：將職缺抓取流程擴展為批次執行（多 URL），並把 job description 與 resumes 全部上傳到向量資料庫，做大規模匹配測試。  
- 練習四：把飛安分類任務加入少量人工標註資料，嘗試比較 Zero-Shot 與 Few-Shot 與小量微調（fine-tune）的效能差異。  
- 練習五：將本地 MLflow 執行改成 remote artifact store（S3）與遠端 tracking server，模擬團隊化部署流程。

---

## 參考與延伸閱讀

- LangChain & LangChain Community / core 文件  
- Qdrant 官方文件與 langchain_qdrant 範例  
- MLflow 官方文件（Tracking、Model Registry）  
- HuggingFace Embeddings、BAAI/bge-m3 model docs

---

如果你想要我把本 notebook 的某個區塊拆成獨立的範例程式碼或教學（例如：完整的 style-extraction pipeline、非同步網頁抓取批次示例、或是 MLflow demo repo 結構），告訴我你想要哪一部分，我可以把它整理成獨立的 README + 執行步驟。