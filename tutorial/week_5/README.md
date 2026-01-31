# 圖像描述與多模態應用實作 (Image Captioning) — Week 5

本專案為多模態（Image → Text）教學筆記，聚焦於如何使用 LangChain 與多模態 LLM 建構圖片描述、延伸創意應用，以及把推論流程包裝成可部署的服務。

---

## 📅 專案資訊

- 初版日期：2026.01.31  
- 筆記版本：v1.0.0  
- Notebook：`notebook.ipynb`（Week 5）

---

## 🚀 核心重點

1. 什麼是 Image Captioning  
   - 將圖片內容自動轉成語言描述（例如：「一隻黑色的狗在草地上奔跑」），是多模態 AI 的基礎任務之一，常用於輔助檢索、可及性、內容審查與互動式代理。

2. 多模態模型輸入格式與圖片傳輸方式  
   常見三種方式：
   - URL：最簡單、效能好（避免 Base64 膨脹）。但需圖片可公開存取。
   - Base64（嵌入 JSON）：通用於只支援文字/JSON 的 API，資料量會膨脹約 33%。
   - multipart/form-data（檔案上傳）：效能最佳、類似標準檔案上傳，適合大型檔案與高頻請求。

3. LangChain 中的 Prompt 設計模式  
   - 使用 HumanMessagePromptTemplate、ImagePromptTemplate 與 SystemMessagePromptTemplate 組合多模態訊息。  
   - 常見流程：image_path → Base64 → build image prompt → chat prompt → model（LLM）→ 文字輸出。

4. 延伸應用：AI 趣味人格占卜（Prototype）  
   - 將 Image Captioning 延伸為創意應用，將多張圖片作為輸入，請模型基於視覺元素（色彩、構圖、主題）做趣味性的人格側寫。  
   - 明確聲明為娛樂性質，避免臨床或科學化解讀。

---

## 🧩 常見 Pipeline 模式（範例概念）

- 直接用 URL：
  - HumanMessage 包含 text + image_url → Chat LLM

- Base64 嵌入（JSON）：
  - image_path -> 轉 Base64 -> HumanMessage（type=image / url=data:image/...）→ Chat LLM

- Chain 化（可串接於後端）：
  - image_to_base64 chain | build_image_prompt | chat_prompt_template | model | StrOutputParser

這類 pipeline 易於包裝成 API（Flask / FastAPI / LangServe）或前端原型（Streamlit）。

---

## 🛠️ 提示設計要點（Prompt Engineering）

- System Prompt：
  - 明確指定任務（例如：「你是多模態助理，請以繁體中文描述圖片內容」），並給出回應風格與限制（長度、所需細節層級）。
- Human Prompt：
  - 同時包含文字指令與 image prompt（ImagePromptTemplate / image_url 欄位）。
- 多張圖片輸入：
  - 將多張 image prompt 以列表形式放入 HumanMessage，或在 System 指示中要求綜合分析。
- 語言：
  - 若模型在英文上表現優於其他語言，可考慮以英文作為 internal prompt，再輸出繁體中文（或直接要求繁體中文）。

---

## ⚠️ 倫理與限縮聲明（重要）

- 人格側寫 / 趣味占卜屬於娛樂用途，**非臨床或科學診斷**。不得用於醫療、法律或身份辨識等高風險場景。  
- 對隱私、偏見與不當推論保持警覺，避免做出基於外貌或敏感屬性的斷定。

---

## 🧰 技術棧（Tech Stack）

- LangChain (langchain_core / langchain_openai)  
- 多模態 LLM（例如 OpenAI ChatOpenAI 多模態模型 / gpt-4o-2024-05-13）  
- 圖像處理：Pillow (PIL)  
- Base64 編碼：標準 library io / base64  
- 部署：Flask / FastAPI / Streamlit / LangServe  
- 額外工具：fal-client (Florence-2)、WD-14 tagging、FAISS（向量庫）  

---

## 📁 檔案結構（示意）

app/
├── app_flask.py               # Flask 後端範例
├── app_server.py              # LangServe / 推論服務
├── app_streamlit.py           # Streamlit 原型介面  
tutorial/week_5/                # Notebook 與示範圖像資源
├── notebook.ipynb
├── image_source/               # 範例圖片集合

---

## 🧭 部署與快速上手（簡要）

1. 安裝依賴（示意）
   - pip install langchain-openai pillow fal-client streamlit uvicorn fastapi

2. 設定環境變數（範例）
   - export OPENAI_API_KEY=your_key

3. 本地運行 Streamlit 原型
   - streamlit run app_streamlit.py

4. 呼叫 Flask API（測試）
   - POST /generate 帶入 image（Base64 或 URL）與 question，回傳 JSON 結果

---

## 🔎 其他工具與參考

- WD-14（ACG-oriented image tagging）  
  - 適合動畫 / 同人圖標籤化，提供本地化 tagging 工具（wd14-tagger-standalone）。
- Florence-2（fal.ai）  
  - 開源/商業混合的視覺模型，支援 OCR、caption 等多種任務，可透過 fal-client 呼叫。
- 本地向量庫測試  
  - 使用 HuggingFace Embeddings（BAAI/bge-m3）與 FAISS 驗證檢索與多模態資料流整合。

---

## 結語

本週重點在於把「圖片」視為可被 LLM 處理的第一類資料：設計合適的輸入格式（URL / Base64 / multipart）、構建多模態 prompt、並把整個流程包成可測試與可部署的 pipeline。完成後，你應能把 Image Captioning 作為基礎能力，進一步延伸到互動式應用（如趣味人格占卜、圖片問答、視覺輔助搜尋等）。

如需範例程式片段或部署樣板（Flask / Streamlit），請打開同目錄下的 notebook.ipynb 參考實作步驟與範例程式碼。祝你玩得開心，做出好玩的多模態應用！