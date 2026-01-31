# Image Generation & Agents 實作 (Week 7)

本專案為 Week 7 教學筆記，涵蓋以 OpenAI Image API 為核心的「文字生圖 / 圖像編輯」實作，以及將影像生成整合入 LangChain / Agent / Middleware 工作流的實務範例。

---

## 📅 專案資訊

- 初版日期：2026.01.31  
- 筆記版本：v1.0.0  
- Notebook：`notebook.ipynb`（tutorial/week_7 範例）

---

## 🚀 本週學習重點

- 理解並使用 OpenAI 的 GPT-Image 系列（gpt-image-1 / gpt-image-1.5）進行 Text-to-Image 與 Image-to-Image（Img2Img）  
- 熟悉 client.images.generate()、client.images.edit() 的常用參數（size、quality、input_fidelity、moderation 等）  
- 掌握把 base64 圖像與前端/Notebook 顯示、儲存的流程  
- 提升提示詞（prompt）工程能力：標籤式（Danbooru tags） vs 自然語言描述（Natural language）  
- 用 LangChain / LCEL 自動生成 prompt，串接影像 API 完成端到端流程（描述 → prompt → 影像）  
- 實作影像編輯（多張影像融合、局部修補／Mask inpaint）、內容審核與 Agent Guardrails（中介軟體）  
- 把模型執行與標記資料導入 MLflow 進行監控與追蹤

---

## 🧠 快覽：GPT-Image 系列要點

- 模型：gpt-image-1（最新家族包含 gpt-image-1.5）  
- 常用參數：
  - size：1024x1024 / 1536x1024 / 1024x1536 / auto
  - quality：low / medium / high / auto
  - moderation：auto / low（用於簡易內容過濾）
  - input_fidelity（影像編輯時用以控制與原圖相似度，需新版 SDK）
- 建議以英文 prompt 為主，能取得較穩定的風格與細節表現
- 注意：部分功能（如遮罩 inpaint）在某些版本上仍有品質或一致性問題（需實驗驗證）

---

## ✍️ 提示詞（Prompt）寫法要點

- 標籤式 Prompt（Danbooru tags）
  - 優勢：可快速疊加具體屬性（ex: 1girl, full-body, masterpiece）  
  - 侷限：不同模型對標籤理解不一致，需多次調校
- 自然語言 Prompt
  - 優勢：語意完整、可描述藝術風格、光影與情緒（提升質感）
  - 侷限：對非英語母語者可能較難寫出精準描述
- 實務建議：兩者可混用 — 用自然語言描述核心（場景/情緒/攝影術語），再以標籤補強細節

---

## 🔁 將 LLM 與 Image API 結合（LCEL / LangChain 範式）

- 流程概念：
  1. 以 Chat 模型（如 gpt-4o-mini）自動把「使用者描述」轉換成高品質的 image-prompt
  2. 把生成的 prompt 傳入 OpenAI image endpoint（client.images.generate / client.images.edit）
  3. 將回傳的 base64 圖像儲存為檔案或直接在 Notebook / 前端展示
- 好處：把提示詞工程自動化，利於批量生成、版本追蹤與 A/B 測試
- 可串接 MLflow 做 prompt 與生成結果的日誌紀錄（便於回溯與評估）

---

## 🖼️ 圖像渲染（Image-to-Image / Img2Img）

- 定義：以一張或多張參考圖 + prompt 生成新圖（可改風格、局部修改、風格融合）  
- 常見應用：風格轉換、廣告素材生成、合成海報、多圖融合（movie-poster）  
- 操作注意：
  - 多圖輸入時，API 會把參考圖合成一個生成方向，prompt 要明確表述合成目標
  - input_fidelity 用來調整生成與原圖的相似度（新版 SDK 支援）
  - 儲存與展示：從返回的 base64 解碼並寫檔，或直接用 HTML 嵌入顯示

---

## 🩹 局部修補（Inpaint / Mask）

- 使用遮罩（mask）指定要修改的區域，mask 必須有 Alpha 通道且與原圖同尺寸、格式
- 注意事項：
  - Mask 編輯品質因模型與提示詞而異，部分情況效果不佳（需多次試驗）
  - 檔案大小限制（通常 < 50MB）
  - 若遇到品質問題，嘗試調整 prompt、input_fidelity 或以多張輔助圖強化指引

---

## 🤖 Agent、Tools 與中介軟體（Middleware）

- 將影像生成 / 編輯流程封裝為 Agent 工具（Tool），可被 Agent 在 ReAct 循環中呼叫  
- 常見 Tool 類型：
  - Image generation tool（呼叫 images.generate）
  - Image edit / inpaint tool（呼叫 images.edit，支援 mask 與多圖）
  - Tagging / Guard tool（呼叫本地或外部標記服務檢查 NSFW / 風格標籤）
- Middleware 用途：
  - PII / 圖像內容過濾（PIIMiddleware、ImageContentFilter）
  - Model fallback（遇錯自動退回備用模型）
  - Context editing（自動裁剪長對話或工具輸出）
  - 自訂 hook（before_model、after_model、wrap_model_call 等）
- 實務建議：把圖像審核（tagging / moderation）放在 agent pipeline 前端，避免生成不當內容並節省成本

---

## 🛡️ 內容審核（Moderation）與 Guardrails

- OpenAI moderation（omni-moderation-latest）能同時處理文本與圖像（image_base64）
- Moderation 回傳：
  - flagged（是否可能違規）
  - categories / category_scores（各類別被檢測到之信心值）
  - category_applied_input_types（哪些輸入類型被辨識為違規）
- 建議作法：
  - 在 pipeline 中加入 moderation 步驟（生成前或生成後皆可，視風險而定）
  - 結合本地 tagger（如 WD14 類服務）做更細緻的屬性判定
  - 把審核結果與 MLflow 拿來做長期監測與警示

---

## ⚙️ 執行環境與監控

- 常用組件：
  - OpenAI SDK（建議更新到 1.97.0+ 以支援新參數）
  - LangChain / LCEL / LangServe（Agent 與 middleware）  
  - MLflow（監控 prompt、tagging、生成結果與模型呼叫）
  - 本地 tagger（WD14 / camie-tagger）做 NSFW 與 Danbooru 標註
- Runtime / Context：
  - Agent 執行時常會用到 Runtime（context、store、stream_writer），可做依賴注入與狀態管理
  - 建議在 tools 與 middleware 中妥善使用 runtime.store 保留使用者偏好或過去標註資料

---

## 🛠️ 技術棧（Tech Stack）

- OpenAI Image API（gpt-image-1 / gpt-image-1.5）  
- LangChain / LCEL / LangServe（Agent、Tool、Middleware）  
- LangChain-community / LangChain-core 工具與向量庫整合  
- MLflow（執行紀錄與實驗追蹤）  
- 本地 Tagger（WD14 類別）做影像標註與 Guardrails  
- Python、PIL、base64（圖像 IO 處理）  

---

## ⚠️ 注意事項與實務建議

- prompt engineering 是高頻調參項目：建議系統化紀錄 prompt 與結果（使用 MLflow）以便回溯  
- 圖像審核應納入 pipeline：先審核再呈現/儲存，降低風險與合規成本  
- 在大量請求時留意成本與速率限制（可加入 fallback 與合適的 quality 設定）  
- 若用於商業或公開產品，務必確認版權與來源授權（尤其是合成或融合第三方素材時）  
- Mask / Inpaint 功能在不同版本可能表現不一，開發時務必建立可回退的測試集與人工檢視流程

---

## 🔗 延伸閱讀（官方文件 & 範例）

- OpenAI Image Generation Guide: https://platform.openai.com/docs/guides/image-generation  
- OpenAI Images API Reference: https://platform.openai.com/docs/api-reference/images  
- OpenAI Moderation API: omni-moderation-latest（moderations endpoint）  

---

若要快速開始，建議先：
1. 更新 openai 套件到 1.97.0+  
2. 在 Notebook 中完成 credential 初始化（OPENAI_API_KEY）  
3. 建立簡單的 prompt → client.images.generate → base64 解碼顯示的最小流程，逐步加入 LangChain 與 MLflow 監控

祝你在本週的影像生成與 Agent 實作中，有豐富的嘗試與收穫！