# Agent 實作範例：自動化有聲書生成 (Week 8)

本專案示範如何建構一個以 Agent 為核心的「故事 → 圖像 → 語音」自動化流程，包含工具封裝、檔案化中介、以及以 LangGraph 實作的 Plan-Action 代理人範例。適合用來做互動式有聲書生成、教學示範或多步驟任務自動化。

---

## 📅 專案資訊

- 初版日期：2026.01.31  
- 筆記版本：v1.0.0  
- Notebook：`notebook.ipynb`（Week 8）

---

## 🚀 核心重點

### 一、整體流程（Story → Image → Audio）
- 由草稿或既有 .txt 作為輸入，依序產生：
  1. 生成並儲存「故事頁面 (.txt)」
  2. 根據該頁文本生成並儲存「圖片 (.png)」(base64 解碼)
  3. 根據該頁文本生成並儲存「語音 (.mp3/.wav)」(base64 解碼)
- 所有步驟的中間產物均以檔案保存，以減少重複傳輸與 Token 消耗，並便於後續重用與審查。

### 二、ToolTemplate：對 Langserve REST API 的統一封裝
- 實作一個通用的 ToolTemplate（繼承 BaseTool），負責：
  - Pydantic 驗證輸入格式（PydanticOutputParser）
  - 前處理 / 後處理（input/output processors）
  - 呼叫 Langserve REST endpoint（錯誤處理、回傳格式檢查）
  - 將產出寫入檔案並回傳檔名（若有指定 filename）
- 三個已封裝好的工具：
  - Story-generation-tool（產生故事並儲存 .txt）
  - Image-generation-tool（產生 image base64 並儲存 .png）
  - Audio-generation-tool（產生 audio base64 並儲存 .mp3/.wav）

### 三、檔案化交換（節省 Tokens 的實務技巧）
- 使用檔名（而非整段文字）作為不同工具間的狀態傳遞介面：
  - 將產出寫成檔案，下一個工具只需讀取檔案內容即可
  - 以 I/O 換取更低的 API 呼叫成本與更高的可重現性
- 常用 helper：
  - export_to_txt / read_from_txt / read_from_list_of_text
  - export_to_image（base64 解碼）/ export_to_audio（base64 解碼）

### 四、影像與語音處理重點
- 圖像：支援將先前圖像編碼為 base64（供 img2img 使用），但在示例中常採用基於文本的 prompt 串接以提高可控性與一致性。
- 語音：以 TTS 產生的 base64 字串解碼後輸出成 .mp3 或 .wav。

---

## 🧠 LangGraph 與 Plan-Action Agent（Week 8 重點）

### LangGraph 簡介（摘要）
- 圖形導向框架，用於構建有狀態的多步驟工作流（節點 = 智能體 / 工具，邊 = 流程/條件轉移）。
- 優勢：狀態管理、任務回溯、動態流程切換、多智能體協作。
- 適用場景：長期任務、需要中間檢查點與重試策略的應用。

### Plan-Action Agent 架構要素
- Planner：把使用者目標拆成有序步驟（Plan）
- Executor（agent）：執行 plan 的第一個步驟，回傳觀察結果並累積 past_steps
- Replanner：根據 past_steps 與執行結果更新或結束計劃
- Stop 判斷（should_end）：根據 state 決定是否結束或繼續循環

### LangGraph 工作流實作概要
- 定義 State 型別（PlanExecute）以儲存：input / plan / past_steps / response
- 將三個主要節點加入 StateGraph：
  - planner → agent → replan（並以條件邊決定回到 agent 或 END）
- 支援 checkpoint 與 replay（可用 InMemorySaver 或外部 checkpointer），可查詢與回放歷史 state。

---

## 🛠️ 技術棧（Tech Stack）

- LangChain / langchain_core  
- LangGraph（狀態圖）  
- Langserve（自建 REST 服務，用於 story/image/audio 生成功能）  
- OpenAI / ChatOpenAI（或其他 LLM）  
- Pydantic / PydanticOutputParser（輸入驗證）  
- Pillow、base64（圖片 I/O）  
- Python（async 支援）  

---

## ⚙️ 執行與快速上手

1. 啟動 Langserve 並確保 endpoint 可用（示例使用：http://localhost:8080）
   - /story_telling/invoke
   - /image_generation/invoke
   - /audio_generation/invoke
2. 設定環境變數（範例）：
   - OPENAI_API_KEY
3. 執行 notebook（或匯入 ToolTemplate 與 tool 實例）：
   - 呼叫 agent.invoke() 或 agent.ainvoke() 送出生成請求
4. 檔案輸出：
   - 範例儲存路徑： `tutorial/week_8/story_test`（同一 base filename 產生 .txt / .png / .mp3）

---

## 🔎 範例測試（Notebook 範例）
- 使用 requests 呼叫本地 Langserve 測試：
  - 生成第一頁故事 → 取得 story_json['output']
  - 以 story 產生圖片 → response_image.json()['output']['image_base64']
  - 以 story 產生語音 → response.json()['output']（base64）
- 以 Agent 一次性請求生成多頁範例（例如 4 頁故事），Agent 會依序呼叫三個工具並將檔案儲存。

---

## 🧩 觀察與優化建議（實務經驗）
1. 加強 State 使用（避免重複查詢）  
   - 在 execute_step 中先檢查 past_steps，若已有可用結果則跳過相同查詢。
2. 提早結束條件（Dynamic Stop）  
   - 若 agent 回傳高置信度的明確答案，可直接切換到 END，減少不必要循環。
3. 合併與優化搜尋（若使用 websearch）  
   - 將多個相似查詢合併成單一明確查詢以減少資源浪費。
4. 檔名與版本化策略  
   - 為避免覆寫並方便回溯，建議使用固定命名規則或加入時間戳/版本號。
5. 增加錯誤與重試機制  
   - ToolTemplate 對 Langserve 呼叫已含基本錯誤處理，建議補上更細緻的重試與退避策略（exponential backoff）。
6. prompt 與輸出格式結構化  
   - 在 planner / replanner / agent prompt 中採用結構化格式（JSON/YAML）能降低解析錯誤與格式漂移風險。

---

## ✅ 小結
- 本週重點整合了「工具化呼叫 Langserve」與「以檔案為中介的低成本資料傳遞」，並示範如何結合 LangGraph 建構具狀態的 Plan-Action 代理人。  
- 此架構適合需要可回溯、可重試、多步驟協作的應用場景（例如互動有聲書、長程任務代理、教學平台等）。  
- 下一步建議：完善容錯、引入檢定/評估流程以及在 production 環境中落地的部署測試。

--- 

若要複製或延伸範例，建議先確認本地 Langserve endpoints 可用並設定好 OpenAI（或替代 LLM）的 API Key。