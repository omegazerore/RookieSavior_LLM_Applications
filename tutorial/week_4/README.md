# MLflow 與 LangChain 實作（二）：可追蹤的 LLM 實驗與部署（Week 4）

本專案為 Week 4 教學筆記，示範如何將 LangChain / OpenAI 類 LLM 工作流整合到 MLflow，並延伸到模型封裝、遠端服務（LangServe）、工具呼叫與 RAG 應用。重點放在「可追蹤性（observability）」、「可重現實驗」與「可部署化」的實作範例。

---

## 📅 專案資訊

- **週次**：week_4  
- **Notebook**：`notebook.ipynb`  
- **最後更新**：2026.01.31  
- **筆記版本**：v1.0.0

---

## 🚀 本週核心學習目標（你將能做什麼）

- 使用 MlflowCallbackHandler 自動將 LangChain / LLM 的輸入、輸出與執行 trace 記錄到 MLflow。  
- 開啟與理解 mlflow.langchain.autolog()（Autolog）機制：在 MLflow UI 看到 prompt、response、span/trace 與耗時。  
- 建構 Reflection → Revision 多階段 Pipeline，並在 MLflow 內追蹤每一階段（用於寫作回饋或多模型修訂流程）。  
- 使用 MLflow 的 ModelSignature 與 pyfunc 封裝並上傳可重用的 LangChain Pipeline（作為 Artifact / Registered Model）。  
- 利用 LangServe 建立簡單 API，並透過 RemoteRunnable 在客戶端遠端呼叫模型。  
- 設計帶工具（tools）、網路搜尋（WebSearch）與外部檢索（FAISS retriever）的 ChatBot 流程，包含 ToolMessage 的呼叫與回填流程。  
- 使用 LLM 進行程式碼生成並執行（code generation → exec pipeline），結合 Pydantic output parser 以取得可解析的結構化輸出。  
- 將整個實驗流程納入 MLflow 追蹤：artifacts、source code、input_example 與 signature。

---

## 🔑 核心重點摘要

1. MLflow 跟 LangChain 的整合模式  
   - MlflowCallbackHandler：手動將 run_id 綁定到 chain / model callbacks，能記錄多次 invoke 到同一 run。  
   - Autolog（mlflow.langchain.autolog()）：自動攔截 LangChain traces/spans，將執行結構寫入 MLflow tracing 與 run artifacts，而不是產生 .json 檔案。

2. Reflection + Revision Pipeline（多階段可追蹤流程）  
   - 利用兩個不同角色/模型（feedback model 與 revision model）實作 teacher-student 流程。  
   - 使用 PydanticOutputParser 或 StrOutputParser 做結構化輸出，方便後續驗證與上傳。  
   - 每個階段都可透過 MlflowCallbackHandler 追蹤，並在 MLflow UI 比對不同版本的結果。

3. 上傳模型為 Python script（pyfunc）與 ModelSignature  
   - 用 mlflow.pyfunc.log_model 上傳模型檔案（或包含源碼的 artifact），並指定 ModelSignature（inputs/outputs schema）。  
   - 指定 signature 可以讓後續部署/驗證時自動做輸入型別檢查，提升可移植性與可用性。

4. LangServe 與 RemoteRunnable  
   - 將模型封裝成後端 API（LangServe），並用 RemoteRunnable 在客戶端以相同 runnable 介面呼叫。  
   - 支援同步/非同步 stream（astream）與逐步輸出（增量回傳）以利 UI 串流顯示。

5. ChatBot 組成：PromptTemplate、MessagesPlaceholder、外部記憶、工具（Tools）  
   - Stateless vs Stateful：是否把對話歷史回填到 prompt 決定記憶行為。  
   - Tools（@tool / StructuredTool）：綁定計算、檢索、WebSearch 等功能，並透過 ToolMessage 回填工具結果再做一次 invoke。

6. WebSearch 與資料擷取策略  
   - 使用 WebSearch tool 取得即時資料，並透過 include 欄位抓取 action.sources 等 metadata。  
   - 設計來源過濾、地理位置設定與推理強度（reasoning effort）來控制工具行為與成本。

7. 程式碼產生與執行（Code-as-Tool）  
   - 使用 LLM 生成 Python 程式片段（嚴格回傳可執行程式），再由本地安全環境 exec 執行以取得結構化結果（local_vars）。  
   - 結合 Pydantic output parser 以回傳可驗證的結構化資料。

8. RAG / FAISS 檢索整合（簡要）  
   - 將向量檢索（FAISS）包成 retriever tool，並以 StructuredTool / PoemRetrieverArgs 等 schema 暴露給 LLM 呼叫。  
   - 結合檢索結果與模型回應，形成可檢索的 ChatBot。

---

## 🛠️ 技術棧（Tech Stack）

- MLflow（Tracking Server、artifacts、pyfunc、Model Registry）  
- LangChain / langchain_core / langchain_openai（PromptTemplate、ChatPromptTemplate、Runnables）  
- LangServe（RemoteRunnable）  
- OpenAI / ChatOpenAI（gpt-4o-mini / gpt-4o 等）  
- FAISS + HuggingFaceEmbeddings（BAAI/bge-m3）作為向量檢索  
- requests、BeautifulSoup（網頁擷取/解析）  
- pandas、pydantic（資料處理與 schema 驗證）  
- Streamlit（Demo + Callback 追蹤整合）

---

## 快速上手（Quick Start）

1. 啟動 MLflow tracking server（範例）：
```bash
mlflow server --host 127.0.0.1 --port 8080
```

2. 設定 notebook 中的 tracking URI 與實驗：
```python
tracking_url = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(tracking_url)
mlflow.set_experiment("Week-4")
```

3. 使用 MlflowCallbackHandler 追蹤 LangChain 呼叫：
- 建立 run → 傳入 run_id 到 MlflowCallbackHandler → 把 callback 加到 ChatOpenAI / chain 上，invoke 會自動記錄。

4. 啟用 Autolog（若要在 MLflow UI 中直接查看 traces）：
```python
mlflow.langchain.autolog()
```

5. 若要上傳模型為 pyfunc（含 signature 與 input_example）：
- 建立 ModelSignature（Schema / ColSpec），呼叫 mlflow.pyfunc.log_model 並提供 python 檔或 model 路徑與 input_example。

6. LangServe 測試：
- 啟動 LangServe 後可透過 requests.post 到 /openai/invoke 或其他 endpoint 呼叫模型；用 RemoteRunnable 指向該 URL 進行遠端呼叫。

---

## 實作提示與注意事項

- Autolog 與 callback 的差異：Autolog 會自動攔截並把 traces 寫到 MLflow tracing；MlflowCallbackHandler 則是你顯式指定 run_id，方便把多次 invoke 聚合到同一 run。兩者可互補，但使用時要注意不要重覆紀錄或造成過多 artifacts。  
- ModelSignature：上傳模型時明確指定 signature 可避免部署時輸入格式錯誤，尤其當 pipeline 需求非純文本（DataFrame、多欄位）時。  
- 安全性：執行 exec / eval 必須在受控環境下（或沙箱化），避免執行不受信任的程式碼。  
- Tool 執行流程：LLM 會回傳 tool_calls → 你必須在 server 端執行相對應工具並把結果以 ToolMessage 回填給模型，再做一次 invoke 以得到最終回答。  
- WebSearch 成本與隱私：WebSearch 可能會產生額外工具費用與涉及個資的外部請求，設計時務必權衡與過濾來源。  
- 可觀測性：把 prompt、input variables、response、執行時間、tool_calls 等資料都記錄在 MLflow，方便後續 debug 與實驗比對。

---

## 範例流程（簡要）

1. 建立 MLflow run → 建立 MlflowCallbackHandler(run_id)  
2. 建立 ChatOpenAI（callbacks=[mlflow_cb]）與 PromptTemplate / Chain  
3. invoke chain（多次）→ Mlflow 會記錄每次 input/output、table artifact（HTML）與 trace  
4. 若啟用 autolog：在 MLflow UI 的 tracing/trace view 中檢視 nested spans 與步驟耗時  
5. 建構 Reflection → Revision Pipeline，將整個 pipeline log 為 pyfunc model 並上傳到 Model Registry  
6. 部署 LangServe 並透過 RemoteRunnable 進行遠端推論測試

---

## 延伸閱讀與下一步建議

- 在 MLflow UI 中比較不同 run 的 prompt 差異、response fidelity 與耗時，作為 prompt engineering 的量化依據。  
- 把 RAG（FAISS retriever）與 Reflection/Revision pipeline 結合，評估 retrieval 對最終生成品質的影響。  
- 把 code generation + exec 的流程改為更安全的沙箱（如 Docker / lambda microservice）以降低風險。  
- 使用 Streamlit 建立 Demo，並將 Mlflow 追蹤連結到 UI（便於展示與驗證）。

---

若想直接執行 notebook，請先準備好：
- MLflow server 可用的 endpoint（範例：127.0.0.1:8080）  
- OpenAI API key 或對應的 LLM 連接憑證  
- 必要的套件（mlflow、langchain、langchain_openai、pydantic、faiss、requests、bs4、streamlit 等）

祝你實驗順利！如果需要，我可以幫你把其中某個流程（例如 Reflection→Revision pipeline 或 LangServe 範例）拆成可執行的示例程式碼或 Streamlit demo 範本。