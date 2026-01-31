# LLM + LangChain 入門教學 (Week 1)

本專案為 LLM（大型語言模型）與 LangChain 實作的 Week 1 教學筆記，從概念、提示詞工程到檢索與實作範例，帶你一步步上手常見的開發流程與實務技巧。

---

## 📅 專案資訊

- 初版日期：2026.01.31  
- 筆記版本：v1.0.0  
- Notebook：`notebook.ipynb`（Week-1：LLM + LangChain 入門）

---

## 🚀 本週學習重點（你將能學會什麼）

1. 理解 LLM 與 LangChain 的核心概念與模組化設計（LLM、PromptTemplate、Chain、Agent、Retriever 等）。  
2. 實際呼叫常見模型（OpenAI / Google Gemini / 本地模型），掌握基本參數（如 temperature、max_tokens）。  
3. 設計與測試不同風格的 Prompt（系統提示、人格化提示、指令式提示），並用 LangChain 的 PromptTemplate 與 ChatPromptTemplate 結構化管理。  
4. 使用 Pydantic 與 OutputParser 強制模型輸出結構化格式，利於後續程式處理與前端呈現。  
5. 建立簡單的 Gradio 應用（包含格式化輸出、聊天介面與 LaTeX 數學公式顯示）。  
6. 認識檢索技術（BM25、Wikipedia Retriever、Ensemble Retriever）與中文分詞／BPE 的要點，了解實務上如何混合字面與語意檢索。  
7. 學會在運行時（runtime）調整檢索器參數以達到更精準的檢索行為。

---

## 🧠 核心概念速覽

- 模組化抽象（Modular Abstractions）  
  LangChain 以「積木」化組件（LLM、Prompt、Retriever、Chain、Agent）組裝流程，降低重複樣板並促進可維護性。

- Prompt 的分層：System → Human → (Optional) Assistant / Tools  
  系統提示用於定義角色與行為風格；人類提示帶入使用者輸入；助手或工具回傳可整合進對話中。

- 結構化輸出（Pydantic OutputParser）  
  使用 Pydantic 定義回應結構，並把 format instruction 提給模型，使輸出可解析、可驗證、適用於自動化流程。

- 檢索與混合檢索（BM25 + Embedding）  
  BM25 屬於關鍵字/字面匹配，Embeddings 支援語意檢索。Ensemble Retriever 將兩者結合以改善召回與精準度的平衡。

---

## 🛠️ 技術棧（Tech Stack）

- LangChain（核心模組、PromptTemplate、Chains、Retrievers）  
- LLM 提供者：OpenAI（ChatOpenAI）、Google Gemini（ChatGoogleGenerativeAI）、本地模型（視情況）  
- Embeddings / Tokenizers：Hugging Face tokenizers、transformers、jieba（中文）、fugashi（日文）  
- 檢索：rank_bm25（BM25）、WikipediaRetriever、EnsembleRetriever  
- 結構化輸出：Pydantic、LangChain 的 PydanticOutputParser / OutputParsers  
- 前端互動：Gradio（聊天介面、LaTeX 支援）  
- 檔案輸出：FPDF（範例將結果寫成 PDF）  
- 語言：Python 3.10+

---

## 📌 重點章節 / 實作要點

1. 呼叫模型與環境設定  
   - 準備 API Key、初始化模型客戶端、理解 temperature 與 token 上限對回應的影響。  
   - 範例包含 OpenAI 與 Gemini 的初始化與錯誤處理建議。

2. Prompt 設計與 ChatPromptTemplate  
   - 系統提示（System Message）決定角色與風格；具體範例示範如何模擬不同人物（Gordon Ramsay、Donald Trump 等）。  
   - 使用 ChatPromptTemplate 管理多段式提示（system + human），並示範如何切換輸出語言（繁體中文）。

3. 結構化輸出（Pydantic + OutputParser）  
   - 定義 Pydantic BaseModel 來描述所需回應格式（例如 question/answer pair、bio/executive_summary/full_description）。  
   - 取得 format_instructions 並把它注入 prompt，使模型依指定格式回傳，最後用 OutputParser.parse() 解析結果。

4. 輸出格式控制與前端顯示  
   - 為商業化或自動化場景，強制輸出格式能降低解析錯誤、提升穩定性。  
   - Gradio 範例示範如何在 Chatbot 中支援 LaTeX（自定義 delimiters）以呈現數學公式或化學式。

5. Gradio 應用（從簡到進階）  
   - Basic：簡單文本輸入→模型回覆。  
   - Advanced：使用 Blocks 組合多欄位、範例、按鈕與輸出欄，並將 Pydantic 解析結果顯示在多個 Output 元件中。  
   - 作業範例為：設計一個能把數學輸出轉成 LaTeX 的 Pydantic OutputParser，並在 Gradio Chatbot 正確顯示。

6. 檢索（BM25 / Wikipedia / Ensemble）  
   - BM25：適合關鍵字匹配場景，介紹 k1、k 參數與詞頻、逆文檔頻率等概念。  
   - 中文處理：示範如何用 Hugging Face tokenizer 或 jieba 作預處理，將中文文本轉為適合 BM25 的 token 列表。  
   - Ensemble Retriever：說明如何把 BM25 與語意檢索混合，並以權重控制回傳來源比重。

7. Tokenization 與 BPE  
   - 解釋 BPE 在中文（從字級合併起始）與日文（需要形態分析器）上的差異，並提供實作上的注意事項與範例程式碼。

8. Runtime Configuration（運行時配置）  
   - 演示如何把檢索器參數標記為可配置（ConfigurableField），在執行階段動態改變 k、fetch_k 等，方便在前端或 CI 中調整行為。

---

## 💼 實務應用範例

- Company KB 檢索：混合 BM25（字面關鍵字）與 Embedding（語意）以回答內部查詢。  
- 教學輔助：自動生成造句、短文與 PDF 匯出（示範使用 FPDF 範例）。  
- 詩詞/文本創作：使用 BM25 作為 retrieval context，再以 LLM 產出指定格律（如五言絕句）的新作。  
- 作文評閱助教：利用 Pydantic 定義分析結構，Gradio 作為介面，提供即時回饋與改寫建議。

---

## 📝 作業（必做 — 練習題）

目標：在 Gradio 中實作一個能正確渲染數學公式的聊天介面，並使用 Pydantic OutputParser 自動把模型返回的數學內容格式化為 LaTeX。

要求：
1. 定義一個 Pydantic BaseModel，描述回傳的數學步驟與公式（必須為 LaTeX 格式）。  
2. 在 prompt 中注入 output_parser.get_format_instructions()，並要求模型以指定結構回覆。  
3. 建立 Gradio Chatbot，設定 latex_delimiters（示範 $$ / $ / ```math 等），能正確顯示區塊公式與行內公式。  
4. 提交時附上簡短說明文檔與範例輸出截圖（或短影片）。

提示：重點不是讓模型自己寫好 LaTeX，而是設計 prompt + OutputParser 的協作流程，確保輸出能被前端穩定解析。

---

## 推薦閱讀與資源

- LangChain 官方文件（最新 API）：https://docs.langchain.com  
- Hugging Face Tokenizers / Transformers 文件  
- rank_bm25（BM25 Python 實作）  
- Gradio 官方文檔（latex_delimiters、Blocks 範例）  
- Pydantic 官方文檔（模型驗證與序列化）

---

若你在實作過程中遇到任何問題（Prompt 設計不穩定、Pydantic 解析錯誤、Gradio 顯示問題或檢索結果不如預期），歡迎提出具體錯誤訊息與最小可重現範例，我可以協助你逐步排查與優化。祝學習愉快、實作順利！