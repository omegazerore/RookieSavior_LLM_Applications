# 語音互動實作：Whisper、Streaming、TTS 與語音點單系統 (Week 6)

本專案為語音處理與語音互動實作的 **Week 6 教學筆記**，示範如何整合 Speech-to-Text (STT)、大型語言模型 (LLM) 與 Text-to-Speech (TTS)，並以「語音自助點飲料系統」作為示範應用。

---

## 📅 專案資訊

- **初版日期**：2026.01.31  
- **筆記版本**：v1.0.0  
- **Notebook**：`notebook.ipynb`（tutorial/week_6）  

---

## 🚀 本章核心重點

- 了解 STT 與 TTS 的應用場景與差異：字幕、會議紀錄、語音助理、語音回覆等。  
- 學會使用 OpenAI Whisper 與 gpt-4o 系列音訊模型進行批次轉錄與串流轉錄（batch vs streaming）。  
- 掌握 prompt / context 如何影響轉錄準確度（尤其是專有名詞與方言詞彙）。  
- 熟悉即時錄音、音檔轉換（numpy → pydub → mp3/wav）與上傳 API 的實務流程。  
- 學會用 TTS（gpt-4o-mini-tts、tts-1 等）產生並串流播放語音（包含 pygame、pyaudio、LocalAudioPlayer 等播放方式）。  
- 範例實作：語音自助點飲料系統（STT → LLM 資料抽取（Pydantic）→ 計價 → TTS 回應）。  
- 評估低資源語言（如閩南語）之 ASR 可行性與挑戰（資料、書寫標準、後處理策略）。

---

## 🔎 STT（語音轉文字）要點

- 主要任務：Transcribe（同語言轉錄）與 Translations（轉錄並翻譯成英文）。  
- 檔案格式支援：mp3, mp4, mpeg, mpga, m4a, wav, webm。  
- 上傳限制：單檔約 25 MB（以 API 文件為準）。  
- Batch vs Streaming：
  - Batch（transcriptions.create）：上傳完整檔案後一次回傳結果，適合非即時處理。  
  - Streaming（stream=True）：上傳仍為一次性，但可在模型產出時逐步收到 partial 結果，適合需要即時回饋的應用（gpt-4o-mini-transcribe / gpt-4o-transcribe 支援）。  
- 模型選擇：gpt-4o-transcribe（品質較佳、成本較高） vs gpt-4o-mini-transcribe（較快、成本較低）。  
- Prompt / Context：在 domain-specific 或含方言詞彙時，加入 prompt 能顯著改善辨識（但不具可擴展性，需謹慎設計）。

---

## 🔊 TTS（文字轉語音）要點

- 推薦模型：gpt-4o-mini-tts（速度與彈性佳）以及 tts-1 等提供高品質語音。  
- 可透過 prompt 指示聲線、情緒、語速、口音、語氣等（instructions / voice 參數）。  
- 播放方式：  
  - Blocking 播放：pydub（簡單）、pygame（GUI / 非阻塞控制）。  
  - 低延遲串流播放：使用 wav/pcm 並搭配 pyaudio 或以 requests stream + wave + pyaudio 邊下載邊播放。  
- 實時串流：使用 chunk transfer encoding 或 SDK 的 streaming 支援，能邊生成邊播放，適合對話式 UI 或低延遲回覆。

---

## 🛠️ 開發實務（錄音、轉檔、串流）

- 錄音工具：sounddevice 可於本地錄音，取得 numpy 陣列音訊資料。  
- 音檔轉換：利用 pydub 將 numpy → mp3 或 wav，並以 BytesIO 直接傳給 API（避免先寫檔）。  
- 非阻塞錄音控制：使用 threading 或 callback 模式，避免錄音時阻塞主線程（可同時更新 UI / log）。  
- 若需跨平台與低延遲，建議使用 uncompressed WAV/PCM 作為串流格式。  
- 依賴：ffmpeg（處理多種音訊編碼）、pyaudio（低延遲播放）、pygame（簡易播放/整合 GUI）。

---

## 🧪 範例應用：語音自助點飲料系統

- 流程概念：
  1. 錄音 → 上傳給 Whisper（STT）  
  2. LLM（gpt-4o-mini / gpt-4o）處理文字抽取（使用 PydanticOutputParser 定義 schema）  
  3. 計價、回傳確認文字  
  4. TTS 生成語音回覆並播放（streaming 或 pre-generate）  
- 防呆設計：讓抽取 schema 回傳一個 `relevant` flag（若非飲料訂單則回報 False），避免模型在不相關輸入上 hallucinate。  
- 可擴充：整合 Langserve / Flask / Streamlit 作為服務端，支援前端/行動裝置連線。

---

## 🧾 閩南語 ASR（可行性評估）結論重點

- 問題：書寫標準不一、平行語料不足、下游接受度分歧。  
- 技術面：模型本身可學習任一一致標註的文字體系，關鍵是取得足夠且一致的平行語料。  
- 實務策略：可採「多輸出 + 後處理轉碼」或以台羅作為中介標註，再做漢字轉換。技術上可行但需高成本資料蒐集與社群共識。

---

## 🔁 Ollama 概論（本地開源模型）

- Ollama 能在本地管理與執行開源 LLM（例如 Llama、Mistral 等），適合需要資料不離機的場景（敏感資料、離線推理）。  
- 版本差異：  
  - `ollama.exe` / 本地執行環境：提供模型引擎與 CLI，實際執行模型需此環境。  
  - `pip install ollama`：Python 客戶端，用於與已啟動的 Ollama 伺服器互動（不包含執行環境本體）。  
- 可與 LangChain / Langserve 結合，作為本地 LLM 的後端。

---

## 🧩 技術棧（Tech Stack）

- OpenAI audio API：whisper-1、gpt-4o-mini-transcribe、gpt-4o-transcribe、gpt-4o-mini-tts、tts-1  
- Python lib：openai / AsyncOpenAI、sounddevice、pydub、pyaudio、pygame、requests、wave、numpy、threading  
- Prompt / parsing：LangChain、PydanticOutputParser（結構化輸出）  
- 可選本地 LLM：Ollama（local server + pip client）  
- 開發工具：ffmpeg（編碼/轉檔）、Streamlit / Flask / Langserve（服務化）

---

## ⚙️ 設計原則與最佳實務

- 儘量在 prompt 提供 domain context（專有名詞表、常見口語化寫法），但避免過度依賴 prompt 做大量修正。  
- 錄音與播放以非壓縮格式（wav/pcm）優先考量低延遲場景；傳輸時再視需求壓縮為 mp3。  
- 使用 schema 驗證（Pydantic）避免 LLM 幻覺導致非預期結果（例如把非訂單內容硬套成訂單）。  
- 以 streaming 為即時體驗優先，但要處理 partial result、斷句與最終結果合併邏輯。  
- 評估方言/低資源語言時，同時考慮資料策略（收集、後處理、使用者教育）與商業可行性。

---

## ⚡ 快速開始（Quick Start）

1. 先安裝必要套件（範例）：
   - pip install openai sounddevice pydub pyaudio pygame requests ffmpeg-python
2. 設定環境變數：
   - OPENAI_API_KEY 等（與 Ollama 時也需 OLLAMA_API_KEY / 本地 Ollama server）  
3. 執行 notebook：
   - 開啟 `tutorial/week_6/notebook.ipynb` 並依步驟執行 cell（包含 credential_init()）。  
4. 若要錄音測試：安裝 ffmpeg 並確保系統可存取音訊設備（麥克風／揚聲器）。

---

## 參考資源

- OpenAI Audio API 文件（Transcriptions / Translations / Speech）  
- ffmpeg 官方下載（跨平台音訊處理）  
- Ollama 官方網站與 CLI 文件  
- Pydantic + LangChain 範例（結構化輸出）  

---

如果你想要，我可以：
- 幫你把 notebook 的範例抽成可重複使用的函式庫模組（錄音、轉檔、上傳、解析、TTS 播放）。  
- 或者把語音點單 demo 改成 Web UI（Streamlit）示範。