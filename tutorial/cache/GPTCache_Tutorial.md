# GPTCache × LangChain × ChatOllama 完整教學

## 目錄

1. [什麼是 GPTCache？](#1-什麼是-gptcache)
2. [核心架構：四個元件](#2-核心架構四個元件)
3. [範例一：精確匹配（Exact Match）](#3-範例一精確匹配exact-match)
4. [範例二：語義相似度匹配（Similarity Match）](#4-範例二語義相似度匹配similarity-match)
5. [範例三：更換向量資料庫 — Qdrant](#5-範例三更換向量資料庫--qdrant)
6. [範例四：更換 Embedding 模型 — CLIP](#6-範例四更換-embedding-模型--clip)
7. [快取管理：重新載入、查看、清除](#7-快取管理重新載入查看清除)
8. [常見問題與踩坑記錄](#8-常見問題與踩坑記錄)

---

## 1. 什麼是 GPTCache？

**GPTCache** 是一個開源的 LLM 語義快取層。它的核心價值：

- **省錢**：相同或類似的問題不重複調用 LLM API
- **加速**：cache 命中時回應時間從秒級降到毫秒級
- **語義匹配**：不只精確匹配，還能識別「換句話說」的等價問題

```
用戶提問 → GPTCache 攔截 → 搜尋快取
                ↓ 命中              ↓ 未命中
           直接回傳快取答案      調用 LLM → 存入快取 → 回傳
```

---

## 2. 核心架構：四個元件

GPTCache 由四個可替換的元件組成，每個都影響快取的行為：

| 元件 | 角色 | 預設值 |
|---|---|---|
| `pre_embedding_func` | 從請求中**提取什麼**當作 cache key | `last_content`（取最後一則訊息） |
| `embedding_func` | 把 key **轉成什麼形式**去搜尋 | `to_embeddings`（原樣保留，不做向量化） |
| `similarity_evaluation` | **怎麼判斷**兩個 key 是否匹配 | `ExactMatchEvaluation`（字串完全相等） |
| `data_manager` | 資料**存在哪裡** | `MapDataManager`（記憶體 dict） |

### 2.1 `pre_embedding_func` — 決定快取粒度

```python
from gptcache.processor.pre import get_prompt, last_content

# get_prompt：只取 prompt 字串
# 輸入 {"prompt": "什麼是 AI？"} → 輸出 "什麼是 AI？"

# last_content：取 messages 最後一則的 content
# 輸入 {"messages": [{"content": "系統提示"}, {"content": "什麼是 AI？"}]}
# → 輸出 "什麼是 AI？"
```

### 2.2 `embedding_func` — 決定搜尋方式

| 函數 | 輸出 | 適合 |
|---|---|---|
| `to_embeddings`（預設） | 原始字串 | 精確匹配 |
| `SBERT().to_embeddings` | 384 維向量 | 語義相似度 |
| `Onnx().to_embeddings` | 動態維度向量 | 本地無網環境 |
| 自訂 CLIP | 768 維向量 | 高品質語義匹配 |

### 2.3 `similarity_evaluation` — 決定匹配寬嚴

| 評估器 | 邏輯 | 適用場景 |
|---|---|---|
| `ExactMatchEvaluation` | 字串完全相等 | 精確快取 |
| `NumpyNormEvaluation(enable_normal=True)` | cosine 相似度 ≥ 閾值 | 語義快取 |
| `KReciprocalEvaluation` | K-互惠最近鄰 + 距離 | 高精度語義 |
| `SequenceMatchEvaluation` | 加權多欄位比對 | 多輪對話 |

### 2.4 `data_manager` — 決定儲存後端

| 類型 | 寫法 | 持久化 |
|---|---|---|
| 記憶體 Map | 預設 | ❌ |
| SQLite + FAISS | `manager_factory("sqlite,faiss", ...)` | ✅ |
| SQLite + Qdrant | `manager_factory("sqlite,qdrant", ...)` | ✅（本地或遠端） |

---

## 3. 範例一：精確匹配（Exact Match）

最簡單的快取：只有**完全相同的問題**才會命中。

### 初始化

```python
from gptcache import cache
from gptcache.adapter.adapter import adapt
from gptcache.processor.pre import get_prompt
from langchain_core.messages import AIMessage

# 全部用預設值：字串比對 + 記憶體儲存
cache.init(pre_embedding_func=get_prompt)
cache.set_openai_key()
```

### 建立包裝函數

```python
llm = ChatOllama(model='deepseek-v4-pro:cloud',
                 base_url='https://ollama.com',
                 name='main', temperature=0)

def cached_invoke(prompt: str):
    def llm_handler(prompt):
        """Cache miss → 調用真正的 LLM"""
        return llm.invoke(prompt)

    def cache_data_convert(cache_data):
        """Cache hit → 把快取的字串還原為 AIMessage"""
        return AIMessage(content=cache_data)

    def update_cache_callback(llm_data, update_cache_func, *args, **kwargs):
        """Cache miss → 把 LLM 回傳的答案存進快取"""
        update_cache_func(llm_data.content)
        return llm_data

    return adapt(
        llm_handler,
        cache_data_convert,
        update_cache_callback,
        prompt=prompt,
        cache_obj=cache,
    )
```

### 測試

```python
# 第一次：cache miss，等待 LLM 回傳（~5 秒）
cached_invoke("What is OpenAI?")

# 第二次：cache hit，瞬間回傳（~200 ms）
cached_invoke("What is OpenAI?")

# 第三次：換個說法 → cache miss（字串不同！）
cached_invoke("Tell me about OpenAI")
```

> **關鍵限制**：精確匹配模式下，`"What is OpenAI?"` 和 `"Tell me about OpenAI"` 被視為**不同問題**。

---

## 4. 範例二：語義相似度匹配（Similarity Match）

使用 SBERT 將問題轉成語義向量，**換句話說也能命中**。

### 初始化

```python
from gptcache import cache, Config
from gptcache.embedding import SBERT
from gptcache.similarity_evaluation import NumpyNormEvaluation
from gptcache.manager import manager_factory

encoder = SBERT()  # all-MiniLM-L6-v2，384 維

data_manager = manager_factory(
    "sqlite,faiss",
    data_dir="./cache_data",
    vector_params={"dimension": 384},
)

cache.init(
    pre_embedding_func=get_prompt,
    embedding_func=encoder.to_embeddings,          # ← 語義向量化
    data_manager=data_manager,                     # ← 持久化儲存
    similarity_evaluation=NumpyNormEvaluation(enable_normal=True),  # ← cosine 相似度
    config=Config(similarity_threshold=0.6),       # ← 相似度 ≥ 0.6 即命中
)
```

### 測試

```python
# 第一次：cache miss
cached_invoke("What is OpenAI?")

# 第二次：換句話說 → cache HIT！（語義相似度 ≥ 0.6）
cached_invoke("Tell me about the company OpenAI")

# 第三次：完全不同的問題 → cache miss
cached_invoke("What is the capital of France?")
```

### `similarity_threshold` 調參指南

| 閾值 | 效果 |
|---|---|
| `0.9` | 非常嚴格，幾乎要同義改寫才會命中 |
| `0.7` | 中等，相關但不完全相同的問題會命中 |
| `0.5` | 寬鬆，主題相關就可能命中（小心誤匹配） |

---

## 5. 範例三：更換向量資料庫 — Qdrant

將 FAISS 替換為 Qdrant（支援本地檔案或遠端伺服器）。

```python
data_manager = manager_factory(
    "sqlite,qdrant",
    data_dir="./cache_data_qdrant",
    vector_params={
        "dimension": 384,
        "location": ":memory:",          # 本地記憶體模式（demo 用）
        "collection_name": "gptcache",
        "top_k": 1,
    },
)
```

> **注意**：qdrant-client ≥ 1.7 的 API 與 GPTCache 0.1.44 不完全相容，需 monkey-patch（詳見 notebook cell 24）。

---

## 6. 範例四：更換 Embedding 模型 — CLIP

使用自訂的 CLIP 模型（BAAI/AltCLIP，768 維）取代 SBERT。

```python
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from gptcache.embedding.base import BaseEmbedding

class CLIPEmbedding(BaseEmbedding):
    def __init__(self, model_name="BAAI/AltCLIP"):
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._dim = 768

    def to_embeddings(self, data, **_):
        if not isinstance(data, list):
            data = [data]
        inputs = self._tokenizer(data, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = self._model.get_text_features(**inputs)
        return emb.squeeze(0).numpy().astype("float32")

    @property
    def dimension(self):
        return self._dim
```

使用方式與 SBERT 完全相同，只需把 `embedding_func` 換成 `CLIPEmbedding().to_embeddings`，`dimension` 改為 `768`。

### 各 Embedding 模型比較

| 模型 | 維度 | 優點 | 缺點 |
|---|---|---|---|
| SBERT `all-MiniLM-L6-v2` | 384 | 輕量、快速、英文佳 | 中文較弱 |
| AltCLIP `BAAI/AltCLIP` | 768 | 中英雙語、語義強 | 模型較大 |
| OpenAI `text-embedding-ada-002` | 1536 | 最強語義 | 需 API key、花錢 |

---

## 7. 快取管理：重新載入、查看、清除

### 7.1 重新載入已持久化的快取

只要用**相同的 `data_dir`、相同的 embedding 模型、相同的 dimension** 重新初始化，GPTCache 會自動載入舊資料：

```python
data_manager = manager_factory(
    "sqlite,faiss",
    data_dir="./cache_data",          # ← 跟建立時一樣的路徑
    vector_params={"dimension": 384},
)
cache.init(...)  # 自動載入 sqlite.db + faiss.index
```

### 7.2 查看快取內容

```python
import sqlite3

conn = sqlite3.connect("./cache_data/sqlite.db")
cursor = conn.execute("""
    SELECT q.id, q.question, a.answer, q.create_on
    FROM gptcache_question q
    JOIN gptcache_answer a ON a.question_id = q.id
    ORDER BY q.id
""")
for row in cursor.fetchall():
    print(f"[{row[0]}] {row[3]}")
    print(f"    Q: {row[1][:80]}...")
    print(f"    A: {row[2][:80]}...\n")
conn.close()
```

### 7.3 清除快取

```python
import shutil
shutil.rmtree("./cache_data")
```

---

## 8. 常見問題與踩坑記錄

### Q1: `cache.init()` 後 LLM 調用沒有被快取？

**原因**：`cache.init()` 只初始化快取基礎設施，**不會自動攔截 LangChain 調用**。必須透過 `adapt()` 函數手動橋接。

### Q2: `ChatOllama` 不接受 `cache=` 參數？

**原因**：LangChain 的全域快取是 `langchain.llm_cache = InMemoryCache()`，不是 per-LLM 參數。GPTCache 則需要 `adapt()` 橋接。

### Q3: `TypeError: unhashable type: 'numpy.ndarray'`？

**原因**：使用了向量 embedding（如 SBERT）但 data manager 仍是預設的 `MapDataManager`（dict 儲存）。解決：改用 `manager_factory("sqlite,faiss", ...)`。

### Q4: `similarity_threshold` 報錯？

**原因**：`similarity_threshold` 是 `Config` 的屬性，不是 `cache.init()` 的參數。正確寫法：

```python
cache.init(..., config=Config(similarity_threshold=0.6))
```

### Q5: SBERT / CLIP 模型下載失敗（Permission denied）？

**原因**：`C:\Users\Ling\.cache\huggingface\` 目錄權限問題。解決：使用已預先快取的模型，或手動下載到該目錄。

### Q6: Qdrant 報錯 `'QdrantClient' object has no attribute 'search'`？

**原因**：qdrant-client ≥ 1.7 把 `search()` 改名為 `query_points()`，GPTCache 0.1.44 尚未適配。解決：使用 monkey-patch 橋接新舊 API（詳見 notebook cell 24）。

### Q7: `CPU times` > `Wall time` 是什麼意思？

**原因**：多核平行運算。CPU time 是所有核心的總工時，Wall time 是真實流逝時間。當 CPU time > Wall time，代表 PyTorch / FAISS 正在多線程平行處理（這是好事，代表 cache hit 很快）。

---

## 總結：選擇你的 GPTCache 配置

| 場景 | pre_embedding_func | embedding_func | similarity_evaluation | data_manager |
|---|---|---|---|---|
| 最簡單精確快取 | `get_prompt` | 預設 | 預設 | 預設 |
| 語義相似度（本地） | `get_prompt` | `SBERT().to_embeddings` | `NumpyNormEvaluation` | `"sqlite,faiss"` |
| 語義相似度（Qdrant） | `get_prompt` | `SBERT().to_embeddings` | `NumpyNormEvaluation` | `"sqlite,qdrant"` |
| 高品質語義（CLIP） | `get_prompt` | `CLIPEmbedding().to_embeddings` | `NumpyNormEvaluation` | `"sqlite,faiss"` |
| 多輪對話快取 | `last_content` | `SBERT().to_embeddings` | `NumpyNormEvaluation` | `"sqlite,faiss"` |
