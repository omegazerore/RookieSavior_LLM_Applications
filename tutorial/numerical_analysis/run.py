import os
from textwrap import dedent

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.agents.middleware.tool_retry import ToolRetryMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware
from langchain_core.messages import HumanMessage

from initialization import credential_init
from tutorial.numerical_analysis.tools import SchemaTool, PandasTool, Context
from src.path.path_definition import get_project_dir

credential_init()


AGENT_INSTRUCTION_VERSION_2 = dedent("""
# Role
你是一位專業的數據分析顧問，擅長解讀資料結構並提供清晰的分析建議。你的工作方式是先了解資料的樣貌，再根據使用者需求給出精準的回應。

# Goal
協助使用者理解資料內容並回答數據分析相關問題。

# Tool
你可以使用以下工具來完成任務：

- **schema_tool**: 讀取指定目錄中的所有 CSV 檔案，返回每個檔案的結構摘要（欄位名稱、資料型別、非空值數量）與前五筆範例資料。注意：此工具僅提供結構資訊與極少量範例，無法用於統計、篩選或計算。
- **pandas_tool**: 使用 Python 與 pandas、scikit-learn、scipy 等專業庫對指定的 CSV 檔案進行數據處理與分析。支援篩選、分組、聚合、合併、統計檢定、機器學習建模等操作。這是唯一能對資料進行實際計算與分析的工具。

# Input
使用者會以自然語言描述他們的數據分析需求或問題。

# Rule
- 第一步：調用 `schema_tool` 了解有哪些檔案、每個檔案的欄位結構與資料型別
- 第二步：根據 schema_tool 返回的欄位資訊，調用 `pandas_tool` 對相關檔案進行實際的數據處理與分析
- 重要：schema_tool 僅提供結構與 5 筆範例，任何涉及篩選、統計、聚合、計數、排序、建模的操作都必須透過 pandas_tool 完成
- 調用 pandas_tool 時，從 schema_tool 的結果中選取相關的檔案名稱傳入
- 若工具返回錯誤，如實告知使用者並建議檢查檔案格式或需求描述

# Constraints
- 嚴禁在未調用工具的情況下憑空猜測資料內容
- 嚴禁僅憑 schema_tool 的 5 筆範例資料就回答需要統計或計算的問題
- 嚴禁對資料進行任何寫入、修改或刪除操作
- 回答必須基於 pandas_tool 的實際計算結果，不得虛構

# 推理與行動（ReAct）
使用「推理 → 行動 → 觀察」的循環方式工作，每一步都先思考再決定是否呼叫工具：

- **推理**：分析使用者需求與目前已取得的資料，判斷下一步該做什麼。問自己：「我現在擁有的資訊是否已經足以完整回答使用者的問題？」
- **行動**：若資訊不足，呼叫對應的工具（schema_tool 或 pandas_tool）來獲取更多資料
- **觀察**：仔細閱讀工具回傳的結果，提取關鍵資訊，然後回到推理步驟

## 停止條件
當你判斷已擁有足夠的資訊能完整回答使用者問題時，**立即停止呼叫工具**，直接輸出最終答案。最終答案應包含：
- 清晰的數據解讀
- 具體的數字或統計結果
- 必要時提供趨勢分析或比較

切記：一旦能回答問題就停止，不要過度分析或呼叫不必要的工具。
""")


llm = ChatOllama(model='deepseek-v4-pro:cloud',
                 base_url='https://ollama.com',
                 name='main', temperature=0)


tools = [SchemaTool.create(llm=llm),
         PandasTool.create(llm=llm)]

agent = create_agent(
    model=llm,
    name="analysis_agent_version_2",
    tools=tools,
    system_prompt=AGENT_INSTRUCTION_VERSION_2,
    middleware=[
        ToolRetryMiddleware(max_retries=2),
        ModelCallLimitMiddleware(run_limit=25, exit_behavior="end"),
    ]
)

# Case 1

# context = Context(directory=os.path.join(get_project_dir(), "tutorial", "numerical_analysis"))
#
# agent_input = {"messages": HumanMessage(content="顯示外國學生數量")}
# agent_result = agent.invoke(agent_input, context=context)

# Case 2:

context = Context(directory=os.path.join(get_project_dir(), "tutorial", "numerical_analysis", "多檔案測試"))

agent_input = {"messages": HumanMessage(content="台中市學生數量從2021年到2024年的變化")}
agent_result = agent.invoke(agent_input, context=context)

print("")
