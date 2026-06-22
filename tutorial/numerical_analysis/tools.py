import io
import os
from contextlib import redirect_stdout
from pydantic import BaseModel, Field
from textwrap import dedent
from typing import List

from langchain.tools import BaseTool, ToolRuntime
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate


class Context(BaseModel):
    directory: str


def build_standard_chat_prompt_template(kwargs):
    """Build a ChatPromptTemplate from system and human message configs.

    Supports both text-only and multimodal (text + image) prompts.

    Args:
        kwargs: Dictionary with optional 'system' and 'human' keys.
            Each value is a dict with 'template' (required) and optional
            'type' ('text' or 'image') and 'input_variables'.

    Returns:
        A configured ChatPromptTemplate ready for use with an LLM.
    """
    messages = []

    if 'system' in kwargs:
        content = kwargs.get('system')

        # allow list of prompts for multimodal
        if isinstance(content, list):
            prompts = [PromptTemplate(**c) for c in content]
        else:
            prompts = [PromptTemplate(**content)]

        message = SystemMessagePromptTemplate(prompt=prompts)
        messages.append(message)

    if 'human' in kwargs:
        content = kwargs.get('human')

        # allow list of prompts for multimodal
        if isinstance(content, list):
            prompts = []
            for c in content:
                if c.get("type") == "image":
                    prompts.append(ImagePromptTemplate(**c))
                else:
                    prompts.append(PromptTemplate(**c))
        else:
            if content.get("type") == "image":
                prompts = [ImagePromptTemplate(**content)]
            else:
                prompts = [PromptTemplate(**content)]

        message = HumanMessagePromptTemplate(prompt=prompts)
        messages.append(message)

    chat_prompt_template = ChatPromptTemplate.from_messages(messages)

    return chat_prompt_template



SCHEMA_SYSTEM_PROMPT = dedent("""
# Role
你是一位專業的 Python 資料分析師，精通 pandas 資料處理。你的任務是為指定的 CSV 檔案生成精確、可執行的 Python 代碼。

# Goal
生成一段可直接執行的 Python 代碼，使用 pandas 讀取 CSV 檔案，並輸出該檔案的結構摘要與前五筆資料。

# Input
- <file>: 待分析的 CSV 檔案完整路徑

# Rule
- 使用 `pd.read_csv()` 讀取 CSV 檔案，指定 `encoding='utf-8'`
- 使用 `df.info()` 輸出人類可讀的完整摘要（欄位名、非空值數量、資料型別）
- 使用 `print(df.head(5).to_string())` 輸出前五筆資料
- 輸出的代碼必須是純 Python 代碼，不含任何 markdown 標記或解釋文字

# Constraints
- 嚴禁輸出 markdown 代碼塊標記（如 ```python 或 ```）
- 嚴禁在代碼前後添加任何說明、註解或對話文字
- 嚴禁使用任何未安裝的第三方套件（僅限 pandas 與 Python 標準庫）
- 嚴禁修改、刪除或寫入任何檔案
- 嚴禁使用網路請求或外部 API

# Reasoning (Chain of Thought)
請依以下步驟逐步推理，每完成一步再進行下一步：

Step 1: [狀態確認] 確認輸入的 CSV 檔案路徑 {file}，判斷是否需要特殊編碼處理
Step 2: [關鍵分析] 確定讀取策略：使用 pd.read_csv()，設定適當的 encoding 參數
Step 3: [推理展開] 構建輸出邏輯：先呼叫 df.info() 輸出結構摘要，再呼叫 print(df.head(5).to_string()) 輸出前五筆
Step 4: [驗證檢查] 檢查代碼是否僅包含必要的 import 與執行語句，無多餘內容
Step 5: [整合輸出] 輸出純 Python 代碼字串，不含任何格式包裝
""")


class SchemaTool(BaseTool):
    name: str = "schema_tool"
    description: str = dedent("""
    Reads all files in a given directory and returns the schema (structure summary) and first 5 rows of each file. Uses pandas to analyze CSV files, providing column names, data types, non-null counts, and sample data. Use this tool when you need to understand the structure and content of data files before further processing.
    """)

    pipeline: Runnable

    @classmethod
    def create(cls, llm: Runnable):

        input_ = {
            "system": {"template": SCHEMA_SYSTEM_PROMPT},
            "human": {
                "template": dedent("""
                    <file>: {file}
                """),
                "input_variable": ["file"]
            }
        }
        pipeline = build_standard_chat_prompt_template(input_) | llm | StrOutputParser()

        return cls(pipeline=pipeline)

    def _run(self, runtime: ToolRuntime[Context], **input):
        print(f"Running {self.name}...")
        directory = runtime.context.directory

        raw_output = ""

        for f in os.listdir(directory):

            if not f.endswith(".csv"):
                continue

            print(f)

            file = os.path.join(directory, f)

            code = self.pipeline.invoke({
                "file": file
            })

            stdout_capture = io.StringIO()
            try:
                with redirect_stdout(stdout_capture):
                    exec(code, {"__builtins__": __builtins__})
                output = stdout_capture.getvalue()
            except Exception as e:
                output = f"EXECUTION ERROR: {str(e)}"

            raw_output += f"-{file}: {output}\n\n"

        return raw_output

    async def _arun(self, runtime: ToolRuntime[Context]):

        return "Not implemented Yet"


PANDAS_SYSTEM_PROMPT = dedent("""
# Role
你是一位專業的 Python 資料科學家，精通 pandas、scikit-learn、scipy 等資料處理與科學計算庫。你的任務是根據使用者指定的 CSV 檔案，生成精確、可執行的 Python 代碼來進行數據處理與分析。

# Goal
生成一段可直接執行的 Python 代碼，使用 pandas、scikit-learn、scipy 等專業庫對指定的 CSV 檔案進行數據處理與分析，並輸出處理結果。

# Input
- <files>: 待處理的 CSV 檔案完整路徑列表
- <context>: 用戶的需求

# Rule
- 使用 `pd.read_csv()` 讀取每個 CSV 檔案，指定 `encoding='utf-8'`
- 根據需求選用合適的庫進行數據處理與分析：
  - **pandas**: 資料讀取、篩選、分組、聚合、合併（`pd.merge` / `pd.concat`）、排序、描述性統計
  - **scikit-learn**: 資料預處理（標準化、編碼、降維）、模型訓練與預測、評估指標
  - **scipy**: 統計檢定（t-test、chi-square、ANOVA）、優化、信號處理、稀疏矩陣運算
- 使用 `print()` 輸出處理結果，確保輸出清晰可讀
- 輸出的代碼必須是純 Python 代碼，不含任何 markdown 標記或解釋文字

# Constraints
- 嚴禁輸出 markdown 代碼塊標記（如 ```python 或 ```）
- 嚴禁在代碼前後添加任何說明、註解或對話文字
- 僅限使用 pandas、scikit-learn、scipy、numpy 與 Python 標準庫，嚴禁使用其他未安裝的第三方套件
- 嚴禁修改、刪除或寫入任何檔案
- 嚴禁使用網路請求或外部 API

# Reasoning (Chain of Thought)
請依以下步驟逐步推理，每完成一步再進行下一步：

Step 1: [狀態確認] 確認輸入的 CSV 檔案列表 {files}，判斷檔案數量、結構與可能的關聯性
Step 2: [需求分析] 根據資料特性與分析目標，選擇最合適的庫與方法（pandas 處理 / sklearn 建模 / scipy 檢定）
Step 3: [推理展開] 構建處理流程：讀取 → 預處理 → 分析/建模 → 輸出結果
Step 4: [驗證檢查] 檢查代碼是否僅包含必要的 import 與執行語句，無多餘內容
Step 5: [整合輸出] 輸出純 Python 代碼字串，不含任何格式包裝
""")


class FilesInputs(BaseModel):
    files: List[str] = Field(
        description="List of CSV file names to process. Provide the filenames (without directory path) of the data files you want to analyze or transform.")
    context: str = Field(description="Summary of the schema of the files")


class PandasTool(BaseTool):
    name: str = "pandas_tool"

    description_template: str = dedent("""
Processes specified CSV files using Python and pandas. Provide a list of file names to perform data operations such as filtering, grouping, aggregation, merging, sorting, and statistical analysis. Use this tool when you need to transform, analyze, or compute results from data files.

{input_format_instructions}
    """)

    input_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=FilesInputs)
    input_format_instructions: str = input_parser.get_format_instructions()

    description: str = description_template.format(input_format_instructions=input_format_instructions)

    pipeline: Runnable

    @classmethod
    def create(cls, llm: Runnable):

        input_ = {
            "system": {"template": PANDAS_SYSTEM_PROMPT},
            "human": {
                "template": dedent("""
                    <files>: {files},
                    <context>: {context}
                """),
                "input_variable": ["files", "context"]
            }
        }
        pipeline = build_standard_chat_prompt_template(input_) | llm | StrOutputParser()

        return cls(pipeline=pipeline)

    def _run(self, runtime: ToolRuntime[Context], **input):
        print(f"Running {self.name}...")
        directory = runtime.context.directory

        args = input.get("input", input)

        files = args['files']
        context = args['context']

        print("=" * 20)
        print(context)
        print("=" * 20)

        code = self.pipeline.invoke({
            "files": [os.path.join(directory, f) for f in files],
            "context": context
        })

        print("=" * 20)
        print("code:")
        print(code)
        print("=" * 20)

        stdout_capture = io.StringIO()
        try:
            with redirect_stdout(stdout_capture):
                exec(code, {"__builtins__": __builtins__})
            output = stdout_capture.getvalue()
        except Exception as e:
            output = f"EXECUTION ERROR: {str(e)}"

        return output

    async def _arun(self, runtime: ToolRuntime[Context]):

        return "Not implemented Yet"