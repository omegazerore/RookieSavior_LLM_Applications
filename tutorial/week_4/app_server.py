import importlib
import os
from textwrap import dedent
from typing import List, Dict
from datetime import datetime

import uvicorn
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import chain
from openai import OpenAI

from src.initialization import credential_init

def build_standard_chat_prompt_template(kwargs):

    messages = []
 
    if 'system' in kwargs:
        content = kwargs.get('system')
        prompt = PromptTemplate(**content)
        message = SystemMessagePromptTemplate(prompt=prompt)
        messages.append(message)  

    if 'human' in kwargs:
        content = kwargs.get('human')
        prompt = PromptTemplate(**content)
        message = HumanMessagePromptTemplate(prompt=prompt)
        messages.append(message)
        
    chat_prompt_template = ChatPromptTemplate.from_messages(messages)
    
    return chat_prompt_template
    

#嘗試單純的加入聊天紀錄

experiment = "Week-4"
uri = "http://127.0.0.1:8080"

# mlflow.set_tracking_uri(uri=uri)
# mlflow.set_experiment(experiment)

# mlflow.langchain.autolog()

# 定義系統提示
system_message = SystemMessage(content="You are a helpful assistant.")

# 讀取API Keys
credential_init()

# 用來總結 Tool 結論的 模型
model_followup = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], model_name="gpt-4o-mini", temperature=0, name='llm_tool_followup')

# 工具
## 數學工具
def build_code_pipeline():
    system_template = dedent("""
        You are a highly skilled Python developer. Your task is to generate Python code strictly based on the user's instructions.
        Leverage statistical and mathematical libraries such as `statsmodels`, `scipy`, and `numpy` where appropriate to solve the problem.
        Your response must contain only the Python code — no explanations, comments, or additional text.
        """)
    
    human_template = dedent("""
                            {query}
                            Always copy the final answer to a variable `answer`
                            Code:
                            """)
    
    
    input_ = {"system": {"template": system_template},
              "human": {"template": human_template,
                        "input_variable": ["query"]}}
    
    
    model_code = ChatOpenAI(model="gpt-4o", name="llm_code")
    
    chat_prompt_template = build_standard_chat_prompt_template(input_)
    
    code_generation = chat_prompt_template|model_code|StrOutputParser()

    return code_generation
    

@chain
def code_execution(code):
    
    match = re.findall(r"python\n(.*?)\n```", code, re.DOTALL)
    python_code = match[0]
    
    lines = python_code.strip()#.split('\n')
    # *stmts, last_line = lines

    local_vars = {}
    exec(lines, local_vars)

    return local_vars

code_generation = build_code_pipeline()

code_pipeline = code_generation|code_execution


class CodeArgs(BaseModel):
    query: str = Field(description="User request; 用戶需求")


def _calculator(query: str,):
    output = code_pipeline.invoke(query)
    return output

# ToDo check calling this function later
# mathematic_tool = StructuredTool.from_function(
#     func=_calculator,
#     args_schema=CodeArgs,
#     description="Use this tool to solve mathematic related problem; 使用這個工具解決數學問題",
# )

@tool
def mathematic_tool(query: str) -> str:
    """
    執行數學運算，適用於需要精確計算的問題。

    參數：
        query (str): 用戶的需求

    回傳：
        str: 計算結果。
    """
    try:
        expression = _calculator(query)
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


### 網路搜索工具
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

@tool
def websearh_tool(query: str) -> str:
    """
    使用網路搜尋取得最新、即時或不在模型既有知識範圍內的資訊。

    此工具應在以下情況下由 LLM 調用：
    1. 問題涉及時事或近期事件，例如：
       - 當前或最近的政治情勢、政策變化、選舉結果
       - 明星、網紅的最新八卦或動態
       - 體育賽事結果、戰績、轉會消息
       - 任何具有時間敏感性、會隨時變動的資訊
    2. 當 LLM 無法確定自身知識是否足以準確回答問題時，
       特別是可能超出模型知識截止時間或需要即時驗證的內容。

    使用原則：
    - 若問題可由一般常識、穩定不變的知識或訓練資料中可靠回答，則不應調用。
    - 若答案的正確性高度依賴最新資訊，應優先調用此工具。
    - 搜尋關鍵字應簡潔且包含必要的上下文，以提高結果相關性。

    參數：
        query (str): 用於搜尋引擎的關鍵字或自然語言查詢。

    回傳：
        str: 來自網路搜尋結果的摘要或相關資訊，用於輔助生成最終回答。

    範例：
        websearch("2025 台灣 總統 最新 民調")
        websearch("今天 NBA 勇士 比賽 結果")
        websearch("某某明星 最近 發生 什麼事")
    """

    response = client.responses.create(
                    model="gpt-4o-mini",
                    tools=[
                        {"type": "web_search",}
                    ],
                    tool_choice="auto",
                    input=query)
    
    return response.output_text

# 搭載工具的模型
model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], 
                   model_name="gpt-4o", 
                   temperature=0, 
                   name='llm_model_with_tool').bind_tools([websearh_tool, mathematic_tool])


# **** 工具使用 ****
def call_function(tool_call):
    
    return eval(tool_call['name']).invoke(**tool_call['args'])


def follow_up_answer(aimessage):

    messages = [aimessage]

    for tool_call in aimessage.tool_calls:
        result = eval(tool_call['name']).invoke(tool_call['args'])
        tool_msg = ToolMessage(
            content=str(result),          # usually a string or simple text
            tool_call_id=tool_call['id']    # must match the AIMessage tool_call id
        )
        messages.append(tool_msg)
    
    return model_followup.invoke(messages)
# **************
    

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.
class Input(BaseModel):
    """
    Field:
     - 第一個參數 ... 代表 這個欄位是必填的。等同於 required=True。
    """
    messages: List[Dict] = Field(
        ...
    )


class Output(BaseModel):
    messages: List[Dict] = Field(
        ...
    )


chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        system_message,
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chatbot_pipeline = chat_prompt_template|model

@chain
def chatbot(input_: Input) -> Output:

    messages = input_["messages"]
    
    output = chatbot_pipeline.invoke({"messages": messages})

    # 將我們在jupyter notebook中範例的代碼直接copy/paste，然後稍微修改一下
    try:
        if output.tool_calls != []:
            print(f"Call tool: {output.tool_calls}")
            final_answer = follow_up_answer(aimessage=output)
        else:
            print("No Tool")
            final_answer = output
        print(f"AI: {final_answer.content}")
        messages.append({"role": "ai", "content": final_answer.content})
    except KeyError:
        print(f"AI: {output.content}")
        messages.append({"role": "ai", "content": output.content})

    return Output(messages=messages)
    
    

app = FastAPI(title="chatbot",
              version="1.0",
              description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    chatbot,
    path="/chatbot",
)



if __name__ == '__main__':

    uvicorn.run(app, host="localhost", port=9001)