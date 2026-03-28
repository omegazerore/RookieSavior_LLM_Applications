import importlib
import os
from textwrap import dedent
from typing import List, Dict
from datetime import datetime

import uvicorn
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import chain, RunnableLambda
from openai import OpenAI

from initialization import credential_init

# 設定好跟 mlflow的連線
uri = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(uri=uri)
# 設定experiment
experiment = "week_4"
mlflow.set_experiment(experiment)

mlflow.langchain.autolog()

# 讀取API Keys
credential_init()

# 用來總結 Tool 結論的 模型
model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], model_name="gpt-4o-mini", temperature=0, name='llm_demo')
    

app = FastAPI(title="chatbot",
              version="1.0",
              description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    model,
    path="/openai",
)

# 建立 feedback-revision 流程

# 因為 Generation_Reflection_Demo 中的檔案需要環境變數 os.environ['experiment']
# 所以要提前給它，否則會報錯
os.environ['experiment'] = experiment
# 建立好 run_id
run = mlflow.start_run(run_name='DEMO')
os.environ["run_id"] = run.info.run_id

model = mlflow.pyfunc.load_model("models:/Generation_Reflection_Demo/1")

# 將模型包裝成 LangChain 的 Runnable
# 注意：這裡的 input 需要符合你模型的輸入格式（通常是 dict 或 DataFrame）
def model_invoke(input_data):
    # 如果你的模型預期的是單純字串，可以直接傳入
    # 如果是複雜物件，可能需要先做預處理
    df_input = pd.DataFrame([input_data])
    
    return model.predict(df_input)

pipeline = RunnableLambda(model_invoke)

add_routes(
    app,
    pipeline,
    path="/demo"
)



if __name__ == '__main__':

    uvicorn.run(app, host="localhost", port=5000)