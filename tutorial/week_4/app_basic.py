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

# mlflow.set_tracking_uri(uri=uri)
# mlflow.set_experiment(experiment)

# mlflow.langchain.autolog()

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
# pipeline

# add_routes(
#     app,
#     pipeline,
#     path="demo"
# )



if __name__ == '__main__':

    uvicorn.run(app, host="localhost", port=5000)