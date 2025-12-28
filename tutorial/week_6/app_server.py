import atexit
import logging
import sys
import os
from typing import Dict, List, Literal
from operator import itemgetter
from textwrap import dedent

# import mlflow
import uvicorn
import pandas as pd
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


from src.initialization import credential_init
# from src.trendweek_report.deep_search import OpenAIWebSearch
# from src.trendweek_report.examples_with_deep_search import SearchQueryAssistant, response_to_output_text

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


credential_init()

model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                   model_name="gpt-4o-mini", temperature=0)


def build_standard_chat_prompt_template(kwargs):
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


class Drink(BaseModel):

    name: Literal['珍珠蜂蜜鮮奶普洱', '茶凍奶綠', '嚴選高山茶', 
                  '咖啡奶茶', '冬瓜檸檬'] = Field(description="飲料名稱")
    ice_level: Literal['正常冰', '少冰', '微冰', '去冰'] = Field(description='冰熱程度')
    sugar_level: Literal['無糖', '微糖', '半糖' , '少糖'] = Field(description='糖度')


class Order(BaseModel):

    names: List[Drink] = Field(description=("用戶點的飲料"))


output_parser = PydanticOutputParser(pydantic_object=Order)
format_instructions = output_parser.get_format_instructions()


# 建立 pipeline
human_template = dedent("""
                 {query}
                 format instruction: {format_instructions}
                 """)

input_ = {
          "human": {"template": human_template,
                    "input_variable": ["query"],
                    "partial_variables": {"format_instructions": 
                                          format_instructions}}}

chat_prompt_template = build_standard_chat_prompt_template(input_)

pipeline = chat_prompt_template | model | output_parser


app = FastAPI(
    title="langchain LLM service",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    pipeline,
    path="/drinking_app",
)


# IMPORTANT: Do not close the run here. Close it when server shuts down.
# For example, register a shutdown hook:

# atexit.register(mlflow.end_run)


if __name__ == '__main__':

    uvicorn.run(app, host="localhost", port=9000)