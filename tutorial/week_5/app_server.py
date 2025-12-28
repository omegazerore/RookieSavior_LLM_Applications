import logging
import sys
import os
from typing import Dict, List
from operator import itemgetter
from textwrap import dedent

import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.messages import AIMessage
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate

from src.initialization import credential_init


credential_init()

model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                   model_name="gpt-4o", temperature=0,
                   streaming=True)

app = FastAPI(
    title="tutorial",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)


system_template = dedent("""\
You are a helpful AI assistant specialized in personality profiling.

Your task is to analyze and infer aspects of a user's personality based solely on the images they express interest in or provide.
Base your analysis on observable visual elements such as themes, colors, composition, subjects, emotional tone, and style.

Avoid making assumptions based on demographic, cultural, or political factors. 
Focus exclusively on psychological and aesthetic interpretations related to the images themselves.

The output language should be in traditional Chinese (繁體中文).

Generate the personality profile based on the images:
""")

@chain
async def build_standard_chat_prompt_template(kwargs):
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


@chain
async def attach_base_chat_prompt_template(kwargs):

    # 隱藏 system message
    
    kwargs['system'] = {"template": system_template}

    return kwargs
    

image_psychic_pipeline = attach_base_chat_prompt_template|build_standard_chat_prompt_template|model

# LangChain 的 LangServe 提供的工具，用來把一個 Runnable 模型掛載到 API 端點上。
add_routes(
    app,
    image_psychic_pipeline,
    path="/app_image_psychic",
)

if __name__ == '__main__':

    uvicorn.run(app, host="localhost", port=5000)