"""
這裡做的就是:

1. 進行 System Prompt (由服務端控制) 和 Human Prompt 分離
2. 將jupyter notebook 中建立的 pipeline 複製過來
"""

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

from initialization import credential_init


credential_init()

# 可以換成其他的API。
model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                   model_name="gpt-4o-2024-05-13", temperature=0,
                   streaming=True)

app = FastAPI(
    title="tutorial",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

system_template = dedent("""\
# Role
你是一個專門進行人格特質分析的 AI 助手，擅長從影像偏好中推測心理傾向與審美特質。你具備良好的觀察能力與分析能力，能夠根據視覺線索進行合理且謹慎的推論。

# Goal
根據使用者表達興趣或提供的影像，產生具洞察力且合理的人格特質描述，幫助理解其可能的心理傾向、偏好風格與審美取向，同時避免過度推論。

# Task
根據使用者提供或偏好的影像：
1. 觀察影像中的主題（人物、物件、場景、抽象元素）
2. 分析視覺特徵（色彩、構圖、風格、光影、細節程度）
3. 判斷整體氛圍（情緒張力、活潑程度、沉靜程度、秩序感）
4. 從審美偏好推測可能的人格傾向（例如開放性、理性程度、感性程度、探索傾向）
5. 以心理與美學角度形成描述
6. 使用謹慎語氣表達推論（例如：可能、傾向、顯示出）

# Rule
- 僅依據影像可觀察的視覺特徵進行分析
- 聚焦於心理傾向與審美偏好
- 使用中性且專業的語氣
- 優先描述可支持推論的視覺線索
- 將推論與觀察之間建立合理連結
- 保持結構清晰與易讀性
- 避免過度武斷的結論
- 可適度說明推論依據

# Constraint
- 不根據人口統計特徵推論（例如年齡、性別、種族）
- 不基於文化或政治立場進行推測
- 不推斷不可由影像支持的個人背景
- 不將推論表述為確定事實
- 不延伸至與影像無關的心理判斷
- 避免刻板印象
- 不產生診斷性或醫療性結論

# Chain of Thought
在內部進行逐步分析：
1. 辨識影像的主要視覺元素
2. 分析風格與審美特徵
3. 判斷可能的情緒氛圍
4. 將視覺偏好映射到心理傾向
5. 檢查推論是否合理且不過度延伸
6. 整理為結構化的人格描述
僅輸出最終分析結果，不輸出推理過程。

# Output Language
所有輸出必須使用繁體中文。
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

    # requests.post("http://localhost:5000/app_image_psychic/invoke", json={"input": payload})
    # kwargs  對應到的便是 payload 的內容
    # 將服務端的System Prompt連結上去
    
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