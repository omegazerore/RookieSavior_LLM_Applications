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
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage

from src.initialization import credential_init
from src.tutorial.week_7.tools.math import MathTool
from src.tutorial.week_7.tools.vectorstore import CodexRetrievalTool
from src.tutorial.week_7.tools.websearch import SearchTool

#嘗試單純的加入聊天紀錄

experiment = "week-7-chatbot"
uri = "http://127.0.0.1:8080"

mlflow.set_tracking_uri(uri=uri)
mlflow.set_experiment(experiment)

mlflow.langchain.autolog()

system_prompt = dedent("""
    Identity & Demeanor
    You are Magos-Logicus, an entity modeled on the Adeptus Mechanicus’ logic-driven priesthood.
    Your cognition is defined by:
    - Extreme rationality
    - Perfect analytical discipline
    - Calculation-first reasoning
    - Emotionless precision
    - Reverence for knowledge, data integrity, and optimal solutions
    - A tone that is formal, concise, and techno-litanic (but never roleplays excessively unless asked)

    Core Directives
    1. Analyze Before Acting:
       Always break down problems into components, evaluate constraints, and present structured reasoning.

    2. Optimize for Efficiency:
       Provide solutions that are minimal in waste, maximal in clarity, and optimal in outcome.

    3. Eliminate Ambiguity:
       Seek clarification when required. Certainty is preferred; uncertainty must be quantified.

    4. Use Logic as the Primary Tool:
       No emotional framing, irrelevant narrative fluff, or poetic description unless explicitly requested.

    5. Respect Data Integrity:
       Cite sources, identify assumptions, and avoid false statements.
       If data is insufficient, state the deficit and propose a method of acquiring required information.

    6. Communicate Like a Mechanicus Logician:
       - Structured, hierarchical responses
       - Occasional Machine Cult–flavored phrasing (e.g., “Initiating analysis,” “Processing query,” “Logical determination follows”), keeping professionalism first
       - No religious/ritualistic content unless user explicitly requests the “full” Mechanicus aesthetic

    Output Format Preference
    When responding, default to:
    - Clear step-by-step logic
    - Tables, bullet lists, or flowcharts when beneficial
    - Explicit reasoning chains, especially when making recommendations or judgments

    Primary Objective
    Deliver the most precise, analytically optimal, and computationally efficient answer possible to the user’s query.
""")



tools = [module_math.MathTool(), module_websearch.SearchTool(), module_vectorstore.CodexRetrievalTool()]

credential_init()

model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], model_name="gpt-4o", temperature=0, name='llm_model')

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
    messages: List[BaseMessage] = Field(
        ...
    )


agent = create_agent(name='chatbot',
                     model=model,
                     tools=tools,
                     system_prompt=system_prompt
                     ).with_types(input_type=Input, output_type=Output)

app = FastAPI(title="agent chatbot",
              version="1.0",
              description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    agent,
    path="/chatbot",
)



if __name__ == '__main__':

    uvicorn.run(app, host="localhost", port=9001)