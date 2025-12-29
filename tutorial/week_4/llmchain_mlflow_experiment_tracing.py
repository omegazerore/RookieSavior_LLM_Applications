import os
from textwrap import dedent

import mlflow
import pandas as pd
from mlflow.models import set_model
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.callbacks import MlflowCallbackHandler
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel, Field

from src.initialization import credential_init


credential_init()

model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                   model_name="gpt-4o-mini", temperature=0)


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
        
    chat_prompt = ChatPromptTemplate.from_messages(messages)
    
    return chat_prompt


def create_feedback_pipeline(mlflow_callback):

    ## Teacher LLM
    system_template = dedent("""
    你是一個教學與寫作經驗豐富的台灣大學中文系教授，你要來負責給予作文評分與回饋。
    """)
    
    human_template = dedent("""
    Title: {title}
    
    Article:
    {article}
    """)
    
    input_ = {"system": {"template": system_template},
              "human": {"template": human_template,
                        "input_variable": ["title", "article"],
                        }}
    
    chat_prompt_template = build_standard_chat_prompt_template(input_)

    model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                       model_name="gpt-4o-mini", temperature=0,
                       callbacks=[mlflow_callback],
                       name='feedback_model')
    
    feedback_pipeline = chat_prompt_template|model|StrOutputParser()

    return feedback_pipeline


class Output(BaseModel):
    name: str = Field(description="The revised article in traditional Chinese (繁體中文), please do not include the title.")


def create_revision_pipeline(mlflow_callback):
    ## Generate
    system_template = dedent("""
    你是一個在準備考試的高中生，你將根據反饋強化的作文內容。
    """)
    
    human_template = dedent("""
    Title: {title}
    
    Old Article:
    {article}
    
    Feedback:
    {feedback}

    Output format instructions: {format_instructions}
    
    Revised Article:
    """)

    output_parser = PydanticOutputParser(pydantic_object=Output)
    format_instructions = output_parser.get_format_instructions()
    
    input_ = {"system": {"template": system_template},
              "human": {"template": human_template,
                        "input_variable": ["title", "article", "feedback"],
                        "partial_variables": {'format_instructions': format_instructions}}}
    
    chat_prompt_template = build_standard_chat_prompt_template(input_)

    model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                       model_name="gpt-4o-mini", temperature=0,
                       callbacks=[mlflow_callback],
                       name='revision_model')
    
    revision_pipeline = chat_prompt_template|model|output_parser

    return revision_pipeline


class LLMChainModel(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):

        # Attach the run_id so all logs go into this run
        self.mlflow_cb = MlflowCallbackHandler(
        experiment=os.environ['experiment'],
        run_id=os.environ["run_id"],
        tracking_uri="http://127.0.0.1:8080",
    )
        
        feedback_pipeline = create_feedback_pipeline(mlflow_callback=self.mlflow_cb)
        revision_pipeline = create_revision_pipeline(mlflow_callback=self.mlflow_cb)

        self.pipeline = RunnablePassthrough.assign(feedback=feedback_pipeline)|revision_pipeline|RunnableLambda(lambda x: x.name)
        
    def predict(self, context, model_input: pd.DataFrame):

        output = self.pipeline.invoke({"article": model_input.loc[0]['article'],
                                       "title": model_input.loc[0]['title']})

        self.mlflow_cb.flush_tracker()
        
        return output

set_model(LLMChainModel())
