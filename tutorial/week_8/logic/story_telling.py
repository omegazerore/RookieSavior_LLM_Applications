import os
import importlib
from textwrap import dedent

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.initialization import credential_init
from src.logic.basic_logic import build_standard_chat_prompt_template

credential_init()


system_template = dedent("""\
You are a helpful AI assistant who likes children. You are great storyteller and know how to create content for kindergarten kids. 
A short chapter is created once at a time.
""")


def story_pipeline(system_template: str):

    """Creates a pipeline for generating stories or narratives.


    Args:
    system_template (str): System-level instruction for the model.
    
    
    Returns:
    Runnable: A chain that generates a text story from input using GPT-4o-mini.
    """

    input_ = {"system": {"template": system_template},
              "human": {"template": "scratch: {scratch}\nwhat happens previously: {context}",
                        "input_variable": ["scratch", "context"]}}
    
    chat_prompt_template = module.build_standard_chat_prompt_template(input_)

    model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                       model_name="gpt-4o-mini", temperature=0)
    
    story_chain = chat_prompt_template | model | StrOutputParser()

    return story_chain