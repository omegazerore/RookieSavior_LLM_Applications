import os
import re
from textwrap import dedent
from typing import Union

from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_core.runnables import chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.initialization import credential_init

credential_init()

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


@chain
def code_execution(code):
    
    match = re.findall(r"python\n(.*?)\n```", code, re.DOTALL)
    python_code = match[0]
    
    lines = python_code.strip()#.split('\n')
    # *stmts, last_line = lines

    local_vars = {}
    exec(lines, {}, local_vars)

    return local_vars


system_template = (
    "You are a highly skilled Python developer. Your task is to generate Python code strictly based on the user's instructions.\n"
    "Leverage statistical and mathematical libraries such as `statsmodels`, `scipy`, and `numpy` where appropriate to solve the problem.\n"
    "Your response must contain only the Python code — no explanations, comments, or additional text.\n\n"
)

human_template = dedent("""{query}\n\n
                            Always copy the final answer to a variable `answer`
                            Code:
                        """)


input_ = {"system": {"template": system_template},
          "human": {"template": human_template,
                    "input_variable": ["query"]}}

code_chat_prompt_template = build_standard_chat_prompt_template(input_)

model_coder = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                         model_name="gpt-4o-mini", temperature=0, 
                         name='coder')

code_generation = code_chat_prompt_template|model_coder|StrOutputParser()

code_pipeline = code_generation|code_execution


class MathTool(BaseTool):
    
    name:str = "Math-Solver" 
    description:str = dedent("""Use this tool to solve algorithmic problem by python programming.""")
    
    def _run(self, query: str):
        
        return  code_pipeline.invoke({"query": query})
    
    async def _arun(self, query: str):
        
        return await code_pipeline.ainvoke({"query": query})