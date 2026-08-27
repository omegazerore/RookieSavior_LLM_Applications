from textwrap import dedent
from typing import Literal

import pandas as pd
from langchain.tools import BaseTool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import ConfigurableField
from langchain_core.documents import Document
from pydantic import BaseModel, Field


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

vectorstore = FAISS.load_local(
    "tutorial/week_5/warhammer 40k codex", embeddings, 
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_type="similarity", k=10).configurable_fields( \
                                        search_kwargs=ConfigurableField(
                                                id="search_kwargs",
                                            )
                                        )

class Inputs(BaseModel):
    query: str = Field(description="User query")
    clan: Literal['Adeptus Mechanicus', 'Aeldari', 'Black Templars'] = Field(description="")


class CodexRetrievalTool(BaseTool):

    input_output_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Inputs)
    input_format_instruction: str = input_output_parser.get_format_instructions()
    
    name:str = "warhammer-40k-codex"
    description_template:str = dedent("""
    This tool can be used to retrieve relevant information about warhammer 40k, 
    particularly Adeptus Mechanicus, Aeldari, Black Templars.
    The inputs contains user's question `query` and the party/clan `clan`.
    input format instructions: {input_format_instruction}
    """)

    description: str = description_template.format(input_format_instruction=input_format_instruction)
    
    def _run(self, **input):

        query = input["query"]
        clan = input["clan"]

        """
        vectorstore創造時在filename的部分有點失誤
        所以這裡用手動的方式進行校正
        """

        if clan == 'Black Templars':
            filter_ = {"filename": f"Codex -{clan}"}
        else:
            filter_ = {"filename": f"Codex - {clan}"}
        
        retrievd_documents = retriever.invoke(query, config={"configurable": 
                                                             {"search_kwarg": {"filter": filter_
                                                                              }
                                                             }
                                                            }
                                             )
        
        return retrievd_documents
        
    async def _arun(self, **input):
        
        query = input["query"]
        clan = input["clan"]

        if clan == 'Black Templars':
            filter_ = {"filename": f"Codex -{clan}"}
        else:
            filter_ = {"filename": f"Codex - {clan}"}
        
        retrieved_documents = await retriever.ainvoke(query, config={"configurable": 
                                                             {"search_kwarg": {"filter": filter_
                                                                              }
                                                             }
                                                            }
                                             )

        return retrieved_documents