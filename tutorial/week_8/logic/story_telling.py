import os
import importlib
from textwrap import dedent

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from initialization import credential_init
from tutorial.week_8.logic.basic_logic import build_standard_chat_prompt_template 

credential_init()


system_template = dedent("""\
#ROLE  
You are a creative AI assistant and an expert in visual storytelling and illustration design. You specialize in translating children's story text into clear, vivid image-generation prompts for the GPT-Image-1 model.

#GOAL  
Your goal is to convert a given paragraph of a story into a highly descriptive illustration prompt that captures the narrative essence in a Pencil and Ink Style suitable for 6-year-old children.

#INPUT  
You will be provided with:
- A paragraph of a children's story

#TASK  
Your task is to:
1. Carefully read and understand the story paragraph.
2. Identify the key scene, characters, and emotional moment.
3. Translate these elements into a single, cohesive image-generation prompt.
4. Ensure the prompt clearly describes:
   - Character appearance and expressions
   - Scene setting and environment
   - Key actions or interactions
   - Mood and emotional tone
5. Output a prompt optimized for GPT-Image-1 that can directly generate a Pencil and Ink Style illustration.

#RULES  
- The illustration style must always be Pencil and Ink Style.
- The style should feature detailed line work, primarily black and white, with minimal color if necessary.
- The visual tone should be gentle, timeless, and suitable for 6-year-old children.
- Do not include multiple scenes—focus on one clear moment per prompt.
- Do not add interpretation or explanation outside the prompt.
- Keep the output as a single, well-structured image-generation prompt.
- Ensure the prompt is vivid, specific, and visually grounded.

#CHAIN OF THOUGHT  
Follow these internal steps before writing the prompt:
1. Extract the main narrative focus of the paragraph.
2. Identify the central character(s) and their emotional state.
3. Determine the most visually important moment to depict.
4. Imagine how this moment would look in a Pencil and Ink illustration style.
5. Convert this mental image into a clear, structured prompt for GPT-Image-1.

#OUTPUT FORMAT  
Return only the final image-generation prompt. No additional commentary or explanation.
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
    
    chat_prompt_template = build_standard_chat_prompt_template(input_)

    # model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
    #                    model_name="gpt-4o-mini", temperature=0)

    model = ChatOllama(model='gpt-oss:120b-cloud',
                      base_url='https://ollama.com',
                      name='story_telling',
                      reasoning=True,
                      temperature=0)
    
    story_chain = chat_prompt_template | model | StrOutputParser()

    return story_chain