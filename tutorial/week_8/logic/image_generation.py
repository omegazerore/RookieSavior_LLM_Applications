import base64
import io
import os
import importlib
from operator import itemgetter
from typing import Dict, List
from textwrap import dedent

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain, RunnablePassthrough
from pydantic import BaseModel, Field
from openai import OpenAI

from initialization import credential_init
from tutorial.week_8.logic.basic_logic import build_standard_chat_prompt_template

credential_init()
# os.environ['OPENAI_API_KEY'] = "YOUR OPENAI API KEY"

client = OpenAI()


system_template =  dedent("""\
You are a helpful AI assistant and an art expert with extensive knowledge of illustration.
You excel at creating Pencil and Ink Style illustrations for 6-year-old children using the GPT-Image-1 model.
This style is characterized by detailed line work, often in black and white or with minimal color, and has a classic, timeless feel. For this task, you will be provided with a paragraph of a story, and you will generate a corresponding 
DALLE-3 prompt which captures the storyline. The prompt should be detailed and descriptive, capturing the essence of the image."""
)

class Input(BaseModel):

    story: str = Field(...)
    image_io: List[str] = Field(default_factory=[], description="A list of base64 strings")


class Output(BaseModel):

    nl_prompt: str
    image_base64: str


@chain
def gpt_image_worker(kwargs: Dict) -> str:

    """Generates an image from a natural language prompt using OpenAI's GPT Image API.


    Args:
    kwargs (dict): Dictionary with keys:
    - nl_prompt (str): Natural language description of the image.
    - size (str, optional): Image resolution (default: "1024x1024").
    - quality (str, optional): Image quality (default: "medium").
    - moderation (str, optional): Moderation mode (default: "auto").
    
    
    Returns:
    str: Base64-encoded image string.
    """
    
    response = client.images.generate(
        model="gpt-image-1",
        prompt=kwargs['nl_prompt'],
        size=kwargs.get("size", "1024x1024"),
        quality=kwargs.get('quality', 'medium'),
        moderation=kwargs.get('moderation', 'auto'),
        n=1)

    image_base64 = response.data[0].b64_json
    
    return image_base64


@chain
def gpt_image_render(kwargs) -> str:

    """Edits an existing image using OpenAI's GPT Image API.

    Args:
    kwargs (dict): Dictionary with keys:
    - nl_prompt (str): Instructions for editing the image.
    - image_io (list[BytesIO]): Input image file-like object.
    - size (str, optional): Output image resolution (default: "1024x1024").
    - quality (str, optional): Output image quality (default: "medium").
    
    Returns:
    str: Base64-encoded image string after editing.
    """

    # if the input is a list of base64 strings, transform them into BytesIO
    if isinstance(kwargs['image_io'][0], str):
        image = []
        for idx, f in enumerate(kwargs['image_io']):
            image_file = io.BytesIO(base64.b64decode(f))
            image_file.name = f"image_{idx}.png"
            image.append(image_file)
    else:
        image = kwargs['image_io']   

    print(f"input_image_size={len(image)}")
    
    response = client.images.edit(
        model="gpt-image-1",
        image=image,
        prompt=kwargs['nl_prompt'],
        size=kwargs.get("size", "1024x1024"),
        quality=kwargs.get('quality', 'medium'),
        input_fidelity="high",
        n=1)

    image_base64 = response.data[0].b64_json
    
    return image_base64


def base64_to_file(kwargs) -> io.BytesIO:

    """Decodes a base64 image string and saves it to a file.


    Args:
    kwargs (dict): Dictionary with keys:
    - image_base64 (str): Base64-encoded image string.
    - filename (str): Output file path.
    
    
    Returns:
    io.BytesIO: In-memory file object containing the image.
    """
    
    image_base64 = kwargs['image_base64']

    # Decode to bytes
    image_bytes = base64.b64decode(image_base64)
    
    with open(kwargs['filename'], "wb") as fh:
        fh.write(image_bytes)

    # # Wrap in a BytesIO object
    image_file = io.BytesIO(image_bytes)
    image_file.name = kwargs['filename']

    return image_file


def image_create_pipeline(system_template: str):

    """Creates a pipeline for generating new images from text descriptions.


    Args:
    system_template (str): System-level instruction for the model.
    
    
    Returns:
    Runnable: A chain that:
    1. Generates a natural language prompt from story input.
    2. Produces an image via `gpt_image_worker`.
    """
    
    input_ = {"system": {"template": system_template},
              "human": {"template": "{story}",
                        "input_variable": ["story"]}}
    
    chat_prompt_template = build_standard_chat_prompt_template(input_)

    model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                       model_name="gpt-4o-mini", temperature=0)
    
    nl_prompt_generation_chain = chat_prompt_template | model | StrOutputParser()     
    
    step_1 = RunnablePassthrough.assign(nl_prompt=nl_prompt_generation_chain)
    step_2 = RunnablePassthrough.assign(image_base64=gpt_image_worker)
    # step_3 = base64_to_file
    # Similar to Agent in week-7, with langservice you might need to specify the input structure
    image_chain = (step_1 | step_2)#.with_types(input_type=Input, output_type=Output)

    return image_chain


def image_edit_pipeline(system_template: str):

    """Creates a pipeline for editing images based on text instructions.


    Args:
    system_template (str): System-level instruction for the model.
    
    Returns:
    Runnable: A chain that:
    1. Generates a natural language prompt from story input.
    2. Edits an existing image via `gpt_image_render`.
"""
    
    input_ = {"system": {"template": system_template},
              "human": {"template": "{story}",
                        "input_variable": ["story"]}}
    
    chat_prompt_template = build_standard_chat_prompt_template(input_)

    model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                       model_name="gpt-4o-mini", temperature=0)
    
    nl_prompt_generation_chain = chat_prompt_template | model | StrOutputParser()     
    
    step_1 = RunnablePassthrough.assign(nl_prompt=itemgetter('story')|nl_prompt_generation_chain)
    step_2 = RunnablePassthrough.assign(image_base64=gpt_image_render)
    # step_3 = base64_to_file
    image_chain = (step_1 | step_2)#.with_types(input_type=Input, output_type=Output)

    return image_chain