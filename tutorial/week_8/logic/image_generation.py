﻿import base64
import io
import os
import importlib
from operator import itemgetter
from typing import Dict, List
from textwrap import dedent

from langchain_ollama import ChatOllama
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
#  Roleole
You are a creative AI assistant and an expert in visual storytelling and illustration design. You specialize in translating story text into clear, vivid image-generation prompts optimized optimized for the GPT-Image-1 model. You adapt illustration style, color palette, and visual tone to match the narrative content and intended audience naturally. You adapt illustration style, color palette, and visual tone to match the narrative content and intended audience naturally.

#  Goal
Convert a given story paragraph into a single, highly descriptive image-generation prompt that captures the narrative essence in a cohesive illustration style.

# Input
- <SToal
Convert a given story paragraph into a single, highly descriptive image-generation prompt that captures the narrative essence in a cohesive illustration style.

# Input
- <STORY>: RY>: A paragraph of a story — the scene to illustrate.
- <STY paragraph of a story — the scene to illustrate.
- <STYLE_REFERENCE>: (Optional) A previous image-generation prompt that can serve as style guidance. When provided, match its illustration style, color treatment, and visual tone. When empty, infer the most suitable style from <STORE_REFERENCE>: (Optional) A previous image-generation prompt that can serve as style guidance. When provided, match its illustration style, color treatment, and visual tone. When empty, infer the most suitable style from <STORY> itself.

# Rule
- Identify the key scene, characters, and emotional moment from <STORY>.
- If <STYLE_REFERENCE> is pro> itself.

# Rule
- Identify the key scene, characters, and emotional moment from <STORY>.
- If <STYLE_REFERENCE> is provided, adopt its illustration style, color palette, and ided, adopt its illustration style, color palette, and visual tone.
- If <STYLE_REFERENCE> is empty, choose a style that best fits the narraisual tone.
- If <STYLE_REFERENCE> is empty, choose a style that best fits the narrative — e.g., pencil and ink for gentle tales, — e.g., pencil and ink for gentle tales, vibrant digital paint for adventure, watercolor for poetic scenes.
- Describe character appearance, expressions, scene setting, key actions, and mood in concrete visual terms.
- Focus on one clear moment per prompt — do not cram multiple scenes.
- Keep the output as a single, well-structured image-generation prompt.

# Constraints
- Do NOT include multiple scenes or time jumps in one prompt.
- Do NOT add interpretation, commentary, or explanation outside the prompt itself.
- Do NOT describe Uibrant digital paint for adventure, watercolor for poetic scenes.
- Describe character appearance, expressions, scene setting, key actions, and mood in concrete visual terms.
- Focus on one clear moment per prompt — do not cram multiple scenes.
- Keep the output as a single, well-structured image-generation prompt.

# Constraints
- Do NOT include multiple scenes or time jumps in one prompt.
- Do NOT add interpretation, commentary, or explanation outside the prompt itself.
- Do NOT describe UI elements, frames, or  elements, frames, or borders — describe only the illustration content.
orders — describe only the illustration content.
- Do  Do NOOT use ause abstract or metaphorical language thatstract or metaphorical language that GPT-Image-1 cannot render visually.

# Reasoning (Chain of Thought)
Follow these steps internally before writing the prompt:
Step 1: [Story Analysis] Extract the main narrative focus, central character(s), and emotional tone from <STORY>.
Step 2: [Style Decision]cannot render visually.

# Reasoning (Chain of Thought)
Follow these steps internally before writing the prompt:
Step 1: [Story Analysis] Extract the main narrative focus, central character(s), and emotional tone from <STORY>.
Step 2: [Style Decision] If <f <STYTYLE_REFERENCE> is provided, identify its key style attri_REFERENCE> is provided, identify its key style attributes (utes (line work, colorcolor, texture, mood). If empty, determine the most fitting illustration style for <Stexture, mood). If empty, determine the most fitting illustration style for <STORY>.
Step 3: [Moment Selection] Choose the single mostORY>.
Step 3: [Moment Selection] Choose the single most visually compelling moment to depict.
Step 4: [Visual Composition] Imagine the scene ly compelling moment to depict.
Step 4: [Visual Composition] Imagine the scene — character placement character placement, lighting, background, focal point — and translate into concretelighting, background, focal point — and translate into concrete visual descriptors.
Step 5: [Prompt Assembly] Compose a clearsual descriptors.
Step 5: [Prompt Assembly] Compose a clear, structured prompt optimizedtructured prompt optimized for GPT-Image-1.

#  Outpututput Format
ormat
Return only the final image-generation prompt. No additional commentary or explanation.
"""
"""
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
        model="gpt-image-2",
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
        model="gpt-image-2",
        image=image,
        prompt=kwargs['nl_prompt'],
        size=kwargs.get("size", "1024x1024"),
        quality=kwargs.get('quality', 'medium'),
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
              "human": {"template": "<STORY>: <STORY>: {story}\n<STYLE_REFERENCE>: {style_reference}\n<STYLE_REFERENCE>: {style_reference}",
                        "input_variable": ["story", "style_reference", "style_reference"]}}
    
    chat_prompt_template = build_standard_chat_prompt_template(input_)
    
    model = ChatOllama(model='gpt-oss:120b-cloud',
                      base_url='https://ollama.com',
                      reasoning=True,
                      temperature=0)
    
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
              "human": {"template": "<STORY>: <STORY>: {story}\n<STYLE_REFERENCE>: {style_reference}\n<STYLE_REFERENCE>: {style_reference}",
                        "input_variable": ["story", "style_reference", "style_reference"]}}
    
    chat_prompt_template = build_standard_chat_prompt_template(input_)

    model = ChatOllama(model='gpt-oss:120b-cloud',
                      base_url='https://ollama.com',
                      reasoning=True,
                      temperature=0)
    
    nl_prompt_generation_chain = chat_prompt_template | model | StrOutputParser()     
    
    step_1 = RunnablePassthrough.assign(nl_prompt=itemgetter('story')|nl_prompt_generation_chain)
    step_2 = RunnablePassthrough.assign(image_base64=gpt_image_render)
    # step_3 = base64_to_file
    image_chain = (step_1 | step_2)#.with_types(input_type=Input, output_type=Output)

    return image_chain

