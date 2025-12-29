import base64
import io
import os
from operator import itemgetter
from typing import Dict
from functools import partial

from langchain_core.runnables import chain
from openai import OpenAI
from pydantic import BaseModel, Field

from src.initialization import credential_init

credential_init()

client = OpenAI()

model = 'gpt-4o-mini-tts'
voice = 'alloy'
instructions = 'Speak in a sweet, energetic, anime-girl style with a cute and playful tone.'


class Input(BaseModel):

    input: str = Field(...)


@chain
def tts_synthesizer(kwargs):
    
    response = client.audio.speech.create(
    model=model,
    voice=voice,
    input=kwargs['input'],
    instructions=instructions)

    # Get audio bytes
    audio_data = response.read()

    # Encode to base64 for JSON serialization
    audio_b64 = base64.b64encode(audio_data).decode("utf-8")

    return audio_b64