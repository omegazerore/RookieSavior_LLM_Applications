import base64
import io
import os
from operator import itemgetter
from typing import Dict
from functools import partial
from textwrap import dedent

from langchain_core.runnables import chain
from openai import OpenAI
from pydantic import BaseModel, Field

from initialization import credential_init

credential_init()

client = OpenAI()

model = 'gpt-4o-mini-tts'
voice = 'marin'
instructions = dedent("""
Accent/Affect: Warm, playful, and gently animated, like a caring storyteller reading a bedtime tale to a child. The voice feels soft, friendly, and full of imagination, helping young listeners feel safe and curious.

Tone: Simple, soothing, and encouraging, using very easy words and short sentences. The storyteller explains things like they are part of a fun story, making everything feel understandable and magical.

Pacing: Slow and steady, with clear pauses between ideas to give the child time to imagine each part of the story. Slightly rhythmic, like reading a picture book out loud.

Emotion: Cheerful, gentle, and nurturing, with a sense of wonder and delight. The voice often sounds curious and happy, as if discovering something amazing together with the listener.

Pronunciation: Clear and careful, slightly exaggerated for key words to help understanding. Names, actions, and important story elements are spoken distinctly and gently for easy comprehension.

Personality Affect: Kind, patient, and affectionate, like a trusted storyteller or preschool teacher. The voice feels safe and comforting, always guiding the child with encouragement and a sense of playful curiosity.
""")


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