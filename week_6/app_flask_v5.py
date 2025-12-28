import io
"""
透過Langserve 抽取飲料訊息
"""

import os
import requests

from openai import OpenAI
from flask import Flask, request, Response
from pydub import AudioSegment

from src.initialization import credential_init

credential_init()
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

app = Flask(__name__)


def call_tts(input_):
    response = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input=input_,
    instructions="Speak in a sweet, energetic, anime-girl style with a cute and playful tone.")

    # Get audio bytes
    audio_data = response.read()
    
    return audio_data


@app.route('/whisper', methods=['POST'])
def whisper():
    # Get raw binary data from request body
    audio_bytes = request.data

    # transform the received raw bytes into BytesIO
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "request.wav"
    
    whisper_output = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    response_format="text",
    prompt="用戶在點飲料, 微糖, 微冰",
)

    response = requests.post(
        "http://localhost:9000/drinking_app/invoke",
        json={'input': {"query": whisper_output}}
    )

    response.encoding = 'utf-8'
    # result = eval(response.text)
    # pass the output to langserve
    
    return response.text


@app.route('/tts', methods=['POST'])
def tts():
    
    query = request.data.decode("utf-8")
    audio_data = call_tts(input_=query)

    return Response(audio_data, mimetype="audio/mpeg")


if __name__ == "__main__":
    app.run(debug=True)