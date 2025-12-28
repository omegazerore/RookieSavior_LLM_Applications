import io
"""
透過Langserve 抽取飲料訊息
"""

import os
import requests

from openai import OpenAI
from flask import Flask, request
from pydub import AudioSegment

from src.initialization import credential_init

credential_init()
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

app = Flask(__name__)


@app.route('/whisper', methods=['POST'])
def whisper():
    # Get raw binary data from request body
    audio_bytes = request.data

    # transform the received raw bytes into BytesIO
    audio_file = io.BytesIO(audio_bytes)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "request.wav"
    
    whisper_output = client.audio.translations.create(
    model="whisper-1", 
    file=audio_file,
    response_format="text",
    prompt="用戶在點飲料, 微糖, 微冰",
)

    response = requests.post(
        "http://localhost:8080/drinking_app/invoke",
        json={'input': {"query": whisper_output}}
    )

    response.encoding = 'utf-8'
    # result = eval(response.text)
    # pass the output to langserve
    
    return response.text

if __name__ == "__main__":
    app.run(debug=True)