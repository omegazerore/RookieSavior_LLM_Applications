import io
import os

from openai import OpenAI
from flask import Flask, request
from pydub import AudioSegment

from src.initialization import credential_init

credential_init()
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
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
    
    return whisper_output

if __name__ == "__main__":
    app.run(debug=True)