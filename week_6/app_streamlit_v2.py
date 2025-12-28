import io
import requests

import streamlit as st
from audiorecorder import audiorecorder

st.title("🎤 Voice Recorder Example")

# Record audio
# 第一個字串（"Click to record"）是在開始錄音之前顯示的標籤。
# 第二個字串（"Click to stop recording"）是在錄音進行中顯示的標籤，讓使用者知道可以停止錄音。
audio = audiorecorder("Click to record", "Click to stop recording")

if len(audio) > 0:
    
    # Convert audio to bytes
    wav_bytes = io.BytesIO()
    audio.export(wav_bytes, format="wav")
    wav_bytes.seek(0)  # reset cursor

    # Send via requests as raw binary data
    
    response = requests.post('http://127.0.0.1:5000/upload', 
                             data=wav_bytes.getvalue(), # <-- this is now raw bytes, not BytesIO
                             headers={'Content-Type': 'audio/wav'})

    st.success(f"Server response: {response.text}")
    