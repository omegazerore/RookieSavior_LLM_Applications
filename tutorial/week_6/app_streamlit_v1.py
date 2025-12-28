from io import BytesIO

import streamlit as st
from audiorecorder import audiorecorder

st.title("🎤 Voice Recorder Example")

# Record audio
# 第一個字串（"Click to record"）是在開始錄音之前顯示的標籤。
# 第二個字串（"Click to stop recording"）是在錄音進行中顯示的標籤，讓使用者知道可以停止錄音。
audio = audiorecorder("Click to record", "Click to stop recording")

if len(audio) > 0:
    
    # Convert audio to bytes
    wav_bytes = BytesIO()
    audio.export(wav_bytes, format="wav")

    # Play audio in Streamlit
    # st.audio(...) 是 Streamlit 提供的元件，會在頁面上渲染一個 HTML5 的音訊播放器。
    # 你給它的參數（這裡是 WAV 格式的 bytes）就是要播放的聲音檔。
    # Streamlit 自動把這些 bytes 包裝成 <audio controls> 標籤，於是 UI 上就有一個播放/暫停的功能。
    st.audio(wav_bytes.getvalue(), format="audio/wav")

    # Save audio
    with open("output.wav", "wb") as f:
        f.write(wav_bytes.getvalue())
        
    st.success("Audio saved as output.wav")