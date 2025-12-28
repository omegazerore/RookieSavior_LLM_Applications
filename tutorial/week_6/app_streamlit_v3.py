import io
import requests

import pandas as pd
import streamlit as st
from audiorecorder import audiorecorder

price_map = {"珍珠蜂蜜鮮奶普洱": 70,
         "茶凍奶綠": 50,
         "嚴選高山茶": 35,
         "咖啡奶茶": 75,
         "冬瓜檸檬": 60}

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

    order = eval(response.text)

    df = pd.DataFrame(order['output']['names'])
    df.rename(columns={'name': "品項", "ice_level": "冰度", "sugar_level": "糖度"}, inplace=True)
    
    st.dataframe(df)

    # 取得總價
    df['price'] = df['name'].map(price_map)

    total_price = df['price'].sum()
    
    st.success(f"總價為{total_price}元")
    