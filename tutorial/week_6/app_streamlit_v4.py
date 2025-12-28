"""
在app_streamlit_v3 中
顯示的表格是read only
我們想要擴展這個功能，讓用戶可以更改內容
畢竟我們無法保證TTS可以100%正確無誤
"""

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

# Define dropdown options for each column
column_config = {
    "品項": st.column_config.SelectboxColumn(
        "品項",
        options=["珍珠蜂蜜鮮奶普洱", "茶凍奶綠", "嚴選高山茶", "咖啡奶茶", "冬瓜檸檬"],
        required=True
    ),
    "冰度": st.column_config.SelectboxColumn(
        "冰度",
        options=['正常冰', '少冰', '微冰', '去冰'],
        required=True
    ),
    "糖度": st.column_config.SelectboxColumn(
        "糖度",
        options=['無糖', '微糖', '半糖' , '少糖'],
        required=True
    ),
    "價格": st.column_config.NumberColumn(
        "價格",
        disabled=True  # make this column read-only
    )
}


st.title("🎤 語音定飲料系統")

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

    df["價格"] = df["品項"].map(price_map)
    
    # Editable table
    edited_df = st.data_editor(
        df,
        column_config=column_config,
        num_rows="dynamic",   # allow adding new rows
        use_container_width=True
    )

    # Recalculate "價格" in case user changes "品項"
    edited_df["價格"] = edited_df["品項"].map(price_map)

    st.write("✅ 最終訂單：")
    st.dataframe(edited_df, use_container_width=True)

    