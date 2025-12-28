import streamlit as st
import requests

st.title("趣味人格占卜")

st.warning("""
- 本應用不具備臨床或科學效力  
- 完全屬於娛樂性質  
- 目的是探索 AI 生成式解讀的趣味與可能性
""")

# Multiple file uploader
uploaded_files = st.file_uploader("請上傳圖片（可多選）", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if st.button("開始"):
    if not uploaded_files:
        st.error("請至少上傳一張圖片！")
    else:
        # Show loading animation
        with st.spinner("AI 正在占卜中…請稍候 ⏳"):
            # Prepare files for POST request
            files = [('images', (f.name, f, f.type)) for f in uploaded_files]

            # post to flask server
            response = requests.post(
                "http://127.0.0.1:8000/generate",
                files=files
            )

        if response.status_code == 200:
            ai_response = response.json().get("ai_response", "伺服器未返回內容。")
            st.success(ai_response)
        else:
            st.error("伺服器返回錯誤，請稍後再試。")
