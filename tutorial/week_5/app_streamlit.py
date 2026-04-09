# 導入必要的庫
import streamlit as st  # 用於建立網頁應用程式界面
import requests         # 用於發送 HTTP 請求

# 設定網頁標題
st.title("趣味人格占卜")

# 顯示警告訊息，說明應用的性質和限制
st.warning("""
- 本應用不具備臨床或科學效力  
- 完全屬於娛樂性質  
- 目的是探索 AI 生成式解讀的趣味與可能性
""")

# 建立多檔案上傳元件
# type: 限制只能上傳 png, jpg, jpeg 格式的檔案
# accept_multiple_files: 允許一次選擇多個檔案
uploaded_files = st.file_uploader("請上傳圖片（可多選）", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

# 建立按鈕，點擊後開始處理
if st.button("開始"):
    # 檢查是否有上傳檔案
    if not uploaded_files:
        # 如果沒有上傳檔案，顯示錯誤訊息
        st.error("請至少上傳一張圖片！")
    else:
        # 顯示載入動畫，讓使用者知道處理正在進行中
        with st.spinner("AI 正在占卜中…請稍候 ⏳"):
            # 準備檔案資料用於 POST 請求
            # 將每個上傳的檔案轉換為 requests 可用的格式
            # 格式: (欄位名稱, (檔案名稱, 檔案內容, 檔案類型))
            files = [('images', (f.name, f, f.type)) for f in uploaded_files]

            # 發送 POST 請求到 Flask 伺服器
            # 這裡假設在本地 8000 埠運行著一個 Flask 伺服器
            response = requests.post(
                "http://127.0.0.1:8000/generate",  # 伺服器端點
                files=files  # 上傳的檔案
            )

        # 檢查伺服器回應狀態碼
        if response.status_code == 200:
            # 如果請求成功 (200 OK)，從 JSON 回應中獲取 AI 回應內容
            ai_response = response.json().get("ai_response", "伺服器未返回內容。")
            # 顯示成功的 AI 回應
            st.success(ai_response)
        else:
            # 如果伺服器返回錯誤狀態碼，顯示錯誤訊息
            st.error("伺服器返回錯誤，請稍後再試。")
