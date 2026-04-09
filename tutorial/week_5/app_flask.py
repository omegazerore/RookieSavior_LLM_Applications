import base64
import os
import requests
import uuid
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from flask import Flask, request, render_template, jsonify, session
from langchain_core.runnables import chain

app = Flask(__name__)
app.secret_key = "super-secret-key"  # 會話管理需要密鑰

# 圖片儲存目錄
IMAGE_DIR = "image_psychic"
os.makedirs(IMAGE_DIR, exist_ok=True)  # 確保目錄存在，不存在則創建

@chain
def image_to_base64(file_storage: FileStorage):
    """將上傳的圖片檔案轉換為 Base64 編碼字串
    
    參數:
        file_storage (FileStorage): Werkzeug 檔案儲存物件
        
    返回:
        str: Base64 編碼的圖片字串
    """
    return base64.b64encode(file_storage.read()).decode("utf-8")

@chain
def build_image_prompt(image_str):
    """構建適合後端 AI 處理的圖片提示詞字典
    
    參數:
        image_str (str): Base64 編碼的圖片字串
        
    返回:
        dict: 包含格式化圖片資料的字典，適用於 AI 提示詞
    """
    return {
        "type": "image",
        "template": {"url": f"data:image/jpeg;base64,{image_str}"}
    }

@app.route("/generate", methods=["POST"])
def generate():
    """從上傳的圖片和使用者提示詞生成 AI 輸出
    
    此路由功能:
      1. 從會話中獲取上傳的圖片和文字提示詞
      2. 將圖片轉換為 Base64 並構建 AI 相容的資料負載
      3. 將資料發送到後端服務進行 AI 處理
      4. 以 JSON 格式返回 AI 生成的回應
      
    返回:
        flask.Response: 包含 AI 輸出訊息的 JSON 回應
        
    異常:
        Exception: 如果聯繫後端服務時出現問題
    """
    
    # 步驟 2: 構建 human_template 並呼叫後端
    # 圖像提示詞: 輸入的圖片
    image_files = request.files.getlist('images')  # 獲取所有上傳的圖片檔案

    if not image_files:
        return jsonify({"ai_response": "沒有上傳圖片。請先上傳圖片。"})

    # 創建圖片處理管道: 先轉 Base64，再構建提示詞
    image_transformation_pipeline_ = image_to_base64 | build_image_prompt

    # 建立模板
    human_template = []
    
    # 批量處理所有圖片檔案並添加到模板中
    human_template.extend(image_transformation_pipeline_.batch(image_files))

    # 構建發送到後端的資料負載
    payload = {
        "human": human_template,
    }

    # 發送到後端 AI 服務
    try:
        resp = requests.post(
            "http://localhost:5000/app_image_psychic/invoke",  # 後端服務端點
            json={"input": payload},  # 發送 JSON 資料
            timeout=180  # 設定 180 秒超時
        )
        # 從回應中提取 AI 生成的內容
        ai_response = resp.json().get("output", {}).get("content", "沒有回應")
    except Exception as e:
        ai_response = f"聯繫後端時發生錯誤: {e}"

    return jsonify({"ai_response": ai_response})  # 返回 JSON 格式的回應

if __name__ == "__main__":
    app.run(port=8000, debug=True)  # 啟動 Flask 應用，端口 8000，調試模式