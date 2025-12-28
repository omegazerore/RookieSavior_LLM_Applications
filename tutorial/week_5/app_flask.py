import base64
import os

import requests
import uuid
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from flask import Flask, request, render_template, jsonify, session
from langchain_core.runnables  import chain

app = Flask(__name__)
app.secret_key = "super-secret-key"  # Needed for session

IMAGE_DIR = "image_psychic"
os.makedirs(IMAGE_DIR, exist_ok=True)

@chain
def image_to_base64(file_storage: FileStorage):

    return base64.b64encode(file_storage.read()).decode("utf-8")


@chain
def build_image_prompt(image_str):
    """Constructs an image prompt dictionary suitable for backend AI processing.

    Args:
        image_str (str): Base64-encoded image string.

    Returns:
        dict: A dictionary containing the image data formatted for an AI prompt.
    """

    return {"type": "image",
            "template": {"url": f"data:image/jpeg;base64,{image_str}"}}



@app.route("/generate", methods=["POST"])
def generate():
    """Generates AI output from the uploaded images and user prompt.

    This route:
      1. Retrieves uploaded images and the text prompt from the session.
      2. Converts images to Base64 and constructs an AI-compatible payload.
      3. Sends the data to a backend service for AI processing.
      4. Returns the AI-generated response as JSON.

    Returns:
        flask.Response: A JSON response containing the AI's output message.

    Raises:
        Exception: If there is an issue contacting the backend service.
    """

    """Step 2: Build human_template and call backend"""
    # 圖像提示詞: 輸入的圖片
    image_files = request.files.getlist('images')

    if not image_files:
        return jsonify({"ai_response": "No uploaded images. Please upload first."})

    image_transformation_pipeline_ = image_to_base64|build_image_prompt

    # 建立模板
    human_template = []
    
    human_template.extend(image_transformation_pipeline_.batch(image_files))

    payload = {
        "human": human_template,
    }

    # Send to backend AI service
    try:
        resp = requests.post(
            "http://localhost:5000/app_image_psychic/invoke",
            json={"input": payload},
            timeout=180
        )
        ai_response = resp.json().get("output", {}).get("content", "No response")
    except Exception as e:
        ai_response = f"Error contacting backend: {e}"

    return jsonify({"ai_response": ai_response})


if __name__ == "__main__":
    app.run(port=8000, debug=True)