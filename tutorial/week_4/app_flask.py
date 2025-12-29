import io
import json
import os
import requests

from flask import Flask, request, Response

app = Flask(__name__)

langserve_url = "http://localhost:9001/chatbot/invoke"

@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get raw binary data from request body
    data = request.get_json()

    # Deserialize
    chat_history = data.get("chat_history")
    
    response = requests.post(
        langserve_url,
        json={'input': {"messages": chat_history}}
    )

    response.encoding = 'utf-8'
    
    return response.json()['output']


if __name__ == "__main__":
    app.run(debug=True)