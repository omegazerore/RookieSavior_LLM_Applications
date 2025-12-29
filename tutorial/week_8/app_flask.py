import base64
import requests

from flask import Flask, request, Response, jsonify

app = Flask(__name__)


audio_endpoint = "http://localhost:8080/audio_generation/invoke"
story_endpoint = "http://localhost:8080/story_telling/invoke"
image_endpoint = "http://localhost:8080/image_generation/invoke"

@app.route('/tts', methods=['POST'])
def tts():

    data = request.get_json(force=True)  # safely parse incoming JSON
    
    # adapt from Week-6
    # call the langserve 
    try:
        response = requests.post(
            audio_endpoint,
            json={"input": data},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return jsonify({"error": f"Langserve call failed: {e}", "inputs": input_}), 500

    output = result['output']

    return jsonify({"output": output}), 200
    # decode the base64 to bytes
    # audio_bytes = base64.b64decode(output)

    # # pass the audio back to streamlit UI
    # return Response(audio_bytes, mimetype="audio/mpeg")


@app.route("/story", methods=["POST"])
def story():

    data = request.get_json(force=True)  # safely parse incoming JSON
    
    try:
        response = requests.post(
            story_endpoint,
            json={"input": data},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        jsonify({"error": f"Langserve call failed: {e}", "inputs": {"input": scratch, "context": context}}), 500

    # `output` is a python string object
    output = result['output']

    return jsonify({
        "output": output,
    }), 200


@app.route("/image", methods=["POST"])
def image():

    data = request.get_json(force=True)  # safely parse incoming JSON

    try:
        response = requests.post(
            image_endpoint,
            json={"input": data},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return jsonify({"error": f"Langserve call failed: {e}", "inputs": data}), 500

    output = result['output']

    return jsonify({"output": output}), 200
    
    # image_bytes = base64.b64decode(output['image_base64'])

    # return Response(image_bytes, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)