"""
Content for Week-7
"""
import base64
import io
import argparse
from flask import Flask, request, jsonify
from PIL import Image
from tagger.interrogator.interrogator import AbsInterrogator
from tagger.interrogators import interrogators

# ----------------------------------
# Parse command-line arguments
# ----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="wd14-convnextv2.v1",
    help="Model name from tagger.interrogators"
)
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Force CPU mode"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.35,
    help="Default threshold for tagging"
)
args = parser.parse_args()

# ----------------------------------
# Load model once at startup
# ----------------------------------
MODEL_NAME = args.model
THRESHOLD = args.threshold

print(f"Loading model: {MODEL_NAME}")
interrogator = interrogators[MODEL_NAME]

if args.cpu:
    interrogator.use_cpu()
    print("Using CPU mode")

# ----------------------------------
# Flask app
# ----------------------------------
app = Flask(__name__)

def decode_base64_image(base64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes))


def process_tags(image: Image.Image,
                 threshold: float = THRESHOLD,
                 exclude_tags=None,
                 additional_tags=None,
                 escape_tags=True):
    exclude_tags = exclude_tags or []
    additional_tags = additional_tags or []

    raw_result = interrogator.interrogate(image)

    tags = AbsInterrogator.postprocess_tags(
        raw_result[1],
        threshold=threshold,
        escape_tag=escape_tags,
        replace_underscore=escape_tags,
        exclude_tags=set(exclude_tags),
        additional_tags=additional_tags,
    )

    return tags

@app.route("/tag", methods=["POST"])
def tag_image():
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image'"}), 400

    try:
        img = decode_base64_image(data["image"])
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    tags = process_tags(
        img,
        threshold=float(data.get("threshold", THRESHOLD)),
        exclude_tags=data.get("exclude_tags", []),
        additional_tags=data.get("additional_tags", []),
        escape_tags=bool(data.get("escape_tags", True)),
    )

    return jsonify({
        "tags": tags,
        "tag_list": list(tags.keys()),
        "model": MODEL_NAME
    })

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)