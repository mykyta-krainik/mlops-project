"""
SageMaker real-time endpoint HTTP server.

SageMaker expects the container to listen on port 8080 with:
  GET  /ping        → 200 OK when ready
  POST /invocations → run inference, return JSON

This script wraps inference.py (model_fn / input_fn / predict_fn / output_fn)
behind a minimal Flask + gunicorn server.
"""

import os
import sys

sys.path.insert(0, "/app")

from flask import Flask, Response, jsonify, request

from src.inference import input_fn, model_fn, output_fn, predict_fn

app = Flask(__name__)

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
_model = None


def _load_model():
    global _model
    if _model is None:
        _model = model_fn(MODEL_DIR)
    return _model


try:
    _load_model()
    _ready = True
except Exception as e:
    print(f"Model load failed at startup: {e}", flush=True)
    _ready = False


@app.route("/ping", methods=["GET"])
def ping():
    if _ready:
        return Response("", status=200)
    return Response("Model not ready", status=503)


@app.route("/invocations", methods=["POST"])
def invocations():
    content_type = request.content_type or "application/json"
    accept = request.accept_mimetypes.best or "application/json"

    try:
        model = _load_model()
        data = input_fn(request.get_data(as_text=True), content_type)
        predictions = predict_fn(data, model)
        body, mime = output_fn(predictions, accept)
        return Response(body, status=200, mimetype=mime)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
