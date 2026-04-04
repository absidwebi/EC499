"""
app.py
======
Flask deployment app for EC499 malware inference demo.
"""
"""
app.py
======
Stage 4 - Flask web API serving the malware inference demo.

Endpoints:
    GET  /          - Demo UI (index.html)
    GET  /health    - Server/model health check
    POST /predict   - PE file upload -> JSON prediction

Usage:
    python app.py
    # or with adversarial model:
    MODEL_TYPE=adversarial python app.py
"""

import os
import sys
import tempfile

from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import MalwareInferenceEngine, AdversarialComparisonEngine

app = Flask(__name__)

# 50 MB upload limit - matches the PE collection filter maximum
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Load model once at startup - not per request
_model_type = os.environ.get("MODEL_TYPE", "clean").lower()
_use_adv = _model_type == "adversarial"
engine = MalwareInferenceEngine(use_adversarial=_use_adv)
_model_label = "3c2d_adversarial" if _use_adv else "3c2d_clean"
print(f"[app] Model ready: {_model_label}")

# Load adversarial comparison engine (loads both clean and AT models)
# This is for the robustness demonstration endpoint
try:
    comparison_engine = AdversarialComparisonEngine()
    print(f"[app] Adversarial comparison engine ready")
except Exception as _e:
    comparison_engine = None
    print(f"[app] WARNING: Comparison engine unavailable: {_e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": _model_label}), 200


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_type=_model_label)


@app.route("/predict", methods=["POST"])
def predict():
    # --- Validate request ---
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "No file selected"}), 400

    # --- Determine safe temp file extension ---
    original_name = uploaded_file.filename
    _, ext = os.path.splitext(original_name)
    if not ext:
        ext = ".bin"

    # --- Save to temp, run inference, always clean up ---
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=ext, delete=False, dir="/tmp"
        ) as tmp:
            tmp_path = tmp.name
            uploaded_file.save(tmp_path)

        result = engine.predict(tmp_path)
        result["file_name"] = original_name   # show original name to user
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": f"Server error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.route("/compare", methods=["POST"])
def compare():
    """
    Adversarial robustness comparison endpoint.

    Accepts a PE file upload (expected to be a malware sample for the most
    compelling demo, but works with any valid PE).

    Returns:
        - byteplot_b64      : variable-size Nataraj visualization
        - clean_256_b64     : 256x256 normalized image (clean input)
        - adv_256_b64       : 256x256 normalized image (after PGD attack)
        - clean_model       : clean model predictions on both inputs
        - at_model          : AT model predictions on both inputs
        - attack_params     : PGD parameters used

    The expected demo result:
        clean_model.clean_input  -> malware (correct)
        clean_model.adv_input    -> benign  (fooled by PGD!)
        at_model.clean_input     -> malware (correct)
        at_model.adv_input       -> malware (defended!)
    """
    if comparison_engine is None:
        return jsonify({"error": "Adversarial comparison engine not available. "
                                 "Check that AT model weights exist."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "No file selected"}), 400

    original_name = uploaded_file.filename
    _, ext = os.path.splitext(original_name)
    if not ext:
        ext = ".bin"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=ext, delete=False, dir="/tmp"
        ) as tmp:
            tmp_path = tmp.name
            uploaded_file.save(tmp_path)

        result = comparison_engine.compare(tmp_path)
        result["file_name"] = original_name
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": f"Server error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum allowed size is 50 MB"}), 413


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[app] Starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
