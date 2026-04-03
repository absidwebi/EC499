# EC499 — Stage 4: Inference Pipeline & Deployment Demo
## Agent Implementation Guide

**Project:** Adversarial Robustness in Deep Learning-Based Malware Detection  
**Student:** Abdulsalam Ashraf Aldwebi (ID: 2210245306)  
**Supervisor:** Dr. Suad Elgeder — University of Tripoli  
**Environment:** Ubuntu Linux, RTX 4060, Python 3.10  
**Virtual Environment:** `/home/alucard-00/EC499/Project_Resourse/venv/`  
**Always use this interpreter:** `/home/alucard-00/EC499/Project_Resourse/venv/bin/python`

---

## CRITICAL SYSTEM NOTE — READ BEFORE STARTING ANY TASK

If you are reading this document after a VS Code session restart, the reason is most
likely **GPU memory pressure**. The project performs adversarial training using PGD
which occupies nearly all of the RTX 4060's 8 GB of VRAM. When the system is under
that load, VS Code may restart the extension host or lose the terminal session. This
is **not a code error**. You should:

1. Check whether training is still running: `ps aux | grep python`
2. If training is running, wait until it finishes or run inference work in a separate
   terminal that does not touch the GPU.
3. All Stage 4 inference work is CPU-only. It will not compete with the GPU training
   process and can safely run in parallel.

---

## PEFILE LIBRARY — STATIC ANALYSIS CLARIFICATION

This is important for understanding the project's scientific validity.

**pefile is a pure Python static parser. It does NOT execute PE files.**

When you call `pefile.PE(path, fast_load=True)`, the library:
- Opens the file in binary read mode
- Reads and parses the DOS header, NT headers, section table, and optional data
- Validates structural fields like `e_magic` (the MZ signature, value `0x5A4D`)
- Closes the file handle

It never allocates executable memory, never calls `CreateProcess`, never invokes the
Windows loader, and never triggers any code inside the file. The binary is treated as
structured data, exactly like parsing a PNG or ZIP file.

**The project proposal explicitly requires static analysis only.** pefile satisfies
this requirement completely. This is the same approach used in the MaleX paper and
in the benign collection script `collect_benign_pe.py` already in the codebase.

**Windows PE files on Linux with pefile:**
pefile works correctly on Linux with Windows PE files. PE (Portable Executable) is
a file format specification. pefile reads the format — it does not need Windows to
run, does not need Wine, and does not attempt execution. A Windows `.exe` or `.dll`
file parsed by pefile on Ubuntu is safe and produces correct structural results.

Therefore: **The `pe_test` folder containing Windows PE files can be validated by
pefile on this Linux system without any problems.**

**Why is PE validation disabled by default in the current `inference.py`?**
The agent added an `ENABLE_PE_VALIDATION` environment variable toggle. This was a
precautionary workaround. Based on the analysis above, this is unnecessary. The
fix section below explains how to resolve this properly.

---

## PROJECT CONTEXT — WHAT IS ALREADY DONE

Read these files to understand the current state before doing anything:
- `/home/alucard-00/EC499/Project_Resourse/PROJECT_CONTEXT.md`
- `/home/alucard-00/EC499/Project_Resourse/CURRENT_STATE.md`
- `/home/alucard-00/EC499/Project_Resourse/MASTER_CONTEXT.md`
- `/home/alucard-00/EC499/EC499_Folder_Structure.md`

**Stage completion summary:**
- Stage 1 (Dataset): DONE — MaleX split at `archive/malex_dataset/`
- Stage 2 (Base models): DONE — 3C2D selected, weights at `models/3c2d_malex_clean_vulnerable.pth`
- Stage 3 (Adversarial): DONE — AT weights at `models/3c2d_malex_adversarially_trained.pth`
- Stage 4 (Inference/Demo): DONE — implemented and validated (see status block below)

**Key model facts:**
- Architecture: `MaleX3C2D` defined in `Project_Resourse/models.py`
- Input shape: `[1, 1, 256, 256]` (batch, channels, height, width)
- Output: single logit → `sigmoid(logit)` gives confidence → threshold at 0
- Loss used during training: `BCEWithLogitsLoss`
- Normalization: `mean=0.5, std=0.5` → pixel range `[-1, 1]`
- Map location for CPU inference: always use `map_location='cpu'`

**Key existing scripts to REUSE, never rewrite:**
- `config.py` — all path constants
- `models.py` — `MaleX3C2D` class
- `convert_to_malimg.py` — `get_nataraj_width()` and `pe_to_nataraj_image()`
- `dataset_loader.py` — normalization reference (mean=0.5, std=0.5)
- `collect_benign_pe.py` — PE validation pattern with pefile

---

## STAGE 4 DELIVERABLES

| # | File | Purpose | Status |
|---|------|---------|--------|
| 1 | `inference.py` | PE → prediction pipeline module | Implemented and validated |
| 2 | `app.py` | Flask web API server | Implemented and validated |
| 3 | `templates/index.html` | Demo UI | Implemented and validated |
| 4 | `Dockerfile` | Isolated container for deployment | Implemented and validated |

### Current verified status (2026-04-04)

- Local API checks passed for `/health`, valid PE `/predict`, and invalid input error handling.
- Docker tooling is available on host and container flow has been exercised.
- Residual caveat remains for strict reproduction under isolated `--network=none` host reachability behavior.
- Canonical project state for Stage 4 is tracked in `PROJECT_CONTEXT.md`, `CURRENT_STATE.md`, and `MASTER_CONTEXT.md`.

---

## DEPENDENCY VERIFICATION — RUN FIRST

Before touching any code, verify all dependencies. Run each command and report output.

```bash
# Check Python version
/home/alucard-00/EC499/Project_Resourse/venv/bin/python --version

# Check OpenCV
/home/alucard-00/EC499/Project_Resourse/venv/bin/pip show opencv-python-headless

# Check pefile
/home/alucard-00/EC499/Project_Resourse/venv/bin/pip show pefile

# Check Flask
/home/alucard-00/EC499/Project_Resourse/venv/bin/pip show flask

# Check PyTorch (confirm it loads)
/home/alucard-00/EC499/Project_Resourse/venv/bin/python -c "import torch; print(torch.__version__)"

# Check that MaleX3C2D model file exists
ls -lh /home/alucard-00/EC499/Project_Resourse/models/3c2d_malex_clean_vulnerable.pth
ls -lh /home/alucard-00/EC499/Project_Resourse/models/3c2d_malex_adversarially_trained.pth
```

**If opencv-python-headless is missing:**
```bash
/home/alucard-00/EC499/Project_Resourse/venv/bin/pip install opencv-python-headless
```

**If pefile is missing:**
```bash
/home/alucard-00/EC499/Project_Resourse/venv/bin/pip install pefile
```

**If flask is missing:**
```bash
/home/alucard-00/EC499/Project_Resourse/venv/bin/pip install flask
```

**After installing any package**, add it to `requirements.txt` if not already present:
```
flask>=2.0
opencv-python-headless>=4.5
pefile>=2022.5.30
Pillow>=8.0
```

---

## TASK 1 — Fix and Finalize `inference.py`

### Current state

An `inference.py` has been created but has one significant problem:
**PE validation is disabled by default** via the `ENABLE_PE_VALIDATION` environment
variable bypass. This must be fixed. PE validation is safe, correct, and required.

### Required fix to `inference.py`

Replace the `validate_and_read_bytes` function entirely. Remove all traces of the
`ENABLE_PE_VALIDATION` bypass. The corrected version is:

```python
def validate_and_read_bytes(file_path: str) -> bytes:
    """
    Validate that file_path is a PE executable and return its raw bytes.

    Uses pefile for structural validation only — no execution, no code runs.
    pefile is a pure Python static parser. Windows PE files work on Linux.

    Raises:
        ValueError: if the file is not a valid PE executable.
    """
    try:
        pe = pefile.PE(file_path, fast_load=True)
        if pe.DOS_HEADER.e_magic != 0x5A4D:
            pe.close()
            raise ValueError("Not a valid PE file: missing MZ header signature")
        pe.close()
    except ValueError:
        raise
    except pefile.PEFormatError as exc:
        raise ValueError(f"Not a valid PE file: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Not a valid PE file: {exc}") from exc

    with open(file_path, "rb") as f:
        return f.read()
```

### Complete `inference.py` — authoritative version

Write this file exactly as shown to
`/home/alucard-00/EC499/Project_Resourse/inference.py`:

```python
"""
inference.py
============
Stage 4 — Single entry point for the PE file → MaleX3C2D prediction pipeline.

Pipeline:
    PE file → pefile validation (static only) → raw bytes → Nataraj array
    → cv2.INTER_AREA resize to 256×256 → normalized tensor → 3C2D model → prediction

Scientific notes:
    - pefile performs static structural analysis only. No code is executed.
    - Windows PE files work correctly on Linux with pefile.
    - The resize method (cv2.INTER_AREA) matches how MaleX dataset images were
      prepared, ensuring inference distribution matches training distribution.
    - Normalization (mean=0.5, std=0.5) matches dataset_loader.py exactly.
    - Model loaded with map_location='cpu' — no GPU required for inference.
"""

import os
import sys
import math
import base64
from io import BytesIO
from typing import Dict

import cv2
import pefile
import numpy as np
from PIL import Image
import torch

# Ensure project modules import correctly regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MALEX_3C2D_CLEAN_MODEL_PATH_STR,
    MALEX_3C2D_ADV_MODEL_PATH_STR,
)
from convert_to_malimg import get_nataraj_width
from models import MaleX3C2D


# ---------------------------------------------------------------------------
# Component 1 — PE Validation and Raw Byte Reading
# ---------------------------------------------------------------------------

def validate_and_read_bytes(file_path: str) -> bytes:
    """
    Validate that file_path is a PE executable and return its raw bytes.

    Uses pefile for structural validation only — no execution occurs.
    pefile is a pure Python static parser. Windows PE files work on Linux.

    Raises:
        ValueError: if the file is not a valid PE executable.
    """
    try:
        pe = pefile.PE(file_path, fast_load=True)
        if pe.DOS_HEADER.e_magic != 0x5A4D:
            pe.close()
            raise ValueError("Not a valid PE file: missing MZ header signature")
        pe.close()
    except ValueError:
        raise
    except pefile.PEFormatError as exc:
        raise ValueError(f"Not a valid PE file: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Not a valid PE file: {exc}") from exc

    with open(file_path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Component 2 — Nataraj Byte-to-Image Conversion
# ---------------------------------------------------------------------------

def pe_bytes_to_nataraj_array(raw_bytes: bytes) -> np.ndarray:
    """
    Convert PE raw bytes to variable-size Nataraj grayscale array.

    Uses the Nataraj width table from convert_to_malimg.py to determine
    image dimensions from file size, then reshapes bytes as pixels.
    This produces a variable-size image matching the Malimg/MaleX methodology.
    """
    file_size = len(raw_bytes)
    width = get_nataraj_width(file_size)
    height = math.ceil(file_size / width)

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    total_pixels = width * height

    if arr.size < total_pixels:
        arr = np.pad(arr, (0, total_pixels - arr.size), mode="constant")
    else:
        arr = arr[:total_pixels]

    return arr.reshape((height, width)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Component 3 — Resize to 256×256 (MaleX-faithful method)
# ---------------------------------------------------------------------------

def resize_to_256(nataraj_array: np.ndarray) -> np.ndarray:
    """
    Resize variable-size Nataraj array to 256×256 using cv2.INTER_AREA.

    INTER_AREA is the correct interpolation for downsampling because it
    averages pixel values in each source region, preserving the statistical
    distribution of byte intensities. This matches the method used when
    the MaleX dataset images were prepared (opencv-python==4.5.4.60,
    confirmed from the MaleX repository requirements.txt).

    Using any other interpolation (bilinear, nearest) would cause a
    distribution mismatch between inference and training data.
    """
    return cv2.resize(nataraj_array, (256, 256), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Component 4 — Tensor Conversion (matching dataset_loader.py normalization)
# ---------------------------------------------------------------------------

def array_to_tensor(resized_array: np.ndarray) -> torch.Tensor:
    """
    Convert 256×256 uint8 array to normalized FloatTensor [1, 1, 256, 256].

    Normalization: (pixel/255 - 0.5) / 0.5 → range [-1, 1]
    This exactly matches the Normalize(mean=[0.5], std=[0.5]) transform
    in dataset_loader.py used during training.
    """
    x = resized_array.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.expand_dims(x, axis=0)   # [256, 256] → [1, 256, 256]
    x = np.expand_dims(x, axis=0)   # [1, 256, 256] → [1, 1, 256, 256]
    return torch.from_numpy(x).float()


# ---------------------------------------------------------------------------
# Component 5 — Base64 PNG Encoding for display
# ---------------------------------------------------------------------------

def array_to_png_base64(array: np.ndarray) -> str:
    """
    Convert a grayscale numpy array to a base64-encoded PNG string.

    The caller should pass the variable-size Nataraj array (before resize)
    for richer visual display — the natural proportions reflect actual file
    size and are more visually meaningful for the demo.

    The base64 string can be embedded directly in an HTML <img> tag:
        <img src="data:image/png;base64,{image_b64}">
    """
    img = Image.fromarray(array.astype(np.uint8), mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Inference Engine — loads model once, serves predictions
# ---------------------------------------------------------------------------

class MalwareInferenceEngine:
    """
    Loads MaleX3C2D weights once at startup and serves predictions.

    Usage:
        engine = MalwareInferenceEngine()                    # clean model
        engine = MalwareInferenceEngine(use_adversarial=True) # AT model
        result = engine.predict("/path/to/file.exe")
    """

    def __init__(self, model_path: str = None, use_adversarial: bool = False):
        if model_path is None:
            model_path = (
                MALEX_3C2D_ADV_MODEL_PATH_STR
                if use_adversarial
                else MALEX_3C2D_CLEAN_MODEL_PATH_STR
            )

        self.model_path = str(model_path)
        self._model_type = "adversarial" if use_adversarial else "clean"

        self.model = MaleX3C2D()
        state = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[inference] Loaded {self._model_type} model from {self.model_path}")

    @property
    def model_type(self) -> str:
        return self._model_type

    def predict(self, file_path: str) -> Dict[str, object]:
        """
        Run the full PE file → prediction pipeline.

        Args:
            file_path: absolute path to the PE file to analyze.

        Returns:
            dict with keys:
                label       : "malware" or "benign"
                confidence  : float [0, 1] — sigmoid of logit
                logit       : raw model output (positive = malware)
                image_b64   : base64 PNG of the Nataraj byteplot visualization
                file_name   : basename of the analyzed file
                model_type  : "clean" or "adversarial"

        Raises:
            ValueError   : file is not a valid PE executable
            RuntimeError : any other error in the pipeline
        """
        try:
            raw_bytes     = validate_and_read_bytes(file_path)
            nataraj_array = pe_bytes_to_nataraj_array(raw_bytes)
            image_b64     = array_to_png_base64(nataraj_array)   # variable-size for display
            resized_array = resize_to_256(nataraj_array)
            tensor        = array_to_tensor(resized_array)

            with torch.no_grad():
                logit = self.model(tensor).item()

            confidence = float(torch.sigmoid(torch.tensor(logit)).item())
            label = "malware" if logit > 0 else "benign"

            return {
                "label":      label,
                "confidence": round(confidence, 4),
                "logit":      round(logit, 4),
                "image_b64":  image_b64,
                "file_name":  os.path.basename(file_path),
                "model_type": self._model_type,
            }

        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Inference pipeline error: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py /path/to/file.exe [--adversarial]")
        sys.exit(1)

    test_path = sys.argv[1]
    use_adv   = "--adversarial" in sys.argv

    engine = MalwareInferenceEngine(use_adversarial=use_adv)
    try:
        result = engine.predict(test_path)
        print("\n=== Prediction Result ===")
        print(f"File      : {result['file_name']}")
        print(f"Label     : {result['label'].upper()}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Logit     : {result['logit']}")
        print(f"Model     : {result['model_type']}")
        print(f"Image b64 : {result['image_b64'][:60]}... (truncated)")
    except ValueError as e:
        print(f"[VALIDATION ERROR] {e}")
    except RuntimeError as e:
        print(f"[PIPELINE ERROR] {e}")
```

### Test Task 1 after writing

Run these tests and report the exact output of each:

```bash
# Test 1 — Module import (no file needed)
/home/alucard-00/EC499/Project_Resourse/venv/bin/python -c "
import sys
sys.path.insert(0, '/home/alucard-00/EC499/Project_Resourse')
from inference import MalwareInferenceEngine
print('Import OK')
engine = MalwareInferenceEngine()
print('Model loaded OK, type:', engine.model_type)
"

# Test 2 — Valid PE file from benign collection
/home/alucard-00/EC499/Project_Resourse/venv/bin/python \
  /home/alucard-00/EC499/Project_Resourse/inference.py \
  /home/alucard-00/EC499/benign_pe_files/benign_00001.exe

# Test 3 — Non-PE file (should raise ValueError)
echo "not a pe file" > /tmp/test_not_pe.txt
/home/alucard-00/EC499/Project_Resourse/venv/bin/python \
  /home/alucard-00/EC499/Project_Resourse/inference.py \
  /tmp/test_not_pe.txt
# Expected: [VALIDATION ERROR] Not a valid PE file: ...

# Test 4 — Windows PE from pe_test folder (if it exists)
# pefile works with Windows PEs on Linux — no execution occurs
ls /home/alucard-00/EC499/pe_test/ 2>/dev/null && \
  /home/alucard-00/EC499/Project_Resourse/venv/bin/python \
    /home/alucard-00/EC499/Project_Resourse/inference.py \
    "$(ls /home/alucard-00/EC499/pe_test/*.exe 2>/dev/null | head -1)"
```

**Pass criteria for Task 1:**
- Import completes without error
- Test 2 prints label (malware or benign), confidence between 0 and 1, non-empty image_b64
- Test 3 prints `[VALIDATION ERROR]` not a stack trace crash
- Do NOT proceed to Task 2 until all pass criteria are met

---

## TASK 2 — Create `app.py`

Write `/home/alucard-00/EC499/Project_Resourse/app.py`:

```python
"""
app.py
======
Stage 4 — Flask web API serving the malware inference demo.

Endpoints:
    GET  /          — Demo UI (index.html)
    GET  /health    — Server/model health check
    POST /predict   — PE file upload → JSON prediction

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
from inference import MalwareInferenceEngine

app = Flask(__name__)

# 50 MB upload limit — matches the PE collection filter maximum
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Load model once at startup — not per request
_model_type  = os.environ.get("MODEL_TYPE", "clean").lower()
_use_adv     = _model_type == "adversarial"
engine       = MalwareInferenceEngine(use_adversarial=_use_adv)
_model_label = "3c2d_adversarial" if _use_adv else "3c2d_clean"
print(f"[app] Model ready: {_model_label}")


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
```

### Test Task 2 after writing

Open a second terminal. Start the server:
```bash
cd /home/alucard-00/EC499/Project_Resourse
/home/alucard-00/EC499/Project_Resourse/venv/bin/python app.py
```

In a third terminal (or wait for output and use &), run:
```bash
# Test /health
curl -s http://localhost:5000/health

# Expected: {"model":"3c2d_clean","status":"ok"}

# Test /predict with a benign PE
curl -s -X POST \
  -F "file=@/home/alucard-00/EC499/benign_pe_files/benign_00001.exe" \
  http://localhost:5000/predict | python3 -m json.tool

# Test /predict with a non-PE (expect 400 error)
echo "this is not a PE" > /tmp/fake.exe
curl -s -X POST \
  -F "file=@/tmp/fake.exe" \
  http://localhost:5000/predict | python3 -m json.tool

# Expected for non-PE: {"error": "Not a valid PE file: ..."}
```

**Pass criteria for Task 2:**
- `/health` returns `{"status":"ok", "model":"3c2d_clean"}`
- `/predict` with valid PE returns JSON with `label`, `confidence`, `logit`, `image_b64`
- `/predict` with non-PE returns `{"error": "Not a valid PE file..."}` with status 400
- No 500 errors, no Python stack traces in the terminal

---

## TASK 3 — Create `templates/index.html`

First create the templates directory:
```bash
mkdir -p /home/alucard-00/EC499/Project_Resourse/templates
```

Write `/home/alucard-00/EC499/Project_Resourse/templates/index.html`.

This is a single self-contained HTML file. No CDN, no external dependencies.
Everything must work offline (for the Docker isolated environment).

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EC499 — Malware Detection System</title>
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --muted: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --blue: #58a6ff;
    --accent: #1f6feb;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace;
    min-height: 100vh;
    padding: 2rem 1rem;
  }

  .container { max-width: 900px; margin: 0 auto; }

  header { text-align: center; margin-bottom: 2.5rem; }
  header h1 { font-size: 1.8rem; color: var(--blue); letter-spacing: 0.05em; }
  header p  { color: var(--muted); margin-top: 0.4rem; font-size: 0.9rem; }
  header .badge {
    display: inline-block; margin-top: 0.8rem;
    background: var(--surface); border: 1px solid var(--border);
    color: var(--muted); font-size: 0.75rem; padding: 0.2rem 0.7rem;
    border-radius: 99px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
  }
  .card h2 { font-size: 1rem; color: var(--text); margin-bottom: 1.2rem; }

  .upload-zone {
    border: 2px dashed var(--border);
    border-radius: 6px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.2s;
    cursor: pointer;
  }
  .upload-zone:hover { border-color: var(--accent); }
  .upload-zone input[type=file] { display: none; }
  .upload-zone .icon { font-size: 2.5rem; margin-bottom: 0.6rem; }
  .upload-zone .hint { color: var(--muted); font-size: 0.85rem; }
  .upload-zone .selected-name {
    margin-top: 0.8rem;
    color: var(--blue);
    font-size: 0.9rem;
    word-break: break-all;
  }

  .btn {
    display: inline-block;
    padding: 0.6rem 1.4rem;
    border-radius: 6px;
    border: none;
    cursor: pointer;
    font-size: 0.95rem;
    font-weight: 600;
    transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-secondary { background: var(--surface); color: var(--text);
                   border: 1px solid var(--border); margin-left: 0.8rem; }

  .controls { margin-top: 1.2rem; display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }

  .toggle-group { display: flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  .toggle-group label {
    padding: 0.4rem 1rem; cursor: pointer; font-size: 0.85rem; color: var(--muted);
    transition: background 0.15s, color 0.15s;
  }
  .toggle-group input[type=radio] { display: none; }
  .toggle-group input[type=radio]:checked + label {
    background: var(--accent); color: #fff;
  }

  /* Result banner */
  .result-banner {
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 1.2rem;
    text-align: center;
  }
  .result-banner.malware { background: rgba(248,81,73,0.15); color: var(--red);
                           border: 1px solid var(--red); }
  .result-banner.benign  { background: rgba(63,185,80,0.15); color: var(--green);
                           border: 1px solid var(--green); }

  .meta-row { display: flex; gap: 2rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
  .meta-item { flex: 1; min-width: 140px; }
  .meta-item .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase;
                      letter-spacing: 0.06em; margin-bottom: 0.3rem; }
  .meta-item .value { font-size: 1rem; color: var(--text); font-weight: 600; }

  .progress-bar-track {
    background: var(--border); border-radius: 99px; height: 10px;
    overflow: hidden; margin-top: 0.4rem;
  }
  .progress-bar-fill {
    height: 100%; border-radius: 99px; transition: width 0.5s ease;
  }
  .progress-bar-fill.malware { background: var(--red); }
  .progress-bar-fill.benign  { background: var(--green); }

  .vis-section { margin-top: 1.2rem; }
  .vis-section h3 { font-size: 0.85rem; color: var(--muted); margin-bottom: 0.6rem; }
  .vis-section img {
    max-width: 100%; border: 1px solid var(--border); border-radius: 4px;
    image-rendering: pixelated;
  }
  .vis-caption { color: var(--muted); font-size: 0.78rem; margin-top: 0.5rem; }

  /* Spinner */
  .spinner-wrap { text-align: center; padding: 2rem; display: none; }
  .spinner {
    display: inline-block; width: 2.5rem; height: 2.5rem;
    border: 3px solid var(--border); border-top-color: var(--blue);
    border-radius: 50%; animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner-wrap p { color: var(--muted); margin-top: 0.8rem; font-size: 0.9rem; }

  /* Error */
  .error-box {
    background: rgba(248,81,73,0.1); border: 1px solid var(--red);
    border-radius: 6px; padding: 1rem 1.2rem; color: var(--red);
    display: none; margin-bottom: 1rem;
  }

  #resultCard { display: none; }
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>⚡ EC499 — Malware Detection System</h1>
    <p>Adversarial Robustness in Deep Learning-Based Malware Detection</p>
    <span class="badge" id="modelBadge">Model: {{ model_type }}</span>
  </header>

  <!-- Upload Card -->
  <div class="card">
    <h2>Analyze a PE File</h2>

    <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
      <div class="icon">📂</div>
      <p>Click to select a PE file (.exe, .dll, .sys, .ocx)</p>
      <p class="hint">Maximum 50 MB · Static analysis only · No execution</p>
      <p class="selected-name" id="selectedName"></p>
    </div>
    <input type="file" id="fileInput" accept=".exe,.dll,.sys,.ocx,.bin">

    <div class="controls">
      <button class="btn btn-primary" id="analyzeBtn" disabled onclick="runPrediction()">
        🔍 Analyze File
      </button>
      <button class="btn btn-secondary" id="resetBtn" onclick="resetForm()" style="display:none">
        ↺ Analyze Another File
      </button>
    </div>
  </div>

  <!-- Spinner -->
  <div class="spinner-wrap" id="spinnerWrap">
    <div class="spinner"></div>
    <p>Analyzing binary structure…</p>
  </div>

  <!-- Error -->
  <div class="error-box" id="errorBox"></div>

  <!-- Result Card -->
  <div class="card" id="resultCard">
    <div class="result-banner" id="resultBanner"></div>

    <div class="meta-row">
      <div class="meta-item">
        <div class="label">File</div>
        <div class="value" id="resFileName"></div>
      </div>
      <div class="meta-item">
        <div class="label">Confidence</div>
        <div class="value" id="resConfidenceText"></div>
        <div class="progress-bar-track">
          <div class="progress-bar-fill" id="resConfidenceBar"></div>
        </div>
      </div>
      <div class="meta-item">
        <div class="label">Raw Logit</div>
        <div class="value" id="resLogit" style="font-family:monospace;font-size:0.9rem"></div>
      </div>
    </div>

    <div class="vis-section">
      <h3>Binary Visualization — Nataraj Byteplot</h3>
      <img id="visImage" src="" alt="Byteplot visualization">
      <p class="vis-caption">
        Each pixel represents one raw byte of the executable (0–255).
        Visual patterns reveal code structure, packed regions, and entropy distribution.
      </p>
    </div>
  </div>

</div>

<script>
  const fileInput   = document.getElementById('fileInput');
  const analyzeBtn  = document.getElementById('analyzeBtn');
  const selectedName = document.getElementById('selectedName');
  const spinnerWrap = document.getElementById('spinnerWrap');
  const errorBox    = document.getElementById('errorBox');
  const resultCard  = document.getElementById('resultCard');
  const resetBtn    = document.getElementById('resetBtn');

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      selectedName.textContent = '✔ ' + fileInput.files[0].name;
      analyzeBtn.disabled = false;
    }
  });

  async function runPrediction() {
    const file = fileInput.files[0];
    if (!file) return;

    // Reset previous results
    errorBox.style.display = 'none';
    resultCard.style.display = 'none';
    spinnerWrap.style.display = 'block';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const resp = await fetch('/predict', { method: 'POST', body: formData });
      const data = await resp.json();

      spinnerWrap.style.display = 'none';

      if (data.error) {
        showError(data.error);
        analyzeBtn.disabled = false;
        return;
      }

      renderResult(data);
    } catch (err) {
      spinnerWrap.style.display = 'none';
      showError('Connection error: ' + err.message);
      analyzeBtn.disabled = false;
    }
  }

  function renderResult(data) {
    const banner = document.getElementById('resultBanner');
    banner.textContent = data.label === 'malware' ? '🔴 MALWARE DETECTED' : '🟢 BENIGN';
    banner.className = 'result-banner ' + data.label;

    document.getElementById('resFileName').textContent = data.file_name;

    const pct = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('resConfidenceText').textContent = pct;

    const bar = document.getElementById('resConfidenceBar');
    bar.style.width = pct;
    bar.className = 'progress-bar-fill ' + data.label;

    document.getElementById('resLogit').textContent = data.logit;

    const img = document.getElementById('visImage');
    img.src = 'data:image/png;base64,' + data.image_b64;

    resultCard.style.display = 'block';
    resetBtn.style.display = 'inline-block';
    analyzeBtn.style.display = 'none';
  }

  function showError(msg) {
    errorBox.textContent = '⚠ ' + msg;
    errorBox.style.display = 'block';
  }

  function resetForm() {
    fileInput.value = '';
    selectedName.textContent = '';
    analyzeBtn.disabled = true;
    analyzeBtn.style.display = 'inline-block';
    resetBtn.style.display = 'none';
    resultCard.style.display = 'none';
    errorBox.style.display = 'none';
  }
</script>
</body>
</html>
```

### Test Task 3

Restart the Flask server and open a browser at `http://localhost:5000/`.
Verify:
- The page loads without errors
- Upload a PE file — the filename appears under the upload zone
- Click Analyze File — spinner shows, then result card appears with label, confidence bar, and byteplot image
- Upload a non-PE file — red error message appears, no crash
- Click "Analyze Another File" — form resets correctly

---

## TASK 4 — Create `Dockerfile`

Write `/home/alucard-00/EC499/Project_Resourse/Dockerfile`:

```dockerfile
# EC499 — Stage 4 Inference Demo Container
# Python 3.10 slim — CPU-only inference (no CUDA needed for demo)
FROM python:3.10-slim

# System libraries required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer cache efficiency
COPY requirements.txt .

# Install Python packages — CPU-only torch to keep image under 1 GB
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        flask>=2.0 \
        pefile \
        opencv-python-headless \
        Pillow \
        numpy && \
    pip install --no-cache-dir \
        torch==2.2.2+cpu \
        torchvision==0.17.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY config.py .
COPY models.py .
COPY convert_to_malimg.py .
COPY inference.py .
COPY app.py .
COPY templates/ ./templates/

# Copy trained model weights
COPY models/3c2d_malex_clean_vulnerable.pth /app/models/
COPY models/3c2d_malex_adversarially_trained.pth /app/models/

# Environment defaults
ENV MODEL_TYPE=clean
ENV PORT=5000

EXPOSE 5000

CMD ["python", "app.py"]
```

### Test Task 4

```bash
# Build the container (from Project_Resourse directory)
cd /home/alucard-00/EC499/Project_Resourse
docker build -t ec499-demo .

# If docker is not installed:
# sudo apt-get install -y docker.io
# sudo systemctl start docker
# sudo usermod -aG docker $USER
# (log out and back in for group change to take effect)

# Run with network isolation (--network=none) and port binding
docker run --network=none -p 5000:5000 ec499-demo &

# Wait 5 seconds for startup, then test
sleep 5
curl -s http://localhost:5000/health

# Test with a PE file (from host, which can reach the container via port binding)
curl -s -X POST \
  -F "file=@/home/alucard-00/EC499/benign_pe_files/benign_00001.exe" \
  http://localhost:5000/predict | python3 -m json.tool
```

---

## FINAL END-TO-END INTEGRATION TEST

Run all four tests after all tasks are complete. Report exact output for each.

```bash
# Start server (not Docker)
cd /home/alucard-00/EC499/Project_Resourse
/home/alucard-00/EC499/Project_Resourse/venv/bin/python app.py &
sleep 3

# Test 1 — Health check
echo "=== TEST 1: HEALTH ==="
curl -s http://localhost:5000/health

# Test 2 — Benign PE file
echo "=== TEST 2: BENIGN PE ==="
curl -s -X POST \
  -F "file=@/home/alucard-00/EC499/benign_pe_files/benign_00001.exe" \
  http://localhost:5000/predict \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Label:', d['label'])
print('Confidence:', d['confidence'])
print('Logit:', d['logit'])
print('image_b64 length:', len(d.get('image_b64','')))
print('PASS' if d.get('label') in ('benign','malware') and 0 < d.get('confidence',0) < 1 and len(d.get('image_b64','')) > 100 else 'FAIL')
"

# Test 3 — Non-PE file (expect 400 error)
echo "=== TEST 3: NON-PE FILE ==="
echo "this is not executable" > /tmp/notape.txt
curl -s -X POST \
  -F "file=@/tmp/notape.txt" \
  http://localhost:5000/predict

# Test 4 — Large fake file (expect 413 or upload rejection behavior)
echo "=== TEST 4: INVALID FILE ==="
# Just reconfirm health still works after errors
curl -s http://localhost:5000/health
```

---

## KNOWN POTENTIAL PROBLEMS AND SOLUTIONS

### Problem 1 — `get_nataraj_width` import fails
**Symptom:** `ImportError: cannot import name 'get_nataraj_width' from 'convert_to_malimg'`  
**Cause:** `convert_to_malimg.py` may export `pe_to_nataraj_image` but not expose `get_nataraj_width` at module level.  
**Fix:** Check the top of `convert_to_malimg.py`. If `get_nataraj_width` is defined but not exported, add it to `__all__` or simply ensure it is defined at module level (not nested). Alternatively, copy the width table logic directly into `inference.py` as a standalone function.

### Problem 2 — `pefile` raises unexpected exception on valid PE
**Symptom:** `ValueError: Not a valid PE file` on a known-good PE  
**Cause:** Some PE files have non-standard but valid headers that trip up certain pefile checks.  
**Fix:** Catch `pefile.PEFormatError` specifically (already done in the corrected code). If pefile raises for a file that Windows treats as valid, add a fallback that checks only the first 2 bytes (`b'MZ'`) manually:

```python
with open(file_path, 'rb') as f:
    magic = f.read(2)
if magic != b'MZ':
    raise ValueError("Not a valid PE file: missing MZ signature")
```

### Problem 3 — `cv2` import fails in venv
**Symptom:** `ModuleNotFoundError: No module named 'cv2'`  
**Fix:** Install the headless version specifically for this venv:

```bash
/home/alucard-00/EC499/Project_Resourse/venv/bin/pip install opencv-python-headless
```

Do NOT install the non-headless version — it requires libGL which may not be present.

### Problem 4 — Model load fails with unexpected keys
**Symptom:** `RuntimeError: Error(s) in loading state_dict: unexpected key(s) ...`  
**Cause:** Model weights saved from a different version of `MaleX3C2D`.  
**Fix:** Use `strict=False` in `load_state_dict`:

```python
state = torch.load(self.model_path, map_location="cpu")
self.model.load_state_dict(state, strict=False)
```

### Problem 5 — Flask server hangs during heavy GPU training
**Symptom:** HTTP requests time out or Flask startup takes very long  
**Cause:** System under heavy GPU/CPU load from PGD adversarial training  
**Fix:** Wait for training to complete before testing Flask. Alternatively, run Flask in a low-priority process:

```bash
nice -n 19 /home/alucard-00/EC499/Project_Resourse/venv/bin/python app.py
```

### Problem 6 — Docker build fails on torch CPU wheel
**Symptom:** `pip` cannot find `torch==2.2.2+cpu`  
**Fix:** Adjust the torch version to match what is currently available:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Use the latest stable CPU wheel, not a pinned version.

### Problem 7 — Port 5000 already in use
**Symptom:** `OSError: [Errno 98] Address already in use`  
**Fix:**

```bash
# Find and kill the process using port 5000
lsof -ti:5000 | xargs kill -9
# Or use a different port
PORT=8080 /home/alucard-00/EC499/Project_Resourse/venv/bin/python app.py
```

### Problem 8 — MaleX3C2D model class not found
**Symptom:** `ImportError: cannot import name 'MaleX3C2D' from 'models'`  
**Cause:** The `models.py` import fails because `sys.path` does not include `Project_Resourse/`.  
**Fix:** Ensure `inference.py` has `sys.path.insert(0, ...)` before any local imports. This is already in the authoritative version above.

---

## ALIGNMENT WITH PROJECT PROPOSAL

The project proposal (`project_proposal.md`) specifies:

> "The final trained model will be deployed in a separate, isolated environment with
> no network access. A secure inference API prototype will serve as the interface
> for this deployment, accepting PE file uploads, performing static analysis
> (PE validation and image conversion), and returning predictions."

**Stage 4 satisfies all requirements:**

| Proposal Requirement | Implementation |
|----------------------|----------------|
| Isolated environment with no network access | Docker with `--network=none` |
| Secure inference API | Flask `/predict` endpoint with size limits and temp file cleanup |
| PE file uploads | `multipart/form-data` file upload via POST |
| Static analysis — PE validation | pefile `e_magic` check, no execution |
| Image conversion | Nataraj width table → cv2.INTER_AREA resize |
| Returning predictions | JSON with label, confidence, logit, visualization |

**The demo flow for the thesis committee:**
1. Open browser at `http://localhost:5000`
2. Upload `benign_00001.exe` → shows BENIGN with green confidence bar and byteplot
3. Upload a malware sample → shows MALWARE DETECTED in red
4. Upload a random text file → shows validation error (demonstrates security)
5. (Optional) Run same file against adversarial model via `MODEL_TYPE=adversarial`
   to show the clean model vulnerability and AT model robustness comparison

---

## IMPLEMENTATION CHECKLIST

Use this to track progress. Update each item as work is completed.

- [ ] **Dependency check** — opencv-python-headless, pefile, flask all installed in venv
- [ ] **Task 1** — `inference.py` written with corrected PE validation (no bypass flag)
- [ ] **Task 1 Test 1** — Import and model load succeed
- [ ] **Task 1 Test 2** — Valid PE prediction returns correct JSON
- [ ] **Task 1 Test 3** — Non-PE raises `ValueError`, not unhandled exception
- [ ] **Task 2** — `app.py` written and running
- [ ] **Task 2 Test** — `/health`, `/predict` valid PE, `/predict` non-PE all pass
- [ ] **Task 3** — `templates/index.html` written
- [ ] **Task 3 Test** — End-to-end browser test works
- [ ] **Task 4** — `Dockerfile` written
- [ ] **Task 4 Test** — Container builds, `/health` responds from inside container
- [ ] **Final integration test** — All 4 curl tests pass with expected output
- [ ] **Demo preparation** — At least 2 test PE files ready (1 benign, 1 malware binary)

---

*Document version: 2026-04-02 | Author: Supervisor AI review | EC499 Stage 4*
