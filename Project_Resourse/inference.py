"""
inference.py
============
Stage 4 - Single entry point for the PE file -> MaleX3C2D prediction pipeline.

Pipeline:
    PE file -> pefile validation (static only) -> raw bytes -> Nataraj array
    -> cv2.INTER_AREA resize to 256x256 -> normalized tensor -> 3C2D model -> prediction

Scientific notes:
    - pefile performs static structural analysis only. No code is executed.
    - Windows PE files work correctly on Linux with pefile.
    - The resize method (cv2.INTER_AREA) matches how MaleX dataset images were
      prepared, ensuring inference distribution matches training distribution.
    - Normalization (mean=0.5, std=0.5) matches dataset_loader.py exactly.
    - Model loaded with map_location='cpu' - no GPU required for inference.
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
import torch.nn as nn

# Ensure project modules import correctly regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MALEX_3C2D_CLEAN_MODEL_PATH_STR,
    MALEX_3C2D_ADV_MODEL_PATH_STR,
)
from convert_to_malimg import get_nataraj_width
from models import MaleX3C2D


# ---------------------------------------------------------------------------
# Component 1 - PE Validation and Raw Byte Reading
# ---------------------------------------------------------------------------

def validate_and_read_bytes(file_path: str) -> bytes:
    """
    Validate that file_path is a PE executable and return its raw bytes.

    Uses pefile for structural validation only - no execution occurs.
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
# Component 2 - Nataraj Byte-to-Image Conversion
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
# Component 3 - Resize to 256x256 (MaleX-faithful method)
# ---------------------------------------------------------------------------

def resize_to_256(nataraj_array: np.ndarray) -> np.ndarray:
    """
    Resize variable-size Nataraj array to 256x256 using cv2.INTER_AREA.

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
# Component 4 - Tensor Conversion (matching dataset_loader.py normalization)
# ---------------------------------------------------------------------------

def array_to_tensor(resized_array: np.ndarray) -> torch.Tensor:
    """
    Convert 256x256 uint8 array to normalized FloatTensor [1, 1, 256, 256].

    Normalization: (pixel/255 - 0.5) / 0.5 -> range [-1, 1]
    This exactly matches the Normalize(mean=[0.5], std=[0.5]) transform
    in dataset_loader.py used during training.
    """
    x = resized_array.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.expand_dims(x, axis=0)   # [256, 256] -> [1, 256, 256]
    x = np.expand_dims(x, axis=0)   # [1, 256, 256] -> [1, 1, 256, 256]
    return torch.from_numpy(x).float()


# ---------------------------------------------------------------------------
# Component 5 - Base64 PNG Encoding for display
# ---------------------------------------------------------------------------

def array_to_png_base64(array: np.ndarray) -> str:
    """
    Convert a grayscale numpy array to a base64-encoded PNG string.

    The caller should pass the variable-size Nataraj array (before resize)
    for richer visual display - the natural proportions reflect actual file
    size and are more visually meaningful for the demo.

    The base64 string can be embedded directly in an HTML <img> tag:
        <img src="data:image/png;base64,{image_b64}">
    """
    img = Image.fromarray(array.astype(np.uint8), mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Inference Engine - loads model once, serves predictions
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
        Run the full PE file -> prediction pipeline.

        Args:
            file_path: absolute path to the PE file to analyze.

        Returns:
            dict with keys:
                label       : "malware" or "benign"
                confidence  : float [0, 1] - sigmoid of logit
                logit       : raw model output (positive = malware)
                image_b64   : base64 PNG of the Nataraj byteplot visualization
                file_name   : basename of the analyzed file
                model_type  : "clean" or "adversarial"

        Raises:
            ValueError   : file is not a valid PE executable
            RuntimeError : any other error in the pipeline
        """
        try:
            raw_bytes = validate_and_read_bytes(file_path)
            nataraj_array = pe_bytes_to_nataraj_array(raw_bytes)
            image_b64 = array_to_png_base64(nataraj_array)   # variable-size for display
            resized_array = resize_to_256(nataraj_array)
            tensor = array_to_tensor(resized_array)

            with torch.no_grad():
                logit = self.model(tensor).item()

            confidence = float(torch.sigmoid(torch.tensor(logit)).item())
            label = "malware" if logit > 0 else "benign"

            return {
                "label": label,
                "confidence": round(confidence, 4),
                "logit": round(logit, 4),
                "image_b64": image_b64,
                "file_name": os.path.basename(file_path),
                "model_type": self._model_type,
            }

        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Inference pipeline error: {exc}") from exc


class AdversarialComparisonEngine:
    """
    Loads both the clean and adversarially trained (AT) MaleX3C2D models and
    provides side-by-side comparison of their behavior on clean vs PGD-perturbed inputs.

    This is the core demonstration of the project's research contribution:
    - Clean model: high accuracy on clean inputs, easily fooled by PGD attack
    - AT model: maintains accuracy on both clean and adversarial inputs

    Attack threat model:
        Attacker has white-box access to the CLEAN model and applies PGD to
        generate adversarial examples. The AT model was trained to resist this.
        This is the exact evaluation protocol used in evaluate_attacks_fixed.py.
    """

    PGD_EPS = 0.05
    PGD_ALPHA = 0.01
    PGD_STEPS = 20   # 20 steps for demo speed; 40 used in paper evaluation

    def __init__(self):
        # Load clean (vulnerable) model
        self.clean_model = MaleX3C2D()
        clean_state = torch.load(str(MALEX_3C2D_CLEAN_MODEL_PATH_STR), map_location="cpu")
        self.clean_model.load_state_dict(clean_state)
        self.clean_model.eval()

        # Load adversarially trained model
        self.at_model = MaleX3C2D()
        at_state = torch.load(str(MALEX_3C2D_ADV_MODEL_PATH_STR), map_location="cpu")
        self.at_model.load_state_dict(at_state)
        self.at_model.eval()

        print(f"[inference] AdversarialComparisonEngine: both models loaded")
        print(f"[inference]   Clean model : {MALEX_3C2D_CLEAN_MODEL_PATH_STR}")
        print(f"[inference]   AT model    : {MALEX_3C2D_ADV_MODEL_PATH_STR}")

    def _pgd_attack(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate PGD adversarial example using the CLEAN model as the white-box oracle.

        The attack maximises the loss with respect to label=1 (malware), which causes
        the clean model to predict benign. This is the evasion threat model.

        Gradient computation uses the clean model in eval mode. BCEWithLogitsLoss
        is used to match the training loss function exactly.
        """
        criterion = nn.BCEWithLogitsLoss()
        # True label is malware (1). Attack maximises this loss -> clean model predicts benign.
        labels_f = torch.ones(1, 1)   # shape [1,1] to match model output shape

        adv = tensor.clone().detach()
        # Random start within epsilon ball (standard PGD initialization)
        adv = adv + torch.empty_like(adv).uniform_(-self.PGD_EPS, self.PGD_EPS)
        adv = torch.clamp(adv, -1.0, 1.0).detach()

        for _ in range(self.PGD_STEPS):
            adv.requires_grad_(True)
            outputs = self.clean_model(adv)
            loss = criterion(outputs, labels_f)
            self.clean_model.zero_grad()
            loss.backward()
            with torch.no_grad():
                adv = adv + self.PGD_ALPHA * adv.grad.sign()
                delta = torch.clamp(adv - tensor, -self.PGD_EPS, self.PGD_EPS)
                adv = torch.clamp(tensor + delta, -1.0, 1.0).detach()

        return adv

    def _tensor_to_display_b64(self, tensor: torch.Tensor) -> str:
        """
        Convert a normalized [-1, 1] tensor of shape [1,1,256,256] to a
        base64-encoded PNG suitable for display in the browser.

        Reverses the normalization: pixel = (tensor + 1) / 2 * 255
        """
        arr = tensor.squeeze().detach().numpy()  # [256, 256]
        arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _predict_single(self, model: torch.nn.Module, tensor: torch.Tensor) -> dict:
        """Run one forward pass and return prediction dict."""
        with torch.no_grad():
            logit = model(tensor).item()
        confidence = float(torch.sigmoid(torch.tensor(logit)).item())
        return {
            "label": "malware" if logit > 0 else "benign",
            "confidence": round(confidence, 4),
            "logit": round(logit, 4)
        }

    def compare(self, file_path: str) -> dict:
        """
        Full adversarial comparison pipeline.

        Steps:
            1. Validate PE and read raw bytes (static analysis only)
            2. Apply Nataraj width table to get variable-size byteplot
            3. Resize to 256x256 with cv2.INTER_AREA (matches training pipeline)
            4. Convert to normalized tensor ([-1, 1])
            5. Apply PGD attack on clean model to produce adversarial tensor
            6. Convert both tensors to display images (base64 PNG)
            7. Run clean model on clean tensor
            8. Run clean model on adversarial tensor
            9. Run AT model on clean tensor
            10. Run AT model on adversarial tensor

        Returns:
            dict with all 4 predictions plus 3 images for display:
                - byteplot_b64      : variable-size Nataraj image (raw byte structure)
                - clean_256_b64     : 256x256 image as seen by models (clean)
                - adv_256_b64       : 256x256 image after PGD perturbation
                - clean_model       : predictions from the vulnerable clean model
                - at_model          : predictions from the adversarially trained model
                - attack_params     : PGD parameters used (for display in UI)

        Raises:
            ValueError   : file is not a valid PE executable
            RuntimeError : any other pipeline error
        """
        try:
            raw_bytes = validate_and_read_bytes(file_path)
            nataraj_array = pe_bytes_to_nataraj_array(raw_bytes)
            byteplot_b64 = array_to_png_base64(nataraj_array)   # original variable-size
            resized_array = resize_to_256(nataraj_array)
            clean_tensor = array_to_tensor(resized_array)        # normalized [-1,1]

            # Generate adversarial example targeting the clean model
            adv_tensor = self._pgd_attack(clean_tensor)

            # Convert both tensors to display images
            clean_256_b64 = self._tensor_to_display_b64(clean_tensor)
            adv_256_b64 = self._tensor_to_display_b64(adv_tensor)

            # Run all 4 predictions
            return {
                "file_name": os.path.basename(file_path),
                "byteplot_b64": byteplot_b64,
                "clean_256_b64": clean_256_b64,
                "adv_256_b64": adv_256_b64,
                "attack_params": {
                    "type": "PGD",
                    "eps": self.PGD_EPS,
                    "alpha": self.PGD_ALPHA,
                    "steps": self.PGD_STEPS
                },
                "clean_model": {
                    "clean_input": self._predict_single(self.clean_model, clean_tensor),
                    "adv_input": self._predict_single(self.clean_model, adv_tensor)
                },
                "at_model": {
                    "clean_input": self._predict_single(self.at_model, clean_tensor),
                    "adv_input": self._predict_single(self.at_model, adv_tensor)
                }
            }

        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Comparison pipeline error: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py /path/to/file.exe [--adversarial]")
        sys.exit(1)

    test_path = sys.argv[1]
    use_adv = "--adversarial" in sys.argv

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
