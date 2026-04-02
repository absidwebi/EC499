import os
import platform
from pathlib import Path

# Project root is the directory containing this config file
PROJECT_ROOT = Path(__file__).resolve().parent

# Define main dataset directories
BENIGN_PE_DIR = PROJECT_ROOT.parent / "benign_pe_files"
BENIGN_IMAGES_DIR = PROJECT_ROOT / "benign_images_256x256"
BENIGN_IMAGES_NATARAJ_DIR = PROJECT_ROOT / "benign_images_nataraj"
BENIGN_IMAGES_NATARAJ_V3_DIR = PROJECT_ROOT.parent / "benign_images_nataraj_v3"
MALIMG_ARCHIVE_DIR = PROJECT_ROOT / "archive" / "malimg_dataset"

# Model and Log directories for Stage 2
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Testing/Comparison specific
COMPARISON_PLOT_PATH = PROJECT_ROOT / "comparison_grid.png"
MALWARE_SAMPLE_DIR = MALIMG_ARCHIVE_DIR / "train" / "Adialer.C"

# Train/Val/Test directories
TRAIN_DIR = MALIMG_ARCHIVE_DIR / "train"
VAL_DIR = MALIMG_ARCHIVE_DIR / "val"
TEST_DIR = MALIMG_ARCHIVE_DIR / "test"

# OS-Specific Source Directories for Benign Collection
if platform.system() == "Windows":
    BENIGN_SOURCE_DIRS = [
        r"C:\Windows\System32",
        r"C:\Windows\SysWOW64",
        r"C:\Program Files",          
        r"C:\Program Files (x86)"     
    ]
else:
    # Linux fallback paths (Note: Linux binaries are ELF, not PE, but kept for compatibility)
    BENIGN_SOURCE_DIRS = [
        "/usr/bin",
        "/bin",
        "/usr/lib"
    ]

# String exports for easy compatibility with os.path
BENIGN_PE_DIR_STR = str(BENIGN_PE_DIR)
BENIGN_IMAGES_DIR_STR = str(BENIGN_IMAGES_DIR)
BENIGN_IMAGES_NATARAJ_DIR_STR = str(BENIGN_IMAGES_NATARAJ_DIR)
BENIGN_IMAGES_NATARAJ_V3_DIR_STR = str(BENIGN_IMAGES_NATARAJ_V3_DIR)
MALIMG_ARCHIVE_DIR_STR = str(MALIMG_ARCHIVE_DIR)
COMPARISON_PLOT_PATH_STR = str(COMPARISON_PLOT_PATH)
MALWARE_SAMPLE_DIR_STR = str(MALWARE_SAMPLE_DIR)
CUSTOM_CNN_CLEAN_MODEL_PATH_STR = str(MODEL_OUTPUT_DIR / "custom_cnn_clean_vulnerable.pth")
RESNET_ADV_TRAINED_MODEL_PATH_STR = str(MODEL_OUTPUT_DIR / "resnet18_adversarially_trained.pth")
RESNET_CLEAN_MODEL_PATH_STR = str(MODEL_OUTPUT_DIR / "resnet18_clean_vulnerable.pth")
EFFICIENTNET_CLEAN_MODEL_PATH_STR = str(MODEL_OUTPUT_DIR / "efficientnet_b0_clean_vulnerable.pth")
EFFICIENTNET_ADV_TRAINED_MODEL_PATH_STR = str(MODEL_OUTPUT_DIR / "efficientnet_b0_adversarially_trained.pth")

# ============================================================
# MaleX Dataset Constants (added for MaleX integration)
# ============================================================
MALEX_DATASET_DIR = PROJECT_ROOT / "archive" / "malex_dataset"
MALEX_DATASET_DIR_STR = str(MALEX_DATASET_DIR)

# MaleX-specific model save paths (separate from Malimg models)
RESNET_MALEX_CLEAN_MODEL_PATH_STR = str(MODEL_OUTPUT_DIR / "resnet18_malex_clean_vulnerable.pth")
RESNET_MALEX_ADV_MODEL_PATH_STR   = str(MODEL_OUTPUT_DIR / "resnet18_malex_adversarially_trained.pth")
MALEX_3C2D_CLEAN_MODEL_PATH_STR = str(MODEL_OUTPUT_DIR / "3c2d_malex_clean_vulnerable.pth")
MALEX_3C2D_ADV_MODEL_PATH_STR   = str(MODEL_OUTPUT_DIR / "3c2d_malex_adversarially_trained.pth")
RESNET_MALEX_PRETRAINED_CLEAN_PATH_STR = str(MODEL_OUTPUT_DIR / "resnet18_malex_pretrained_clean.pth")
RESNET_MALEX_PRETRAINED_ADV_PATH_STR   = str(MODEL_OUTPUT_DIR / "resnet18_malex_pretrained_adversarial.pth")

# ============================================================
# Stage 3 — Fixed Adversarial Test Set (MaleX)
# ============================================================
# Pre-generated adversarial examples from the MaleX test set.
# Generated once from the clean 3C2D model, reused for all evaluations.
MALEX_ADV_TEST_SET_DIR = PROJECT_ROOT / "adversarial_test_set_malex"
MALEX_ADV_TEST_SET_DIR_STR = str(MALEX_ADV_TEST_SET_DIR)

# FGSM adversarial training model paths (separate from PGD-trained model)
MALEX_3C2D_FGSM_ADV_MODEL_PATH_STR = str(
    MODEL_OUTPUT_DIR / "3c2d_malex_fgsm_adversarially_trained.pth"
)
