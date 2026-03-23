# EC499 Folder Structure Overview

This document provides a comprehensive view of the `EC499` directory structure, highlighting key project files, datasets, and configurations. Note: Large image datasets and virtual environment internals are summarized for readability.

## 📁 Root Directory: `EC499/`

- **📄 2026904.2016908.pdf**: Reference PDF (669 KB)
- **📄 2402.15267v1.pdf**: Heavy research paper/PDF (3.8 MB)
- **📄 Adversarial_Robustness_Presentaion.pdf**: Project presentation (5.6 MB)
- **📄 Adversarial_Robustness_in_Deep_Learning_Based_Malware_DetectionProjectProposal.pdf**: Project proposal (50 KB)
- **📄 archive.zip**: Original Malimg dataset compressed (1.17 GB)
- **📄 train_dirs.txt**: List of training subdirectories
- **📄 val_dirs.txt**: List of validation subdirectories
- **📂 benign_images_256x256/**: Staging area for converted benign images
- **📂 benign_images_test/**: Test batch of benign images
- **📂 benign_pe_files/**: Collection of raw benign Windows PE (.exe) files
- **📂 venv/**: Python Virtual Environment (Standard library and dependencies)
- **📂 Project_Resourse/**: Main project development folder (Detailed below)

---

## 📂 main Project Development: `Project_Resourse/`

This folder contains the core implementation of the Malware Detection pipeline.

### 🐍 Python Scripts (Core Logic)
- **📄 adversarial_train.py**: Stage 3 script for hardening models against PGD attacks.
- **📄 collect_benign_pe.py**: Script to gather benign PE files from the OS.
- **📄 config.py**: Centralized configuration for all paths and global variables.
- **📄 convert_to_malimg.py**: Converts PE files into Malimg-standard grayscale images.
- **📄 dataset_loader.py**: PyTorch Dataset/DataLoader implementation with binary mapping.
- **📄 evaluate_attacks.py**: Script to test model vulnerability against FGSM/PGD.
- **📄 models.py**: Architecture definitions (Custom CNN & ResNet-18 Grayscale).
- **📄 split_benign_dataset.py**: Distributes converted benign images into Malimg folders.
- **📄 train.py**: Standard Stage 2 training script for clean baselines.
- **📄 verify_benign_collection.py**: Integrity checker for PE files.
- **📄 check_sizes.py**: Utility to analyze file dimension distribution.

### 📝 Reports & Documentation
- **📄 CNN Architecture Design.md**: High-level design notes.
- **📄 CNN_Architecture_for_Robustness.md**: Research on robust CNN designs.
- **📄 Malware_Detection_CNN_Architecture_Recommendations.md**: Specific recommendations from literature.
- **📄 stage2_architecture_report.md**: Summary of Stage 2 findings and decisions.
- **📄 implementation_plan.md.resolved**: Finalized plan for Stage 1/2.
- **📄 presentation.md**: Source for the project presentation.
- **📄 project_proposal.md**: Full project proposal in markdown format.

### 📦 Trained Models (`models/`)
- **📄 custom_cnn_clean_vulnerable.pth**: Initial experimental CNN weights.
- **📄 resnet18_clean_vulnerable.pth**: Converged ResNet-18 clean baseline.

### 📊 Dataset Storage (`archive/malimg_dataset/`)
- **📂 train/**: Training split containing 25 Malware family folders + `benign/`.
- **📂 val/**: Validation split (Balanced across families).
- **📂 test/**: Final hold-out test set for evaluation.

### 📈 Logs & Visuals (`logs/`)
- **📄 attack_evaluation_results.txt**: Metrics from PGD/FGSM evaluations.
- **📄 stage2_forensic_plot.png**: Visualization of model predictions vs actual pixels.
- **📄 comparison_grid.png**: Side-by-side comparison of malware families.

---

*Note: The `__pycache__` folders and individual image files inside family subdirectories are excluded from this overview for clarity (~24,000+ total image files).*
