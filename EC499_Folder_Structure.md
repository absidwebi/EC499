# EC499 Folder Structure Overview

Last updated: 2026-04-01

This file captures the current high-level workspace layout and the key files relevant to the latest Stage 2 finalization and Stage 3 baseline defense/comparison cycle.

---

## 1. Top-Level Layout (High Level)

```text
EC499/
|- AGENTS.md
|- EC499_Folder_Structure.md
|- ben_byteplot_imgs_zipped/
|  `- byteplot_imgs_RxR/256/
|- mal_byteplot_imgs_zipped/
|  `- byteplot_imgs_RxR/256/
|- benign_images_nataraj_v3/
|- Project_Resourse/
|  |- archive/
|  |  |- malex_dataset/
|  |  |  |- train/
|  |  |  |- val/
|  |  |  `- test/
|  |  `- malimg_dataset/
|  |- logs/
|  |- models/
|  |- venv/
|  `- *.py scripts
`- run_logs/
```

Notes:
- Large binary datasets and generated PNG artifacts are intentionally summarized and not expanded in this document.
- Local workspace contains additional large untracked binaries/archives; these are not part of normal code-commit scope.

---

## 2. Active Project_Resourse Python Files (Current Focus)

Primary active scripts for this cycle:
- `config.py`
- `dataset_loader.py`
- `models.py`
- `train.py`
- `train_3c2d.py`
- `train_resnet_pretrained.py`
- `evaluate_attacks.py`
- `adversarial_train.py`
- `evaluate_base_models_testset.py` (new in this update cycle)

Other supporting scripts remain present (diagnostics, split tools, legacy experiments).

---

## 3. Key Artifact Files Added/Updated in This Cycle

### 3.1 New results artifact
- `Project_Resourse/base_model_testset_results.json`

### 3.2 New clean-vs-defended comparison artifact
- `Project_Resourse/logs/attack_comparison_3c2d_before_after_stage3.json`

### 3.3 Updated Stage 3 attack output
- `Project_Resourse/logs/attack_evaluation_results_3c2d.txt`

### 3.4 Relevant model checkpoints currently present
- `Project_Resourse/models/3c2d_malex_adversarially_trained.pth`
- `Project_Resourse/models/3c2d_malex_clean_vulnerable.pth`
- `Project_Resourse/models/3c2d_malex_full_checkpoint.pth`
- `Project_Resourse/models/resnet18_malex_pretrained_clean.pth`
- `Project_Resourse/models/resnet18_malex_pretrained_full_checkpoint.pth`
- `Project_Resourse/models/resnet18_malex_clean_vulnerable.pth`
- `Project_Resourse/models/resnet18_malex_adversarially_trained.pth`

---

## 4. Run Logs (Representative)

`run_logs/` currently includes:
- `adversarial_train_ 3C2D_Fixed_malex_stage3.log`
- `train_3c2d_malex_fixed.log`
- `train_resnet_pretrained_malex.log`
- `train_resnet18_malex_stage2.log`
- `evaluate_attacks_malex_stage3.log`
- `adversarial_train_malex_stage3.log`
- `evaluate_attacks_3c2d_clean_vs_defended_stage3.log`

`Project_Resourse/logs/` includes:
- `attack_evaluation_results_3c2d.txt`
- `attack_comparison_3c2d_before_after_stage3.json`
- `adversarial_training_log_3c2d.txt`
- `attack_evaluation_results.txt` (legacy/general)
- `adversarial_training_log.txt` (legacy/general)
- additional plots and historical outputs

---

## 5. Current Runtime Status

At update time (2026-04-01), no active Stage 3 training process was detected.

This indicates the baseline Stage 3 adversarial training run is complete, with follow-up extension runs pending user start.
