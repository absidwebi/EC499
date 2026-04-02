# EC499 Folder Structure Overview

Last updated: 2026-04-02

This file captures the current high-level workspace layout and the key files relevant to active Stage 3 continuation work and Stage 4 malware-inference deployment implementation.

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
|- benign_images_256x256/
|- benign_images_nataraj_v3/
|- benign_images_test/
|- benign_pe_files/
|- Project_Resourse/
|  |- adversarial_test_set/
|  |- adversarial_test_set_malex/
|  |- archive/
|  |- logs/
|  |- models/
|  |- templates/
|  |- venv/
|  `- active *.py scripts + Stage 4 app assets
|- run_logs/
|- venv/
|- train_list.txt / val_list.txt
`- train_dirs.txt / val_dirs.txt
```

Notes:
- Large binary datasets, archives, and generated PNG artifacts are intentionally summarized.
- Normal commit scope remains code, configs, and documentation; large data and model binaries are not routine commit targets.

---

## 2. Active Project_Resourse Python Files (Current Focus)

Core pipeline and training:
- `config.py`
- `dataset_loader.py`
- `models.py`
- `train.py`
- `train_3c2d.py`
- `train_resnet_pretrained.py`
- `adversarial_train.py` (PGD continuation)
- `adversarial_train_fgsm.py` (FGSM branch)

Evaluation and deterministic robustness comparison:
- `evaluate_attacks.py`
- `generate_malex_adv_testset.py`
- `evaluate_attacks_fixed.py`
- `evaluate_base_models_testset.py`

Stage 4 inference and service:
- `inference.py`
- `app.py`

Additional diagnostics/utility scripts remain in place and are unchanged unless noted in run logs or context docs.

---

## 3. Key Stage 3/4 Artifacts (Current)

### 3.1 Primary model outputs and checkpoints
- `Project_Resourse/models/3c2d_malex_adversarially_trained.pth`
- `Project_Resourse/models/3c2d_malex_clean_vulnerable.pth`
- `Project_Resourse/models/at_3c2d_full_checkpoint.pth`
- `Project_Resourse/models/at_3c2d_fgsm.pth`
- `Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth`

### 3.2 Evaluation outputs
- `Project_Resourse/base_model_testset_results.json`
- `Project_Resourse/logs/attack_comparison_3c2d_before_after_stage3.json`
- `Project_Resourse/logs/attack_evaluation_results_3c2d.txt`

### 3.3 Stage 4 deployment assets
- `Project_Resourse/Dockerfile`
- `Project_Resourse/templates/index.html`

---

## 4. Run Logs (Representative)

`run_logs/` currently includes:
- `adversarial_train_ 3C2D_Fixed_malex_stage3.log`
- `adversarial_train_fgsm_20260401_052733.log`
- `evaluate_attacks_3c2d_clean_vs_defended_stage3.log`
- `fixed_adv_eval_all_models_final.log`
- `fixed_adv_eval_3c2d_clean.log`
- `fixed_adv_eval_3c2d_pgd.log`
- `generate_adv_testset_malex.log`
- `train_3c2d_malex_fixed.log`

`Project_Resourse/logs/` includes:
- `adversarial_training_log_3c2d.txt`
- `adversarial_training_log_fgsm.txt`
- `attack_evaluation_results_3c2d.txt`
- `attack_comparison_3c2d_before_after_stage3.json`
- additional historical plots and legacy outputs

---

## 5. Current Runtime Status

At update time (2026-04-02), both of the following processes are active:
- Stage 3 PGD continuation training:
	- `/home/alucard-00/EC499/venv/bin/python /home/alucard-00/EC499/Project_Resourse/adversarial_train.py`
- Stage 4 Flask application:
	- `/home/alucard-00/EC499/Project_Resourse/venv/bin/python app.py`

Checkpoint metadata snapshot at update time:
- `Project_Resourse/models/at_3c2d_full_checkpoint.pth`:
	- saved epoch (zero-based): 14
	- best robust val acc: 71.5675%
	- best epoch (one-based): 15
- `Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth`:
	- saved epoch (zero-based): 19
	- best robust val acc: 72.7306%
	- best epoch (one-based): 19

Operational note:
- Local Stage 4 health and predict endpoint tests pass; Docker runtime validation remains pending due host docker availability.
