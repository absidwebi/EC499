# EC499 Folder Structure Overview

Last updated: 2026-04-04 (post Stage 3 completion and Stage 4 compare deployment)

This file captures the current high-level workspace layout and the key files relevant to the completed adversarial-training cycle and deployed Stage 4 inference demo.

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
- Typical commit scope should remain code/config/docs unless explicitly requested otherwise.

---

## 2. Active Project_Resourse Python Files (Current Focus)

Core pipeline and training:
- config.py
- dataset_loader.py
- models.py
- train.py
- train_3c2d.py
- train_resnet_pretrained.py
- adversarial_train.py (PGD branch, now completed for current cycle)
- adversarial_train_fgsm.py (FGSM branch, completed)

Evaluation and deterministic robustness comparison:
- evaluate_attacks.py
- generate_malex_adv_testset.py
- evaluate_attacks_fixed.py
- evaluate_base_models_testset.py

Stage 4 inference and service:
- inference.py (includes MalwareInferenceEngine and AdversarialComparisonEngine)
- app.py (/health, /predict, /compare)

---

## 3. Key Stage 3 and Stage 4 Artifacts (Current)

### 3.1 Primary model outputs and checkpoints
- Project_Resourse/models/3c2d_malex_clean_vulnerable.pth
- Project_Resourse/models/3c2d_malex_adversarially_trained.pth
- Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_full_checkpoint.pth
- Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth

### 3.2 Stage 3 evaluation outputs
- Project_Resourse/base_model_testset_results.json
- Project_Resourse/logs/attack_evaluation_results_3c2d.txt
- Project_Resourse/logs/attack_evaluation_results_3c2d_at.txt
- Project_Resourse/logs/attack_comparison_3c2d_before_after_stage3.json
- Project_Resourse/logs/attack_comparison_3c2d_clean_vs_at_post35.txt
- Project_Resourse/logs/fixed_adv_eval_all.txt

### 3.3 Stage 4 deployment assets
- Project_Resourse/Dockerfile
- Project_Resourse/app.py
- Project_Resourse/inference.py
- Project_Resourse/templates/index.html

---

## 4. Run Logs (Representative)

run_logs/ includes:
- adversarial_train_ 3C2D_Fixed_malex_stage3.log
- adversarial_train_fgsm_20260401_052733.log
- evaluate_attacks_3c2d_post35.log
- evaluate_attacks_3c2d_at_post35.log
- evaluate_attacks_3c2d_clean_vs_defended_stage3.log
- fixed_adv_eval_all_models_final.log
- generate_adv_testset_malex.log
- train_3c2d_malex_fixed.log

Project_Resourse/logs/ includes:
- adversarial_training_log_3c2d.txt
- adversarial_training_log_fgsm.txt
- adversarial_training_curve_3c2d.png
- adversarial_training_curve_fgsm.png
- attack_evaluation_results_3c2d.txt
- attack_evaluation_results_3c2d_at.txt
- fixed_adv_eval_all.txt

---

## 5. Current Runtime Status

Stage 3 PGD training run status:
- Completed for current cycle.
- Early stopped at epoch 40/50 with best robust val 74.12% at epoch 35.

Stage 4 deployment status:
- Docker container ec499-inference is up on port 5000.
- /health endpoint returns model status JSON.
- /compare endpoint now active in container after rebuild and supports both valid-PE and invalid-file paths.

Latest smoke checks:
- Invalid upload (/etc/hosts) -> /compare returns 400 with PE validation error.
- Valid PE upload (Project_Resourse/benign_pe_files_test/benign_00091.exe) -> /compare returns 200 with comparison payload.

---

## 6. Remaining Tasks To Track

1) Resolve remaining cross-split overlap risk and rerun diagnostics.
2) Run one final mirrored attack-evaluation confirmation pass for locked best checkpoint reporting.
3) Assemble final Stage 4 reproducibility/demo bundle for thesis appendix (sample files, commands, outputs, screenshots).