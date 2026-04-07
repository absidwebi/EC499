# EC499 Folder Structure Overview

Last updated: 2026-04-07 (aligned with current Stage 3/4 artifacts and script updates)

This document records the current workspace layout and identifies which files are part of the active research pipeline versus large data/artifact areas that should not be casually committed.

---

## 1. Top-Level Layout (High Level)

EC499/
- AGENTS.md
- EC499_Folder_Structure.md
- ben_byteplot_imgs_zipped/
- mal_byteplot_imgs_zipped/
- benign_images_256x256/
- benign_images_nataraj_v3/
- benign_images_test/
- Project_Resourse/
- run_logs/
- train_list.txt, val_list.txt
- train_dirs.txt, val_dirs.txt

Commit hygiene note:
- Large archives/zips/binaries and generated image datasets should stay out of normal source-control commits unless explicitly requested.

---

## 2. Active Project_Resourse Code Files

Core pipeline:
- config.py
- dataset_loader.py
- models.py
- train.py
- train_efficientnet.py
- adversarial_train.py
- adversarial_train_efficientnet.py

Evaluation:
- evaluate_base_models_testset.py
- evaluate_attacks.py
- evaluate_attacks_fixed.py
- generate_adversarial_test_set.py

Stage 4 service:
- inference.py
- app.py
- templates/index.html
- Dockerfile

---

## 3. Key Directories Under Project_Resourse

- archive/
  - malex_dataset/ (active split root)
- logs/
  - training/evaluation text logs and curves
- models/
  - clean and defended checkpoints
- adversarial_test_set_malex/
  - primary fixed adversarial PNG sets for Stage 3 deterministic comparison
- adversarial_test_set/
  - legacy smaller fixed adversarial PNG sets
- templates/
  - Stage 4 UI

---

## 4. Current Core Artifacts

Model artifacts:
- Project_Resourse/models/3c2d_malex_clean_vulnerable.pth
- Project_Resourse/models/3c2d_malex_adversarially_trained.pth
- Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_full_checkpoint.pth
- Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth

Evaluation artifacts:
- Project_Resourse/base_model_testset_results.json
- Project_Resourse/logs/attack_evaluation_results_3c2d.txt
- Project_Resourse/logs/attack_evaluation_results_3c2d_at.txt
- Project_Resourse/logs/fixed_adv_eval_all.txt
- Project_Resourse/logs/attack_comparison_3c2d_clean_vs_at_post35.txt

Run logs (examples):
- run_logs/adversarial_train_ 3C2D_Fixed_malex_stage3.log
- run_logs/fixed_adv_eval_all_models_final.log
- run_logs/evaluate_attacks_3c2d_post35.log
- run_logs/evaluate_attacks_3c2d_at_post35.log

---

## 5. Verified Dataset and Fixed-Set Counts

MaleX split counts:
- train total: 287,560
- val total: 33,015
- test total: 33,004
- test benign: 17,153
- test malware: 15,851

Fixed adversarial set counts:
- Project_Resourse/adversarial_test_set_malex/fgsm_eps0.05: 15,851 PNG
- Project_Resourse/adversarial_test_set_malex/pgd_eps0.05_steps40: 15,851 PNG
- Project_Resourse/adversarial_test_set/fgsm_eps0.05: 847 PNG
- Project_Resourse/adversarial_test_set/pgd_eps0.05_steps40: 847 PNG

---

## 6. Current Runtime State

Stage 3 state:
- PGD training cycle complete for current run.
- Best robust validation checkpoint remains epoch 35 at 74.123883% robust val accuracy.

Stage 4 state:
- /health and /compare endpoints verified.
- Browser access requires bridge networking with -p in this environment.

---

## 7. Remaining Structural/Process Tasks

1) Resolve cross-split overlap risk and rerun diagnostics.
2) Run one final audit pass with upgraded metrics scripts and archive outputs.
3) Package a final Stage 4 reproducibility/demo bundle for thesis appendix.
4) Keep commit scope tight due very large untracked binary data in workspace.
