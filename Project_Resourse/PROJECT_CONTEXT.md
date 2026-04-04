# EC499 - Adversarial Robustness in Deep Learning-Based Malware Detection

Student: Abdulsalam Ashraf Aldwebi (ID: 2210245306)
Supervisor: Dr. Suad Elgeder | University of Tripoli
GitHub: https://github.com/absidwebi/EC499
Primary machine: Ubuntu Linux, RTX 4060, /home/alucard-00/EC499
Python environment used in latest runs: /home/alucard-00/EC499/Project_Resourse/venv

Last updated: 2026-04-04 (post Stage 3 finalization + Stage 4 adversarial comparison deployment)

---

## 1. Project Objective

Build a reproducible malware image classification pipeline and evaluate adversarial vulnerability and defense under controlled methodology, then provide a static-analysis inference demo.

Binary task:
- Benign = 0
- Malware = 1

Core methodological constraints currently enforced in active scripts:
- Single-logit output with BCEWithLogitsLoss
- Adversarial clamp range is [-1.0, 1.0]
- No geometric augmentation in byteplot pipeline
- num_workers = 0 for Linux/CUDA stability
- Paths imported from config.py (no hardcoded project paths in active pipeline code)

---

## 2. Active Data Track

Current active branch for Stage 2/3/4 is MaleX byteplot data.

Source roots:
- Benign: /home/alucard-00/EC499/ben_byteplot_imgs_zipped/byteplot_imgs_RxR/256/
- Malware: /home/alucard-00/EC499/mal_byteplot_imgs_zipped/byteplot_imgs_RxR/256/

Working split root:
- /home/alucard-00/EC499/Project_Resourse/archive/malex_dataset/

MaleX split sizes from loader output:
- Train: 287,560 (143,780 benign + 143,780 malware)
- Val: 33,015
- Test: 33,004

Open methodological note:
- Cross-split overlap remediation is still tracked as an unresolved scientific risk and must be addressed before final thesis-grade claims.

---

## 3. Stage 2 Final Selection

Final clean baseline candidates:
- MaleX3C2D
- ResNet-18 pretrained grayscale variant

Authoritative artifact:
- Project_Resourse/base_model_testset_results.json

Selected Stage 3 baseline:
- MaleX3C2D

Reason:
- Slight edge on accuracy and malware F1 while macro quality stayed close.

---

## 4. Stage 3 Defense Track (Final State)

### 4.1 Stage 3 part 1 (clean vulnerability) - complete

Script:
- Project_Resourse/evaluate_attacks.py

Artifact:
- Project_Resourse/logs/attack_evaluation_results_3c2d.txt

Outcome:
- Clean baseline remains highly vulnerable to stronger FGSM/PGD attacks.

### 4.2 Stage 3 part 2 (PGD adversarial training) - complete

Primary run log:
- run_logs/adversarial_train_ 3C2D_Fixed_malex_stage3.log

Canonical per-epoch artifact:
- Project_Resourse/logs/adversarial_training_log_3c2d.txt

Final training outcome:
- Training finished via early stopping at epoch 40/50 (patience reached, no robust-val improvement for 5 consecutive epochs)
- Best robust validation accuracy: 74.12% at epoch 35

Final completion block from run log:
- Epoch 40/50 | Train Loss 0.4675 | Train Acc 75.14% | Val Clean 79.72% | Val Robust 73.87%
- Early stopping at epoch 40. Best robust val acc: 74.12% (epoch 35)

Produced artifacts:
- Project_Resourse/models/3c2d_malex_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_full_checkpoint.pth
- Project_Resourse/logs/adversarial_training_curve_3c2d.png
- Project_Resourse/logs/adversarial_training_log_3c2d.txt

Interpretation:
- The continuation beyond epoch 35 did not improve robust validation beyond 74.12%, so the epoch-35 checkpoint remains the best defended model.

### 4.3 Fixed adversarial test set and deterministic comparison - complete

Generator and output:
- Project_Resourse/generate_malex_adv_testset.py
- Project_Resourse/adversarial_test_set_malex/
  - fgsm_eps0.05_steps1/images: 15,851
  - pgd_eps0.05_steps40/images: 15,851

Evaluator:
- Project_Resourse/evaluate_attacks_fixed.py

Artifacts:
- run_logs/fixed_adv_eval_all_models_final.log
- Project_Resourse/logs/fixed_adv_eval_all.txt

Outcome summary:
- Deterministic fixed-set comparisons confirm major robustness gains for defended models over clean baseline.

### 4.4 FGSM adversarial training branch - complete

Script:
- Project_Resourse/adversarial_train_fgsm.py

Artifacts:
- Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth
- Project_Resourse/logs/adversarial_training_log_fgsm.txt
- Project_Resourse/logs/adversarial_training_curve_fgsm.png

Final checkpoint summary:
- epoch_zero_based: 19
- best_robust_val_acc: 72.7306%
- best_epoch: 19

### 4.5 Clean vs AT mirrored attack comparison (post-epoch-35 reference) - complete

Artifacts:
- run_logs/evaluate_attacks_3c2d_post35.log
- run_logs/evaluate_attacks_3c2d_at_post35.log
- Project_Resourse/logs/attack_evaluation_results_3c2d.txt
- Project_Resourse/logs/attack_evaluation_results_3c2d_at.txt
- Project_Resourse/logs/attack_comparison_3c2d_clean_vs_at_post35.txt

Key side-by-side results:
- Clean (no attack): clean model 85.29% vs AT model 80.03% (delta -5.26 pp)
- FGSM e=0.10: clean model 3.49% vs AT model 71.39% (delta +67.90 pp)
- PGD e=0.05 steps=40: clean model 0.62% vs AT model 74.22% (delta +73.60 pp)

Interpretation:
- Robustness gains are large and consistent across attack strengths.
- Clean-accuracy tradeoff remains expected.

---

## 5. Stage 4 Inference and Deployment (Current State)

### 5.1 Implemented code paths

Updated/active files:
- Project_Resourse/inference.py
  - Static PE validation (pefile), byte extraction, Nataraj conversion, resize, normalization, single-model prediction
  - New AdversarialComparisonEngine that loads both clean and AT models
  - PGD attack generation for demonstration and side-by-side clean-vs-adversarial inference outputs
- Project_Resourse/app.py
  - Existing endpoints: /, /health, /predict
  - New /compare endpoint with temp-file handling and structured JSON errors
- Project_Resourse/templates/index.html
  - Replaced with two-tab UI:
    - PE analysis tab (/predict)
    - Adversarial robustness demo tab (/compare)
  - Displays byteplot, clean and adversarial 256x256 inputs, and both model predictions
- Project_Resourse/Dockerfile
  - Containerized CPU inference deployment

### 5.2 Runtime verification status

Local and container checks completed:
- GET /health on port 5000 returns model status JSON
- POST /compare with invalid non-PE file returns 400 with validation error
- POST /compare with real PE file returns 200 and full payload
  - keys: byteplot_b64, clean_256_b64, adv_256_b64, clean_model, at_model, attack_params

Container state:
- Docker permission issue was resolved by refreshing group context with newgrp docker
- ec499-demo image rebuilt from latest code
- ec499-inference container restarted on port 5000
- /compare endpoint is now active in the running container (previous 404 resolved)

### 5.3 Stage 4 interpretation and caveats

- Current endpoint behavior confirms the Stage 4 adversarial comparison feature is deployed and functioning.
- A benign PE sample was used in latest smoke verification, so both clean and AT model predictions remained benign; this is expected.
- For committee-facing demonstration of clean-model failure vs AT-model robustness, run /compare with a malware PE sample that the clean model classifies as malware before attack.

---

## 6. Code Changes Since Last Context Sync

### 6.1 Stage 3 training continuation config

1) Project_Resourse/adversarial_train.py
- NUM_EPOCHS changed from 20 to 50 for continuation.

Why:
- Continue training from checkpoint without restarting optimizer/scheduler state.

### 6.2 Stage 4 adversarial comparison feature update

2) Project_Resourse/inference.py
- Added AdversarialComparisonEngine
- Added PGD attack path and image/tensor conversion helpers for demo output
- Added compare() pipeline returning 4 prediction slots (clean model clean/adv, AT model clean/adv)

Why:
- Implement Stage 4 adversarial robustness comparison endpoint logic.

3) Project_Resourse/app.py
- Added startup loading for comparison engine
- Added POST /compare endpoint
- Added robust error responses for unavailable engine and invalid files

Why:
- Expose comparison engine through Flask API.

4) Project_Resourse/templates/index.html
- Replaced single-mode UI with two-tab UI and dedicated adversarial comparison workflow

Why:
- Provide complete browser-side Stage 4 demo for both standard inference and robustness comparison.

---

## 7. Known Open Items and Risks

1) Cross-split overlap risk remains unresolved and should be treated as an active scientific risk.
2) Post-final-training mirrored attack evaluation has not yet been rerun specifically against epoch-40 end state; best model remains epoch 35 so current post35 comparison is still representative, but one final confirmation pass is still recommended for audit completeness.
3) Stage 4 committee demo still needs a pinned malware sample list and one recorded run bundle (commands plus outputs/screenshots) for thesis appendix reproducibility.
4) Workspace still contains many large untracked datasets and generated artifacts that should remain outside normal git commit scope.

---

## 8. Immediate Next Actions

1) Run one final attack-evaluation confirmation pass against the locked best checkpoint and store results with explicit date tag.
2) Resolve split-overlap issue and rerun overlap diagnostics before final thesis claims.
3) Create a small fixed Stage 4 demo sample set (valid malware PE, valid benign PE, invalid file) and archive expected API outputs.
4) Prepare final thesis-ready robustness summary table using clean baseline, PGD-AT best checkpoint, and FGSM-AT checkpoint.