# CURRENT_STATE.md
# EC499 - Current Project State

Last updated: 2026-04-02

---

## 1. Stage Progress Snapshot

| Stage | Status | Notes |
|---|---|---|
| Stage 1 - Dataset Preparation | DONE | MaleX split pipeline is active under archive/malex_dataset |
| Stage 2 - Base Model Selection | DONE | 3C2D selected from final test-set comparison |
| Stage 3 Part 1 - Attack Evaluation | DONE | Clean baseline vulnerability confirmed |
| Stage 3 Part 2 - Adversarial Training | IN PROGRESS (continuation run) | PGD resume run active; best robust val currently > 71% |
| Stage 3 Part 2b - FGSM Defense Branch | DONE | 20-epoch FGSM run completed with checkpoint + curve |
| Stage 4 - Inference API + Demo | IN PROGRESS | inference.py, app.py, UI, Dockerfile implemented; Docker runtime test blocked locally |

---

## 2. What Changed Since 2026-04-01

### 2.1 New/updated Python scripts

1) Updated: Project_Resourse/config.py
- Added constants for fixed adversarial test set and FGSM-trained model path.

2) New: Project_Resourse/generate_malex_adv_testset.py
- One-time generator + verifier for deterministic MaleX adversarial test subsets.

3) New: Project_Resourse/evaluate_attacks_fixed.py
- Evaluates clean/PGD/FGSM models on fixed subsets for reproducible comparison.

4) New: Project_Resourse/adversarial_train_fgsm.py
- FGSM adversarial training with full resume checkpoints and per-epoch logs/curves.

5) New: Project_Resourse/inference.py
- Static PE validation + Nataraj conversion + OpenCV resize + CPU inference pipeline.

6) New: Project_Resourse/app.py
- Flask API with /health, /predict, 50MB cap, tempfile handling, JSON error paths.

### 2.2 New major artifacts

- Project_Resourse/adversarial_test_set_malex/
- Project_Resourse/models/3c2d_malex_fgsm_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_fgsm_full_checkpoint.pth
- Project_Resourse/models/at_3c2d_full_checkpoint.pth
- Project_Resourse/logs/fixed_adv_eval_3c2d.txt
- Project_Resourse/logs/fixed_adv_eval_3c2d_pgd.txt
- Project_Resourse/logs/fixed_adv_eval_all.txt
- Project_Resourse/logs/adversarial_training_log_fgsm.txt
- Project_Resourse/logs/adversarial_training_curve_fgsm.png
- Project_Resourse/logs/adversarial_training_curve_3c2d.png

---

## 3. Current Quantitative State

### 3.1 PGD defense continuation (active)

From current full-checkpoint metadata:
- epoch_zero_based: 14
- resume_next_epoch_1based: 16
- best_robust_val_acc: 71.5675%
- best_epoch: 15
- epochs_no_improve: 0

Current state summary:
- PGD continuation run is still active in terminal and writing to canonical run log.
- Model and full checkpoint are updating under models/.

### 3.2 FGSM defense branch (complete)

Final FGSM epoch log snapshot:
- Epoch 20/20
- Train Loss: 0.4971
- Train Acc: 72.73%
- Val Clean: 77.37%
- Val Robust (FGSM): 71.68%
- Best robust val acc (checkpoint): 72.7306% at epoch 19

### 3.3 Fixed adversarial-set comparison (deterministic)

From run_logs/fixed_adv_eval_all_models_final.log:
- Clean model:
  - Clean acc 85.29%
  - FGSM recall 15.29%
  - PGD recall 0.53%
- PGD-defended:
  - Clean acc 73.07%
  - FGSM recall 58.56%
  - PGD recall 58.29%
- FGSM-defended:
  - Clean acc 77.33%
  - FGSM recall 67.97%
  - PGD recall 67.84%

Interpretation:
- Both defended models are substantially stronger than clean baseline under fixed adversarial evaluation.

---

## 4. Stage 4 Inference and Demo Status

Implemented and locally validated:
- Project_Resourse/inference.py
- Project_Resourse/app.py
- Project_Resourse/templates/index.html
- Project_Resourse/Dockerfile

Local endpoint tests passed:
- GET /health returned status/model json.
- POST /predict with valid PE returned label/confidence/logit/image_b64.
- POST /predict with non-PE returned ValueError-backed JSON error.

---

## 5. Current Problems / Blockers

1) Split-overlap risk remains open (scientific methodology risk).
2) PGD continuation run is still in progress, so final continuation metrics are not locked yet.
3) Docker runtime validation is blocked in this environment:
- docker command missing
- sudo install path requires interactive password
4) Large unrelated local untracked files remain in workspace and should stay out of scoped commits.

---

## 6. Immediate Next Actions

1) Let active PGD continuation run finish and record final best robust-val metric.
2) Re-run fixed-set all-model comparison if PGD continuation improves materially.
3) Resolve split-overlap issue and rerun overlap integrity diagnostics.
4) Complete Docker build/run test once docker is available.
