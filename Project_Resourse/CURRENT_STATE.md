# EC499 - Current Project State

Last updated: 2026-04-04 (after PGD AT completion and Stage 4 compare deployment)

---

## 1. Stage Progress Snapshot

| Stage | Status | Notes |
|---|---|---|
| Stage 1 - Dataset Preparation | DONE | MaleX split pipeline active under archive/malex_dataset |
| Stage 2 - Base Model Selection | DONE | 3C2D selected from final test-set comparison |
| Stage 3 Part 1 - Attack Evaluation | DONE | Clean baseline vulnerability confirmed |
| Stage 3 Part 2 - PGD Adversarial Training | DONE | Extended run completed with early stopping at epoch 40/50; best robust val 74.12% at epoch 35 |
| Stage 3 Part 2b - FGSM Defense Branch | DONE | 20 epochs complete; best robust val 72.73% at epoch 19 |
| Stage 4 - Inference API + Demo | IMPLEMENTED + DEPLOYED | /predict and /compare working in rebuilt Docker container on port 5000 |

---

## 2. What Changed In This Update Window

### 2.1 Finalized Stage 3 PGD continuation run

- Canonical run log now includes completion through epoch 40/50.
- Early stopping triggered after 5 no-improvement epochs.
- Best robust val remained 74.12% at epoch 35.

Final completion lines recorded:
- Epoch 40/50 | Train Loss: 0.4675 | Train Acc: 75.14% | Val Clean: 79.72% | Val Robust: 73.87%
- Early stopping at epoch 40. Best robust val acc: 74.12% (epoch 35)

### 2.2 Stage 4 adversarial comparison path fully activated

- inference.py: added AdversarialComparisonEngine and PGD comparison logic.
- app.py: added /compare endpoint and comparison-engine startup handling.
- templates/index.html: replaced with two-tab UI for prediction and adversarial comparison.
- Docker image rebuilt and container redeployed after permissions fix.

### 2.3 Docker and runtime issue resolution

- Docker daemon access was blocked due shell/group context mismatch.
- Access restored by entering docker group context (newgrp docker).
- Old container serving stale API was replaced by rebuilt image.
- /compare changed from 404 to functioning endpoint.

---

## 3. Current Quantitative State

### 3.1 PGD adversarial training (final)

Primary artifacts:
- run_logs/adversarial_train_ 3C2D_Fixed_malex_stage3.log
- Project_Resourse/logs/adversarial_training_log_3c2d.txt
- Project_Resourse/models/3c2d_malex_adversarially_trained.pth
- Project_Resourse/models/at_3c2d_full_checkpoint.pth

Best defended checkpoint summary:
- Best robust val acc: 74.12% (epoch 35)
- End of continuation: early stop at epoch 40/50

### 3.2 FGSM defense branch (reference)

Final FGSM branch summary:
- Best robust val acc: 72.7306% (epoch 19)
- Final run length: 20 epochs

### 3.3 Clean vs AT mirrored attack comparison (post-35 reference)

From Project_Resourse/logs/attack_comparison_3c2d_clean_vs_at_post35.txt:
- Clean (no attack): clean model 85.29%, AT model 80.03% (delta -5.26 pp)
- FGSM e=0.10: clean model 3.49%, AT model 71.39% (delta +67.90 pp)
- PGD e=0.05 steps=40: clean model 0.62%, AT model 74.22% (delta +73.60 pp)

Interpretation:
- Large robustness gains are sustained under strong attacks, with expected clean-accuracy tradeoff.

---

## 4. Stage 4 Runtime State (Now Live)

Deployed container:
- Name: ec499-inference
- Image: ec499-demo
- Port mapping: 0.0.0.0:5000 -> 5000/tcp

Latest endpoint checks:
- GET /health returns status ok and model id (3c2d_clean in current container env)
- POST /compare with invalid non-PE returns 400 and PE validation message
- POST /compare with valid PE returns 200 and full comparison JSON payload

Validation note:
- Latest valid /compare smoke test used benign_00091.exe, so both clean and AT models predicted benign on clean and adversarial inputs. This confirms endpoint correctness, not attack-success demonstration.

---

## 5. Current Problems / Unchecked Requirements

1) Cross-split overlap risk remains open and unresolved.
2) One final mirrored attack-evaluation confirmation pass against the locked best checkpoint is still recommended for audit completeness after run finalization.
3) Stage 4 committee-demo package is not yet fully assembled (fixed malware/benign/invalid sample set + reproducible command/output bundle).
4) Large unrelated untracked datasets and artifacts remain in workspace and should not be included in scoped code commits.

---

## 6. Immediate Next Actions

1) Run final attack-evaluation confirmation against best checkpoint and store dated output.
2) Resolve overlap issue and rerun overlap diagnostics.
3) Create and archive Stage 4 demonstration bundle (inputs, commands, outputs, screenshots).
4) Prepare thesis-facing final robustness table: clean baseline vs PGD-AT best vs FGSM-AT best.