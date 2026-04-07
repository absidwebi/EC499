# MASTER_CONTEXT.md
# EC499 — Master Decision Log & Lessons Learned
**Running record of key decisions, bugs found, fixes applied, and insights gained**

---

## Decision Log

### D1 — Nataraj Variable-Width vs Fixed 256×256
**Decision:** Use Nataraj variable-width images matching Malimg methodology.
**Rejected:** Fixed 256×256 with zero-padding.
**Reason:** Fixed 256×256 means small files produce largely black images. Malimg uses variable widths reflecting actual file size. Using fixed size creates structural mismatch.
**Date:** Early Stage 1

### D2 — No Interpolation Ever
**Decision:** NEAREST interpolation in PadTo256, never bilinear/bicubic.
**Reason:** Bilinear interpolation smooths pixel values, creating blur artifacts. A model trained on blurred malware images and sharp benign images learns to detect blur, not malware. This was the root cause of the first 99% accuracy anomaly.
**Lesson learned:** Even well-intentioned preprocessing choices can introduce cheating shortcuts.

### D3 — ResNet-18 over CustomCNN as Primary Model
**Decision:** Use ResNet-18 as the main architecture after CustomCNN produced suspicious results.
**Reason:** ResNet-18 is the standard in adversarial robustness literature (Madry et al.). Skip connections provide better gradient flow for adversarial training. CustomCNN (4-block) was too simple for meaningful adversarial analysis.
**Note:** CustomCNN results are kept in models/ folder for comparison.

### D4 — Manual FGSM/PGD vs torchattacks Library
**Decision:** Implement FGSM and PGD manually using BCEWithLogitsLoss.
**Reason:** torchattacks uses CrossEntropyLoss internally. Our binary single-logit model uses BCEWithLogitsLoss. The library was producing incorrect gradients for binary classification.
**Implementation detail:** Clamp range must be [-1, 1] not [0, 1] because normalisation uses mean=0.5, std=0.5.

### D5 — Per-Split Benign Index (Critical Bug Fix)
**Decision:** Each split computes its own benign class index rather than reusing train index.
**Reason:** After cross-split deduplication, Yuner.A was removed from val/test. This changed the alphabetical ordering of classes. "benign" moved from index 25 (train) to index 24 (val/test). Using the train index for all splits caused all benign images in val/test to be silently labelled as malware.
**Impact:** This was a critical correctness bug. Detection method: val accuracy was anomalously high in early runs.

### D6 — Class Weights Instead of Downsampling
**Decision:** Use BCEWithLogitsLoss pos_weight to handle class imbalance.
**Rejected:** Downsampling benign to match malware count.
**Reason:** Keeping all benign samples maximises diversity which is important for adversarial defence. More diverse benign representation makes it harder for attackers to craft universal adversarial examples.

### D7 — Adversarial Training Attack Strength
**Decision:** PGD with ε=0.05, α=0.01, 7 steps for training. 20 steps for validation.
**Reason:** Madry default. 7 steps is the minimum for effective hardening. Training with weaker attacks than evaluation creates a false robustness illusion.

### D8 — num_workers=0 in All DataLoaders
**Decision:** Always set num_workers=0 on Linux.
**Reason:** Higher num_workers causes CUDA fork OOM (Out of Memory) errors on Linux because each worker process attempts to reserve its own GPU memory slice.

### D9 — Benign Source: PortableApps + Program Files, NOT System32
**Decision:** Collect benign PE files from installed applications, explicitly excluding System32/SysWOW64.
**Reason:** System32 files are predominantly small system utilities (< 60 KB). They produce sparse, mostly-black 32px/64px images. Malimg malware (especially Allaple.A, 31% of dataset) are large worms producing dense 512px–768px images. This structural gap caused the model to learn density, not malware features.
**Evidence:** Logit gap of 22 units (healthy: 5–10) confirmed structural bias.

### D10 — Per-Vendor Cap of 500 Files
**Decision:** No single software vendor directory contributes more than 500 files.
**Reason:** MATLAB contributed ~5,000 files (25% of total) in v2 collection. All MATLAB DLLs share similar PE structure from the same compiler toolchain. Single-vendor dominance reintroduces structural similarity bias.

### D11 — Fixed Adversarial Test Set
**Decision:** Pre-generate adversarial examples once and save as PNG files.
**Reason:** On-the-fly attack generation produces different perturbations for each model evaluation run due to random PGD initialisation. A fixed test set ensures both clean and AT models are evaluated on identical pixel values, making the comparison scientifically rigorous.

### D12 — EfficientNet-B0 as Comparison Architecture
**Decision:** Add EfficientNet-B0 alongside ResNet-18.
**Reason:** If both architectures show similar logit distributions and evasion rates, this confirms the behaviour is dataset-level (source bias) rather than architecture-specific. EfficientNet-B0 has fundamentally different design (compound scaling, depthwise separable convolutions) vs ResNet's residual blocks.

### D13 — Treat PadTo256 Padding as a Shortcut Risk
**Decision:** Explicitly test and mitigate the padding-layout shortcut created by `PadTo256`.
**Reason:** After `Normalize(mean=0.5, std=0.5)`, padded zeros become exactly `-1.0`. The resulting all-`-1` rows/cols and their geometry correlate with original image dimensions (and therefore file size distribution), enabling a strong non-semantic shortcut.

### D14 — Pad-Matched Dataset Subset as a Control
**Decision:** Build a pad-matched subset dataset (`archive/malimg_dataset_padmatched_v2`) matched on `(pad_rows_all_-1, pad_cols_all_-1, frac_-1_pixels)`.
**Reason:** A pad-matched subset is a clean control to estimate how much performance comes from padding geometry alone.

### D15 — Two Mitigation Paths (Sampling vs Padding Noise)
**Decision:** Implement two mitigation experiments:
- Training-time sampling neutralization via `train.py` (`PAD_NEUTRALIZE_SAMPLER=1`) using a padding-only LR propensity model.
- Transform-time padding noise via `dataset_loader.py` (`PAD_NOISE=1`) that randomizes only the padded bottom-right rectangle.
**Reason:** Sampling attempts to remove the shortcut signal by balancing exposure; padding noise removes the shortcut at the input level while preserving true content.

### D16 — Add Full Resume Checkpointing for Long MaleX Runs
**Decision:** Enable full-checkpoint resume in long-running base-model scripts (starting with 3C2D, then pretrained ResNet-18).
**Reason:** Multi-hour runs should be interruption-safe and allow changing max epochs without losing optimizer/scheduler trajectory.
**Implementation:** Save and restore model, optimizer, scheduler, epoch index, best metrics, and elapsed time.

### D17 — Pause Stage 3 Finalization Until Base-Model Root Cause Is Found
**Decision:** Temporarily pause final adversarial benchmarking.
**Reason:** Both candidate base models failed to show continued validation improvement: 3C2D plateaued after epoch 49 region; pretrained ResNet-18 overfit early.
**Date:** 2026-03-30

---

## Bug Log

### Bug 1 — Bilinear Interpolation Creating Shortcut (Stage 2, Major)
**Symptom:** 99.88% accuracy from Epoch 1, attacks dropping accuracy by only 3–6%.
**Cause:** `transforms.Resize((256,256))` in dataset_loader used bilinear interpolation by default. Malimg images (variable size) were blurred. Benign images (already 256×256) were not. Model learned blur = malware.
**Fix:** Replaced with PadTo256 class using center_crop + zero_pad (no interpolation).

### Bug 2 — Benign Index Mismatch Across Splits (Stage 2, Critical)
**Symptom:** Perfect val/test accuracy despite training on what appeared to be random noise.
**Cause:** Yuner.A removal from val/test changed alphabetical class ordering. Using train's benign index (25) for val/test where benign was at index 24 caused all benign images to be labelled malware.
**Fix:** Per-split index computation with lambda closure `lambda x, i=idx: 0 if x == i else 1`.

### Bug 3 — torchattacks Library Incompatibility (Stage 3)
**Symptom:** CUDA assertion errors, attacks showing < 5% accuracy drop on undefended model.
**Cause:** torchattacks uses CrossEntropyLoss which expects multi-class integer labels. Binary model uses BCEWithLogitsLoss with float labels.
**Fix:** Manually implemented FGSM and PGD using BCEWithLogitsLoss.

### Bug 4 — Clamp Range [-1,1] vs [0,1] (Stage 3)
**Symptom:** Adversarial perturbations producing out-of-range tensors.
**Cause:** Normalisation with mean=0.5, std=0.5 maps pixel values to [-1, 1]. Initial attack code clamped to [0, 1].
**Fix:** All attack functions now clamp to [-1.0, 1.0].

### Bug 5 — GPU Driver Hang During Adversarial Training
**Symptom:** Training process hung silently before first batch.
**Cause:** Multiple Python processes from failed previous runs holding CUDA locks.
**Fix:** `pkill -f python3` to clear zombie processes, then restart training.

### Bug 6 — CUDA OOM with num_workers > 0
**Symptom:** Out of memory error when starting adversarial training.
**Cause:** Each DataLoader worker on Linux attempts to reserve GPU memory, exhausting 8GB VRAM.
**Fix:** Set num_workers=0 in all training and evaluation scripts.

### Bug 7 — Source Bias in Benign Dataset (Dataset Level)
**Symptom:** Logit gap of 22 units (healthy: 5–10), AT model evasion rate 4.49% (literature: 15–35%).
**Cause:** System32/SysWOW64 files are mostly < 60 KB, producing sparse 32/64px images. Malimg malware files are dense 512–768px images. Structural separation trivially separable.
**Fix:** New benign collection from PortableApps + Program Files (18,541 files, 0% at 32/64px).

### Bug 8 — Padding-Layout Shortcut Survives Benign v3 Fix
**Symptom:** Padding-only logistic regression achieves high ROC-AUC using only padding-derived features; trained ResNet-18 exhibits a large logit gap mean (~22 units) despite high accuracy.
**Cause:** `PadTo256` introduces a structured padded region that is easy for the model to exploit. This is independent of the System32-vs-Malimg benign source issue.
**Fix (in progress):** Pad-matching control dataset; sampling neutralization; padding noise transform.

### Bug 9 — 3C2D Validation Plateau After Long Run Extension
**Symptom:** Validation metrics stagnate despite additional epochs and resumed training.
**Evidence:** Best val loss remained 0.3246 at epoch 49; run continued to epoch 60/70 and early-stopped with no improvement.
**Impact:** 3C2D cannot be selected as a clearly improving base model beyond the epoch-49 region.

### Bug 10 — Pretrained ResNet-18 Early Overfitting on MaleX
**Symptom:** Training accuracy rises rapidly (~95.59% by epoch 7) while validation fails to improve and val loss degrades.
**Evidence:** Best val loss 0.3354 at epoch 3; by epoch 7 val loss increased to 0.4565 with val acc ~85.43%.
**Impact:** Pretrained model training was manually stopped pending root-cause diagnosis.

---

## Architecture Evolution

| Version | Architecture | Dataset | Notes |
|---|---|---|---|
| v1 | CustomCNN 4-block | Old benign (System32) | Abandoned — too simple, interpolation bug |
| v2 | ResNet-18 (biased) | Old benign (System32) | Completed but results inflated |
| v3 (current) | ResNet-18 + EfficientNet-B0 | New benign (PortableApps) | Integrated + retrained; padding shortcut mitigation ongoing |

---

## Validation Diagnostics Run

All five diagnostics were run on the biased dataset results. Results documented for comparison against post-fix results.

| Diagnostic | Result | Interpretation |
|---|---|---|
| Logit distribution | Gap: 21.86 units | Confirms source bias |
| Accuracy vs epsilon curve | Textbook shape (correct) | AT works as expected |
| Split-class evasion rate | 97.52% clean, 4.49% AT | Inflated by bias |
| Black-box transfer | 1.71 pp gap | Gradient masking ruled out |
| Saliency maps | Diffuse gradients | Global texture, not structural features |

### Current (v3 benign integrated; padding-neutralize sampler experiment)
**Run log:** `run_logs/train_resnet18_full_padneutralized_trainpy.log`

- Dataset sizes: train=22,288; val=2,663; test=2,702
- Training (ResNet-18, clean): early stop epoch 11, best epoch 6, best val loss 0.0416
- Test accuracy: 0.9848
- Logit gap mean (test): 22.2084 (still large)
- Padding-only LR (subset): ROC-AUC ~0.91 on v3 dataset
**Interpretation:** v3 benign fixed the System32 size bias, but a strong padding-layout shortcut remains.

---

## Literature Benchmarks

| Paper | Model | AT Evasion Rate After Defense |
|---|---|---|
| EICC 2023 (byteplot + GAN AT) | ResNet-50 | 29.55% |
| CMU CCS 2024 (BODMAS) | Custom DNN | 20–40% |
| PMC (Malimg-style DenseNet) | DenseNet | 4–8% misclassification under FGSM |
| **Target (this project, corrected)** | **ResNet-18** | **15–35%** |

---

## Environment Notes

- **OS:** Ubuntu (local machine)
- **Python:** 3.10.12 via `/home/alucard-00/EC499/Project_Resourse/venv/`
- **GPU:** RTX 4060, 8 GB VRAM
- **PyTorch:** 2.x with CUDA
- **Training time:** ~12 min per epoch (clean), ~6 min per epoch (AT with 7-step PGD)

---

## Thesis Reporting Notes

- Current biased results should appear in thesis as "Dataset Bias Analysis" section
- Show before/after comparison to demonstrate the importance of benign source selection
- Cite: Nataraj 2011 (Malimg), Madry 2018 (PGD), PortableApps (benign source)
- Report TPR@FPR=1% alongside accuracy (literature standard per CCS 2024)
- Black-box transfer test result (1.71 pp gap) is a strong methodological contribution

---

## 2026-03-29 Execution Snapshot (MaleX)

### S1. Dataset switch and script wiring

- `config.py` now contains dedicated MaleX constants and MaleX-specific model paths.
- `dataset_loader.py` was replaced with the MaleX-specific binary loader variant (grayscale + tensor + normalize, no PadTo256).
- `train.py`, `evaluate_attacks.py`, and `adversarial_train.py` were rewired to use MaleX constants.

### S2. MaleX clean baseline completion

- Clean training completed (early stop at epoch 7).
- Best checkpoint saved as `models/resnet18_malex_clean_vulnerable.pth`.
- Validation curve indicates expected overfitting pattern after epoch 2, matching early-stopping trigger.

### S3. Vulnerability confirmation

- Stage 3 attack evaluation produced major accuracy collapse under FGSM/PGD, confirming vulnerability of the clean baseline.
- This validates the need for adversarial training in the MaleX track.

### S4. Adversarial defense run status

- Adversarial training was launched successfully, warm-starting from the clean MaleX checkpoint.
- Current run is active with stable batch-level loss logging and no startup/runtime failures.

### S5. Open risk preserved in context

- Hash-overlap diagnostic (`check_malex_hash_overlaps.py`) still flags cross-split content overlap.
- This remains tracked as a reproducibility/methodology risk item for the MaleX branch and should be resolved before final thesis-grade benchmarking.

### S6. Agent-prompt phase gate ledger

To preserve exact traceability against `AGENT_PROMPT_MaleX_Integration.md`, current gate status is:

- Phase 1: PASS
- Phase 2: PASS
- Phase 3: PASS
- Phase 4: PASS
- Phase 5: PASS
- Phase 6A: FAIL (hash overlap)
- Phase 6B: PASS
- Phase 6C: PASS
- Phase 7: PASS (clean MaleX model training completed)
- Stage 3 Part 1: PASS (attack vulnerability completed)
- Stage 3 Part 2: RUNNING (adversarial training in progress)

---

## 2026-03-31 Delta Update (Stage 2 Final Evaluation + Stage 3 Preparation)

### D18 - Stage 2 Final Selector Uses Full Test-Metric Bundle
Decision:
- Select the Stage 3 baseline using a combined metric view (accuracy, F1 macro, F1 malware, AUC, confusion matrix, per-class logit stats), not a single metric.

Reason:
- The two candidates were very close; a single metric could hide class-specific tradeoffs.

Outcome:
- 3C2D selected for Stage 3 due to slight edge in accuracy and malware F1.

### D19 - Standardize Stage 3 Scripts with Explicit Model Variants
Decision:
- Add explicit model-variant routing in both attack evaluation and adversarial training scripts.

Variants:
- `3c2d`
- `resnet`
- `resnet_pretrained`

Reason:
- Prevent hardcoded model mismatch and make Stage 3 reproducible when switching baselines.

### D20 - Model-Tagged Log Naming
Decision:
- Save Stage 3 logs using model-tagged filenames.

Examples:
- `attack_evaluation_results_3c2d.txt`
- `adversarial_training_log_3c2d.txt`

Reason:
- Avoid accidental overwrite and preserve traceability across model variants.

### D21 - Enforce venv Interpreter for All Runs
Decision:
- All commands in this cycle were constrained to:
	`/home/alucard-00/EC499/Project_Resourse/venv/bin/python`

Reason:
- Ensure dependency consistency and avoid system-Python mismatch.

### D22 - Keep Stage 3 Running While Deferring Unrelated Binary Churn
Decision:
- Continue Stage 3 work without acting on unrelated large binary additions under dataset folders.

Reason:
- Those changes are not part of model-code correctness and would pollute commit scope.

### D23 - Complete Initial 3C2D Adversarial Defense Baseline
Decision:
- Finalize one full adversarial training baseline pass for the selected 3C2D Stage 3 model before extending runs.

Reason:
- Establish a concrete defended checkpoint and measurable robustness delta against the clean baseline.

Outcome:
- Completed run produced best robust validation of 64.91% and saved defended weights.

### D24 - Enforce Live File Logging for Long Stage 3 Runs
Decision:
- Launch long-running training/evaluation commands with explicit live file logging in `run_logs/`.

Reason:
- Terminal-attached output alone is not sufficient for monitoring/recovery and reproducibility trace.

Outcome:
- Active Stage 3 logs now include explicit tee-based run artifacts.

### D25 - Harden `adversarial_train.py` for Resume-Safe Training
Decision:
- Add full-checkpoint resume support, per-epoch checkpointing, curve plotting, and robust-validation early stopping.

Reason:
- Multi-hour adversarial runs must be interruption-safe and restartable without losing optimizer/scheduler state.

Outcome:
- Script now supports deterministic continuation and future extension beyond baseline epoch budget.

---

## Additional Bug Log Entries (2026-03-31)

### Bug 11 - JSON Serialization Failure in Base-Model Evaluator
Symptom:
- `evaluate_base_models_testset.py` computed metrics successfully, then failed while writing JSON.

Cause:
- Non-serializable NumPy objects (`ndarray`) passed directly to `json.dump`.

Fix:
- Convert confusion matrix to list and cast scalar outputs to native Python floats.

Impact:
- Final results now persist in `base_model_testset_results.json` for downstream docs/comparison.

### Bug 12 - Stage 3 Script Hardcoded to One Architecture
Symptom:
- `evaluate_attacks.py` and `adversarial_train.py` originally assumed one fixed model path/model class.

Cause:
- No abstraction for selecting among 3C2D, clean ResNet, and pretrained ResNet flows.

Fix:
- Added model-bundle helper functions and `MODEL_VARIANT` selector.

Impact:
- Stage 3 now aligns with Stage 2 selection without manual script rewrites.

### Bug 13 - Incorrect Log Variable in Adversarial Training
Symptom:
- End-of-run log write path reference could fail or write to wrong variable.

Cause:
- Used `LOG_PATH` instead of runtime `log_path` in final write section.

Fix:
- Use runtime `log_path` consistently.

Impact:
- Model-tagged adversarial training logs now write predictably.

---

## 2026-03-31 Execution Snapshot

### Stage 2 final evaluation
- Script: `evaluate_base_models_testset.py`
- Output: `base_model_testset_results.json`
- Result: both candidates exceed paper baselines; 3C2D selected.

### Stage 3 part 1
- Script: `evaluate_attacks.py`
- Output: `logs/attack_evaluation_results_3c2d.txt`
- Result: clean model strongly vulnerable under FGSM/PGD, confirming need for defense training.

### Stage 3 part 2
- Script: `adversarial_train.py`
- Result: completed baseline 3C2D defense run.
- Best robust validation: 64.91%
- Final epoch metrics: train loss 0.5619, train acc 65.93%, val clean 73.29%, val robust 64.91%
- Output checkpoint: `models/3c2d_malex_adversarially_trained.pth`

---

## 2026-04-01 Delta Update (Stage 3 Completion + Comparison)

### S7. Stage 3 defense run completion

- Primary training log: `run_logs/adversarial_train_ 3C2D_Fixed_malex_stage3.log`
- Internal per-epoch log: `logs/adversarial_training_log_3c2d.txt`
- Defended checkpoint: `models/3c2d_malex_adversarially_trained.pth`

### S8. Clean vs defended robustness comparison completed

- Comparison run log: `run_logs/evaluate_attacks_3c2d_clean_vs_defended_stage3.log`
- Structured comparison output: `logs/attack_comparison_3c2d_before_after_stage3.json`

Selected outcomes:
- Clean accuracy: 85.29% -> 73.07% (tradeoff)
- FGSM eps=0.10: 3.49% -> 61.16% (+57.67 pp)
- PGD eps=0.05, steps=40: 0.64% -> 64.61% (+63.98 pp)

Interpretation:
- Defense achieved major robustness gains under both FGSM and PGD.
- Clean-performance drop is within expected robustness tradeoff behavior.

### S9. Script hardening applied for next Stage 3 extensions

`adversarial_train.py` now includes:
- `NUM_EPOCHS = 20`
- `EARLY_STOP_PATIENCE = 5`
- `RESUME_IF_CHECKPOINT_EXISTS = True`
- full-checkpoint save/load helpers
- per-epoch curve generation and log persistence

This enables safe continuation runs targeting robust val > 65% without restarting from scratch.

---

## 2026-04-02 Delta Update (Stage 3 Continuation + Stage 4 Implementation)

### D26 - Resume PGD Training from Last Robust State Instead of Restarting
Decision:
- Continue PGD adversarial training from saved state and append to the same canonical run log.

Reason:
- Preserve training history, avoid losing multi-hour compute progress, and satisfy continuity constraints.

Outcome:
- Full-checkpoint state currently tracks continuation through epoch 15 with best robust val 71.57% and active progress into later epochs.

### D27 - Add Fixed-Set Deterministic Evaluation as Primary Comparison View
Decision:
- Use pre-generated fixed adversarial subsets for clean vs defended comparison in addition to on-the-fly attack reports.

Reason:
- Deterministic pixel-identical adversarial inputs remove run-to-run variance and strengthen scientific reproducibility.

Outcome:
- Final fixed-set table now includes clean, PGD-defended, and FGSM-defended models under identical FGSM/PGD subsets.

### D28 - Build Parallel FGSM Defense Branch Under Same Resume Framework
Decision:
- Add a dedicated FGSM adversarial training script with full checkpoint resume, early stopping, and curves.

Reason:
- Provide a second defense baseline under the same reporting rigor and allow direct robustness tradeoff comparison.

Outcome:
- FGSM branch completed all 20 epochs with best robust validation 72.73%.

### D29 - Stage 4 Inference Must Use Strict Static PE Validation (No Bypass)
Decision:
- Re-enable strict PE validation in inference pipeline and remove validation-bypass default behavior.

Reason:
- The deployment proposal requires static analysis; pefile is a static parser and does not execute binaries.

Outcome:
- Inference pipeline now validates PE structure via pefile and returns structured ValueError JSON for non-PE uploads.

### D30 - Local API First, Docker Pending Environment Enablement
Decision:
- Complete and validate local Flask inference flow first; keep Docker validation pending when docker binary is unavailable.

Reason:
- Environment-level tooling constraints should not block code-level pipeline completion.

Outcome:
- Local health/predict tests pass end-to-end; Docker build/run remains blocked by missing docker + privileged install requirement.

---

## Additional Bug Log Entries (2026-04-02)

### Bug 14 - Missing PGD Full-Checkpoint in Earlier Baseline Continuation Attempt
Symptom:
- Resume metadata query failed to find `at_3c2d_full_checkpoint.pth` in an earlier continuation attempt.

Cause:
- Initial baseline run lineage did not persist expected full-checkpoint state at that time.

Fix:
- Re-established full-checkpoint continuity and resumed with explicit checkpoint tracking and append-only log routing.

Impact:
- Current PGD continuation now has valid resume metadata (epoch, best metric, no-improve counter).

### Bug 15 - Benign Test Path Mismatch for Stage 4 Commands
Symptom:
- Guide commands referenced `/home/alucard-00/EC499/benign_pe_files/benign_00001.exe` while local files existed under `Project_Resourse/benign_pe_files_test/`.

Cause:
- Workspace path naming divergence.

Fix:
- Added local compatibility symlink to satisfy guide command path without changing scripts.

Impact:
- Stage 4 command sequence runs consistently in current workspace.

### Bug 16 - Docker Runtime Validation Blocked by Host Tooling
Symptom:
- Docker build commands unavailable.

Cause:
- `docker` binary missing in environment; non-interactive sudo install not permitted.

Fix:
- Documented blocker and preserved Dockerfile for immediate execution once tooling is available.

Impact:
- Stage 4 code is ready; container validation pending environment enablement.

---

## 2026-04-02 Execution Snapshot

### Stage 3 continuation (PGD)
- Script: `adversarial_train.py`
- Canonical run log: `run_logs/adversarial_train_ 3C2D_Fixed_malex_stage3.log`
- Current saved checkpoint state:
	- epoch_zero_based=14 (resume target epoch 16)
	- best_robust_val_acc=71.57% at epoch 15

### Stage 3 fixed-set pipeline
- Generator: `generate_malex_adv_testset.py`
- Evaluator: `evaluate_attacks_fixed.py`
- Final all-model fixed-set comparison confirms major robustness gains for PGD and FGSM defended models versus clean baseline.

### Stage 3 FGSM branch
- Script: `adversarial_train_fgsm.py`
- Completed 20 epochs; best robust val 72.73% at epoch 19.

### Stage 4 inference/demo implementation
- `inference.py`, `app.py`, `templates/index.html`, `Dockerfile` implemented.
- Local integration tests pass for:
	- `GET /health`
	- `POST /predict` with valid PE
	- `POST /predict` with non-PE (returns validation error JSON)
- Docker verification pending host-side docker availability.

---

## 2026-04-03 to 2026-04-04 Delta Update (Post-35 Evaluation + Training Extension)

### D31 - Evaluate Clean 3C2D at Post-35 Milestone Before Further Continuation
Decision:
- Run `evaluate_attacks.py` with `MODEL_VARIANT='3c2d'` at the epoch-35 milestone to lock a clean-model reference table before extending training.

Reason:
- Preserve a stable clean baseline snapshot for later side-by-side comparison with defended checkpoints.

Outcome:
- Clean vulnerability table persisted to `logs/attack_evaluation_results_3c2d.txt` and `run_logs/evaluate_attacks_3c2d_post35.log`.

### D32 - Mirror the Same Metrics on AT Model Without Re-running Clean Baseline
Decision:
- Run a dedicated AT evaluation pass using the exact same FGSM/PGD settings used for clean-model evaluation.
- Do not re-run clean baseline.

Reason:
- Ensure direct comparability while avoiding redundant computation and accidental baseline drift.

Outcome:
- AT table saved to `logs/attack_evaluation_results_3c2d_at.txt`.
- Side-by-side delta report saved to `logs/attack_comparison_3c2d_clean_vs_at_post35.txt`.

### D33 - Extend PGD Adversarial Training Epoch Budget to 50
Decision:
- Modify `adversarial_train.py` from `NUM_EPOCHS = 20` to `NUM_EPOCHS = 50` and continue from full checkpoint.

Reason:
- Continue robust training trajectory without resetting optimizer/scheduler state.

Outcome:
- Training resumed and progressed into epoch 37/50.
- Checkpoint now records best robust validation 74.1239% at epoch 35.

### D34 - Keep Canonical Stage 3 Run Log Append-Only During Resume Cycles
Decision:
- Continue using `run_logs/adversarial_train_ 3C2D_Fixed_malex_stage3.log` as the authoritative append-only training trace.

Reason:
- Single-source traceability for interruption/resume events and thesis reproducibility.

Outcome:
- Latest completed epoch summary in canonical log: Epoch 36/50, Val Robust 73.97%.

### D35 - Stage 4 Environment State Updated (Docker Available)
Decision:
- Mark host-side Docker availability as restored and update context docs accordingly.

Reason:
- Previous context stated Docker was unavailable; that is no longer accurate (`docker --version` present).

Outcome:
- Stage 4 status now tracked as implemented/validated with one residual caveat: isolated-network host reachability behavior during strict reproduction mode.

---

## Additional Bug Log Entries (2026-04-03 to 2026-04-04)

### Bug 17 - Module Import Failure When Launching AT Evaluation from Repo Root
Symptom:
- `ModuleNotFoundError: No module named 'config'` during AT evaluation attempt.

Cause:
- Script-local imports expected execution from `Project_Resourse` working directory.

Fix:
- Re-run from the `Project_Resourse` directory.

Impact:
- AT evaluation completed successfully with aligned metrics.

### Bug 18 - GPU Resource Contention Risk Between Long Training and Attack Evaluation
Symptom:
- Running heavy evaluation while PGD training is active can produce contention and unstable throughput.

Cause:
- Both tasks are compute-heavy and share the same device resources.

Fix:
- Pause/stop active training before AT evaluation, run evaluation, then resume training from checkpoint.

Impact:
- Preserved deterministic evaluation workflow and maintained checkpoint-safe continuation.

### Bug 19 - Context Drift in Stage 4 Docker Availability Statement
Symptom:
- Docs reported missing Docker binary while host now has Docker.

Cause:
- Environment changed after prior context snapshot.

Fix:
- Re-check environment and update all context files to reflect current availability.

Impact:
- Planning context no longer carries stale environment assumptions.

---

## 2026-04-04 Execution Snapshot

### Stage 3 continuation state
- Active process: `/home/alucard-00/EC499/Project_Resourse/venv/bin/python /home/alucard-00/EC499/Project_Resourse/adversarial_train.py`
- Current checkpoint (`models/at_3c2d_full_checkpoint.pth`):
	- epoch_zero_based=35
	- resume_next_epoch_1based=37
	- best_robust_val_acc=74.1239%
	- best_epoch=35
	- epochs_no_improve=1

### Post-35 clean vs AT attack results
- Clean model (from `attack_evaluation_results_3c2d.txt`):
	- Clean 85.29%
	- FGSM e=0.10: 3.49%
	- PGD e=0.05 (40): 0.62%
- AT model (from `attack_evaluation_results_3c2d_at.txt`):
	- Clean 80.03%
	- FGSM e=0.10: 71.39%
	- PGD e=0.05 (40): 74.22%
- Side-by-side delta (AT-clean):
	- Clean: -5.26 pp
	- PGD e=0.05 (40): +73.60 pp
	- FGSM e=0.10: +67.90 pp

### Stage 4 runtime state
- `app.py` service process is running.
- Local inference endpoint behavior remains consistent with static PE-validation requirements.

---

## 2026-04-04 Final Delta Update (Stage 3 Finalization + Stage 4 Compare Deployment)

### D36 - Lock the best PGD checkpoint at epoch 35 after continuation to 50
Decision:
- Continue the PGD run to the 50-epoch cap and rely on early stopping to decide if post-35 continuation provides meaningful gains.

Reason:
- Prevent premature stopping and confirm whether robustness can exceed the epoch-35 best value.

Outcome:
- Run completed at epoch 40/50 via early stopping.
- Best robust validation remained 74.12% at epoch 35.
- Epoch 35 checkpoint remains the selected defended model for reporting.

### D37 - Add Stage 4 adversarial comparison endpoint and full UI flow
Decision:
- Extend inference deployment from single prediction to full clean-vs-adversarial comparison using two models (clean and AT).

Reason:
- This is the core demonstration requirement for Stage 4 and directly reflects the thesis contribution.

Outcome:
- `inference.py` now includes `AdversarialComparisonEngine` and PGD comparison logic.
- `app.py` now exposes `POST /compare`.
- `templates/index.html` now includes a dedicated adversarial comparison tab and rendering path.

### D38 - Rebuild and replace running Docker container after permissions fix
Decision:
- Rebuild `ec499-demo` and recreate `ec499-inference` from current source once docker socket access was restored.

Reason:
- Previous running container served stale API code and returned 404 on `/compare`.

Outcome:
- Current container serves updated code.
- `/compare` now responds as expected in the live container.

---

## Additional Bug Log Entries (2026-04-04)

### Bug 20 - Docker socket ACL command used invalid shell variable syntax
Symptom:
- `docker ps` continued to fail with permission denied even after ACL command attempt.

Cause:
- Command used `"$alucard-00"`, which is not a valid shell variable expansion for the username.

Fix:
- Use proper user literal for ACL and refresh group context with `newgrp docker`.

Impact:
- Docker daemon access restored in active shell context.

### Bug 21 - Stale running container missing `/compare` route
Symptom:
- `/compare` returned 404 while code in workspace already included endpoint.

Cause:
- Existing container was running an older image.

Fix:
- Rebuild image and recreate container.

Impact:
- `/compare` route now available and validated in deployed container.

---

## 2026-04-04 Final Execution Snapshot

### Stage 3 final training state
- Canonical log: `run_logs/adversarial_train_ 3C2D_Fixed_malex_stage3.log`
- Final continuation endpoint:
	- Epoch 40/50 | Train Loss 0.4675 | Train Acc 75.14% | Val Clean 79.72% | Val Robust 73.87%
	- Early stop triggered after 5 no-improvement epochs
- Best robust checkpoint:
	- Robust val: 74.12% at epoch 35
	- Weights: `models/3c2d_malex_adversarially_trained.pth`

### Stage 3 clean vs AT post35 comparison (retained reference)
- Clean (no attack): 85.29% clean model vs 80.03% AT model
- FGSM e=0.10: 3.49% clean model vs 71.39% AT model
- PGD e=0.05, steps=40: 0.62% clean model vs 74.22% AT model

### Stage 4 live deployment state
- Container `ec499-inference` running on port 5000 from rebuilt `ec499-demo` image.
- `GET /health` returns status JSON.
- `POST /compare` checks:
	- invalid non-PE upload -> HTTP 400 with PE validation message
	- valid PE upload -> HTTP 200 with comparison payload (`byteplot_b64`, `clean_256_b64`, `adv_256_b64`, `clean_model`, `at_model`, `attack_params`)


---

## 2026-04-07 Delta Update (Metrics Harmonization + Provenance Upgrade + Doc Sync)

### D39 - Harmonize Stage 2/3 evaluators to full confusion-style metrics
Decision:
- Upgrade key evaluation scripts from accuracy-centric output to a richer metric bundle.

Scripts impacted:
- evaluate_base_models_testset.py
- evaluate_attacks.py
- evaluate_attacks_fixed.py

Reason:
- Thesis reporting and cross-stage comparison require class-level and threshold-aware metrics, not only top-line accuracy.

Outcome:
- Added confusion counts (TN/FP/FN/TP), precision/recall per class, specificity, FPR/FNR, balanced accuracy, MCC, AUC handling, and TPR@FPR thresholds.

### D40 - Keep deterministic fixed-set logic while expanding clean baseline context
Decision:
- Preserve malware-only deterministic fixed-set attack comparison protocol in evaluate_attacks_fixed.py while adding full clean-test metric reporting.

Reason:
- Deterministic adversarial comparison is still core methodology, but it must be interpreted alongside complete clean-set quality.

Outcome:
- Fixed-set recall/evasion workflow retained.
- Clean evaluation output now includes full metric suite for each compared model.

### D41 - Add checkpoint-sidecar provenance in adversarial_train.py
Decision:
- Introduce sidecar metadata JSON writer for best defended weights and add resume backfill behavior.

Reason:
- Improve reproducibility traceability by persisting run config, attack config, and best-checkpoint identity next to saved best weights.

Outcome:
- Metadata writer integrated in best-checkpoint save path and resume flow.
- Historical best artifact currently has no sidecar yet (feature added after that artifact was created).

### D42 - Clarify Docker networking claim using direct runtime validation
Decision:
- Update project context to explicitly state observed networking behavior: --network=none with -p was not browser-reachable in this environment.

Reason:
- Prevent reporting an unverified networking assumption and keep Stage 4 deployment claims evidence-based.

Outcome:
- Context now distinguishes two truths:
  - Bridge + -p is needed for local demo reachability here.
  - Current Docker package is reproducible for Stage 4 inference demo, not full training pipeline reproduction.

---

## Additional Bug Log Entries (2026-04-07)

### Bug 22 - Stage docs drifted behind upgraded evaluator outputs
Symptom:
- Context documents still described older accuracy-focused evaluator behavior.

Cause:
- Python script upgrades landed before synchronized markdown refresh.

Fix:
- Rewrite PROJECT_CONTEXT.md, CURRENT_STATE.md, and EC499_Folder_Structure.md with verified latest script and artifact state.

Impact:
- Documentation now matches current code behavior and checked artifact counts.

### Bug 23 - Sidecar metadata availability assumed before artifact generation
Symptom:
- Workflow narrative implied best-weights sidecar metadata already existed for current defended model.

Cause:
- Metadata writer was added after historical best checkpoint was already produced.

Fix:
- Context now states clearly: capability exists in code, but current historical artifact has no sidecar file yet.

Impact:
- Avoids provenance overclaim and preserves audit accuracy.

### Bug 24 - Ambiguity between fixed-set directories
Symptom:
- Two fixed adversarial directories coexisted without clear primary designation.

Cause:
- Legacy smaller set and MaleX-scale set both remained in repository.

Fix:
- Documentation now labels adversarial_test_set_malex as primary and adversarial_test_set as legacy/smaller.

Impact:
- Reduces interpretation mistakes during evaluation and thesis writing.

---

## 2026-04-07 Execution Snapshot

### Repository and branch state
- Active branch: stage4-demo
- Modified tracked code files in scope:
  - Project_Resourse/evaluate_base_models_testset.py
  - Project_Resourse/evaluate_attacks.py
  - Project_Resourse/evaluate_attacks_fixed.py
  - Project_Resourse/adversarial_train.py
- New untracked requested doc file present:
  - Project_Resourse/Tamplate_of_Graduation_Project.md

### Verified quantitative facts refreshed
- MaleX test split:
  - benign = 17,153
  - malware = 15,851
  - total = 33,004
- Primary fixed adversarial set (malex):
  - fgsm_eps0.05 PNG count = 15,851
  - pgd_eps0.05_steps40 PNG count = 15,851
- Legacy fixed set:
  - fgsm_eps0.05 PNG count = 847
  - pgd_eps0.05_steps40 PNG count = 847

### Checkpoint provenance refresh
- at_3c2d_full_checkpoint.pth:
  - epoch_zero_based = 39
  - best_robust_val_acc = 74.12388308344691
  - best_epoch = 35
- Sidecar metadata file for current best weights:
  - not present yet
