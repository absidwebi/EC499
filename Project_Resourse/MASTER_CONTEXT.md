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
