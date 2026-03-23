# Priority 1 — Root Cause Fix: Implementation Plan
## Nataraj-Faithful Benign Image Conversion

**Date:** Stage 1 Remediation  
**Status:** Ready to Execute  
**Author:** EC499 Supervisor

---

## 1. Problem Statement (What We Are Fixing)

The current `convert_to_malimg.py` always produces a fixed **256×256** image by
zero-padding or truncating every benign PE file to exactly 65,536 bytes before
reshaping. This creates a **structural source bias**:

| Class | Conversion | Result |
|---|---|---|
| **Benign** (ours) | Always 256×256 — zero-padded | Mostly black bottom region for small files |
| **Malware** (Malimg) | Variable dimensions from Nataraj method | Dense byte texture across the full image |

When `dataset_loader.py` NEAREST-resizes both to 256×256, the benign images
still carry a large dark region while malware images look uniformly textured.
The CNN learns this trivial "black region = benign" shortcut instead of genuine
binary structure — explaining the 95% Train / 72% Val gap on epoch 1.

---

## 2. The Fix: Apply the Nataraj Width Table to Benign Conversion

Malimg malware images were created using the Nataraj method where **image width
is determined by file size**. We must apply the **identical conversion logic** to
our benign PE files so both classes go through the same pipeline.

### Nataraj Width Table

| PE File Size | Assigned Width | Resulting Image Shape |
|---|---|---|
| < 10 KB | 32 px | 32 × H |
| 10 KB – 30 KB | 64 px | 64 × H |
| 30 KB – 60 KB | 128 px | 128 × H |
| 60 KB – 100 KB | 256 px | 256 × H |
| 100 KB – 200 KB | 384 px | 384 × H |
| 200 KB – 500 KB | 512 px | 512 × H |
| > 500 KB | 768 px | 768 × H |

Height = `ceil(file_size_bytes / width)`.

This produces variable-size images — exactly as Malimg is structured.
`dataset_loader.py` (already configured for NEAREST interpolation) then resizes
both benign and malware images to 256×256 identically. The texture density
distribution of both classes becomes compatible.

---

## 3. What Changes and What Does Not

### Files That Change

| File | Change | Reason |
|---|---|---|
| `convert_to_malimg.py` | Full rewrite of conversion logic | Apply Nataraj width table |
| `config.py` | Add `BENIGN_IMAGES_NATARAJ_DIR` constant | New output directory for converted images |

### Files That Do NOT Change

| File | Reason Unchanged |
|---|---|
| `dataset_loader.py` | Already uses NEAREST resize to 256×256 — exactly what we need |
| `split_benign_dataset.py` | Reads from the benign images directory — we just point it to the new one |
| `models.py` | Architecture unchanged |
| `train.py` | Training logic unchanged |
| `evaluate_attacks.py` | Stage 3 unchanged |
| `adversarial_train.py` | Stage 3 unchanged |

### Key Design Decisions

- **Minimum file size guard:** PE files under **1 KB** are skipped — they produce
  images with near-zero content and contribute no useful signal.
- **Maximum file size guard:** PE files over **10 MB** are skipped — these produce
  enormous images that are slow to process and then heavily cropped by NEAREST
  resize; the first 65,536 bytes (the Malimg default) would have been more
  representative. You can raise this limit if needed.
- **Output directory is new:** `benign_images_nataraj/` — the old
  `benign_images_256x256/` directory is left untouched so no prior work is lost.
- **Width-distribution statistics** are printed at the end so you can verify
  your benign PE file collection spans a healthy range of sizes.
- **All PE extensions supported:** `.exe`, `.dll`, `.sys`, `.ocx` — not just `.exe`.

---

## 4. Execution Order (Step by Step)

### Step 1 — Update `config.py`
Add the `BENIGN_IMAGES_NATARAJ_DIR` path constant. (Included in the script below
as an instruction — apply the one-line addition shown.)

### Step 2 — Replace `convert_to_malimg.py`
Drop in the new version of the script (provided in Section 5).

### Step 3 — Run the conversion
```bash
/home/alucard-00/EC499/venv/bin/python \
  /home/alucard-00/EC499/Project_Resourse/convert_to_malimg.py
```
Watch the width-distribution report printed at the end. A healthy distribution
looks like most samples falling in the 64–256 width range, with some at 32 and
some at 384–512. If 90%+ of your files fall at width=32, your PE collection
is dominated by tiny files and the source bias will remain — recollect from
Program Files or larger software installers.

### Step 4 — Verify pixel density parity
Run the density check (built into the new script's `--verify` mode):
```bash
/home/alucard-00/EC499/venv/bin/python \
  /home/alucard-00/EC499/Project_Resourse/convert_to_malimg.py --verify
```
This samples 200 images from each class and prints mean ± std pixel intensity.
Success criterion: the means should be within **30 pixel units** of each other
(both classes on a 0–255 scale). If benign mean is still < 20 and malware mean
is > 100, the source bias persists and you need more large benign PE files.

### Step 5 — Update `split_benign_dataset.py` source directory
Change the `SOURCE_BENIGN_DIR` to read from `BENIGN_IMAGES_NATARAJ_DIR_STR`
instead of the old `BENIGN_IMAGES_DIR_STR`. This is a one-line config import
change.

### Step 6 — Re-run `split_benign_dataset.py`
This distributes the new Nataraj-converted benign images into
`archive/malimg_dataset/train|val|test/benign/`.

```bash
/home/alucard-00/EC499/venv/bin/python \
  /home/alucard-00/EC499/Project_Resourse/split_benign_dataset.py
```

### Step 7 — Re-run `check_hash_overlaps.py`
Confirm zero hash overlaps after the fresh split.

### Step 8 — Retrain
```bash
/home/alucard-00/EC499/venv/bin/python \
  /home/alucard-00/EC499/Project_Resourse/train.py
```

---

## 5. Success Criteria

| Check | Expected Result |
|---|---|
| Width distribution | Images span at least 3 different widths (32, 64, 128, 256) |
| Benign mean pixel intensity | > 50 (not predominantly black) |
| Malware mean pixel intensity | < 160 (not blown out) |
| Difference between class means | < 30 pixel units |
| Hash overlaps (train/val/test) | 0 |
| Epoch 1 Train Accuracy | 65 – 82% |
| Epoch 1 Val Accuracy | 60 – 78% |
| Train – Val gap | < 12 percentage points |

---

## 6. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Too few PE files in 32KB–512KB range | Medium | The script reports a size histogram; recollect from Program Files if needed |
| Bias persists despite width table | Low | Run `--verify` check; if mean gap > 30, filter out files below 10KB |
| Benign image count drops below 9K | Medium | Use all valid PE extensions (.exe + .dll + .sys); do not restrict to .exe only |
| NEAREST resize introduces block artifacts | Not a risk | This is identical to how Malimg images are handled in the loader |
