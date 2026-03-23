✅ Phase 1: Dataset Preparation — Verification \& Implementation Guide

1\. Source of Benign PE Files: VALID \& SAFE

Approved Sources:

C:\\Windows\\System32\\ (e.g., notepad.exe, kernel32.dll)

Open-source software installers (e.g., VLC, Firefox)

Why It’s Valid:

Matches academic best practices (Ahmadian et al., 2022).

Legally permissible from your licensed Windows installation.

Critical Filters:

File Extensions: Only .exe, .dll, .sys, .ocx.

PE Validation: Use pefile to skip invalid files (pefile.PEFormatError).

Deduplication: Hash files with hashlib.sha256 to avoid duplicates.





2\. Image Conversion: CORRECTED FOR MALIMG COMPATIBILITY

⚠️ Critical Fix: The original plan used dynamic width + interpolation resizing. This is incorrect.

Malimg uses fixed 256×256 with zero-padding/truncation — NO INTERPOLATION.

Why Interpolation Breaks Compatibility

Interpolation (bilinear/bicubic) smoothes pixel values, destroying sharp byte-level textures (e.g., section boundaries).

Malimg images are lossless reshapes — not resized photos.

Corrected Conversion Method

python sudo code

import numpy as np

from PIL import Image



def pe\_to\_malimg\_compatible\_image(pe\_path, output\_path, target\_size=256):

    with open(pe\_path, 'rb') as f:

        byte\_arr = np.frombuffer(f.read(), dtype=np.uint8)

 

    total\_pixels = target\_size \* target\_size

    if len(byte\_arr) < total\_pixels:

        # Zero-pad to 65,536 bytes







3\. Verification Checklist

Pixel Values: All pixels must be integers in \[0, 255] (no floats).

Image Size: All outputs must be exactly 256x256 (verify with PIL.Image.open().size).

Visual Sanity:

Compare your benign image (e.g., notepad.png) with a Malimg sample (e.g., Allaple.A/01.png).

Both should have identical "grainy" texture — no blurring/smoothing.





4\. Academic Validation

Nataraj et al. (2011):

“We take the first N bytes of the executable and reshape them into a √N × √N grayscale image.”

→ No interpolation; fixed size per dataset.

Malimg Kaggle:

“All images are 256×256 grayscale PNGs.”

Ahmadian et al. (2022):

“Zero-pad or truncate files to 65,536 bytes before reshaping.”





5\. What to Avoid



|Mistake|Consequence|
|-|-|
|Interpolation (bilinear, etc.)|Blurs byte patterns → lower accuracy|
|Dynamic widths + resizing|Breaks Malimg compatibility|
|Skipping PE validation|Corrupted files → training crashes|
|Not deduplicating|Data leakage → inflated metrics|





Final Workflow

Collect benign PE files (with filters/deduplication).

Convert using the corrected pe\_to\_malimg\_compatible\_image function.

Verify all outputs are 256×256, integer-valued, and visually consistent with Malimg.

Combine with Malimg into a balanced binary dataset (benign/ + malicious/).

✅ Outcome: A fully Malimg-compatible dataset ready for CNN training.

