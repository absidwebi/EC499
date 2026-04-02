"""
generate_malex_adv_testset.py
==============================
Generates a FIXED adversarial test set from the MaleX test split using the
clean 3C2D model as the attacker.

Scientific purpose:
    Pre-generating adversarial examples eliminates stochastic variation
    between evaluation runs (PGD starts from a random point inside the
    epsilon ball). Both the clean and adversarially trained models are then
    evaluated against the IDENTICAL pixel values, making comparisons
    scientifically rigorous.

Threat model:
    Only malware images are attacked. The adversary wants malware to evade
    detection (malware -> classified as benign). Attacking benign images has
    no real-world motivation.

Output structure:
    adversarial_test_set_malex/
    |- fgsm_eps0.05/
    |  |- images/          <- adversarial PNGs (one per malware test sample)
    |  `- labels.txt       <- ground truth labels (all 1 = malware)
    |- pgd_eps0.05_steps40/
    |  |- images/
    |  `- labels.txt
    `- README.txt

Usage:
    # Generate the fixed set (run once)
    python generate_malex_adv_testset.py

    # Verify the set was created correctly
    python generate_malex_adv_testset.py --verify
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MALEX_DATASET_DIR_STR,
    MALEX_ADV_TEST_SET_DIR_STR,
    MALEX_3C2D_CLEAN_MODEL_PATH_STR,
    PROJECT_ROOT,
)
from dataset_loader import get_data_loaders
from models import MaleX3C2D

# ============================================================
# ATTACK CONFIGURATIONS
# Each config defines one sub-folder of adversarial examples.
# ============================================================
ATTACK_CONFIGS = [
    {
        "name": "fgsm_eps0.05",
        "type": "fgsm",
        "eps": 0.05,
        "alpha": None,
        "steps": 1,
    },
    {
        "name": "pgd_eps0.05_steps40",
        "type": "pgd",
        "eps": 0.05,
        "alpha": 0.01,
        "steps": 40,
    },
]

BATCH_SIZE = 32
NUM_WORKERS = 0
_criterion = nn.BCEWithLogitsLoss()


# ============================================================
# Attack implementations
# ============================================================

def fgsm_attack(model, images, labels, eps):
    """Fast Gradient Sign Method (Goodfellow et al. 2014)."""
    adv = images.clone().detach().requires_grad_(True)
    labels_f = labels.float().unsqueeze(1)
    loss = _criterion(model(adv), labels_f)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        adv = torch.clamp(adv + eps * adv.grad.sign(), -1.0, 1.0)
    return adv.detach()


def pgd_attack(model, images, labels, eps, alpha, steps):
    """Projected Gradient Descent (Madry et al. 2018)."""
    labels_f = labels.float().unsqueeze(1)
    adv = images.clone().detach()
    adv = torch.clamp(adv + torch.empty_like(adv).uniform_(-eps, eps), -1.0, 1.0)
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = _criterion(model(adv), labels_f)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            delta = torch.clamp(adv - images, -eps, eps)
            adv = torch.clamp(images + delta, -1.0, 1.0)
    return adv.detach()


def tensor_to_png(tensor_img):
    """Convert normalised [-1,1] single-channel tensor to uint8 PIL image."""
    arr = tensor_img.squeeze(0).cpu().numpy()
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


# ============================================================
# Generation
# ============================================================

def generate_fixed_set(model, test_loader, attack_cfg, output_dir, device):
    """
    Attack all malware samples in test_loader and save adversarial PNGs.
    Returns count of saved images.
    """
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    attack_type = attack_cfg["type"]
    eps = attack_cfg["eps"]
    alpha = attack_cfg.get("alpha")
    steps = attack_cfg["steps"]

    label_lines = []
    saved = 0
    model.eval()

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        # Only attack malware samples (label == 1)
        mask = labels == 1
        if not mask.any():
            continue

        mal_imgs = images[mask]
        mal_labels = labels[mask]

        if attack_type == "fgsm":
            adv_imgs = fgsm_attack(model, mal_imgs, mal_labels, eps)
        else:
            adv_imgs = pgd_attack(model, mal_imgs, mal_labels, eps, alpha, steps)

        for i in range(adv_imgs.size(0)):
            fname = f"adv_malware_{saved:05d}.png"
            tensor_to_png(adv_imgs[i]).save(img_dir / fname)
            label_lines.append(f"{fname} 1\n")
            saved += 1

        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {saved} adversarial images...", end="\r")

    # Write labels file
    with open(output_dir / "labels.txt", "w") as f:
        f.write("# filename  true_label (1 = malware)\n")
        f.writelines(label_lines)

    print(f"\n  Saved {saved} adversarial images -> {img_dir}")
    return saved


def main_generate():
    """Generate the fixed adversarial test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  Generating Fixed MaleX Adversarial Test Set")
    print("  Attacker model : clean 3C2D")
    print(f"  Device         : {device}")
    print(f"  Output root    : {MALEX_ADV_TEST_SET_DIR_STR}")
    print("=" * 65)

    # Load test data
    print("[*] Loading MaleX test split...")
    _, _, test_loader, _ = get_data_loaders(
        data_dir=MALEX_DATASET_DIR_STR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Load clean 3C2D model
    print(f"[*] Loading clean 3C2D: {MALEX_3C2D_CLEAN_MODEL_PATH_STR}")
    if not os.path.exists(MALEX_3C2D_CLEAN_MODEL_PATH_STR):
        raise FileNotFoundError(
            f"Clean 3C2D model not found: {MALEX_3C2D_CLEAN_MODEL_PATH_STR}\n"
            "Run train_3c2d.py first."
        )
    model = MaleX3C2D().to(device)
    model.load_state_dict(
        torch.load(MALEX_3C2D_CLEAN_MODEL_PATH_STR, map_location=device)
    )
    model.eval()
    print(f"[*] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate each attack set
    out_root = Path(MALEX_ADV_TEST_SET_DIR_STR)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for cfg in ATTACK_CONFIGS:
        out_dir = out_root / cfg["name"]
        print(f"\n[*] Generating: {cfg['name']}")
        print(f"    type={cfg['type']}  eps={cfg['eps']}  steps={cfg['steps']}")
        n = generate_fixed_set(model, test_loader, cfg, out_dir, device)
        summary.append((cfg["name"], n))

    # Write README
    readme_path = out_root / "README.txt"
    with open(readme_path, "w") as f:
        f.write("Fixed Adversarial Test Set - MaleX Dataset\n")
        f.write("=========================================\n")
        f.write("Generated from: MaleX test split (malware samples only)\n")
        f.write("Attacker model: clean MaleX3C2D (3c2d_malex_clean_vulnerable.pth)\n")
        f.write("Threat model: attacker causes malware -> classified as benign\n\n")
        f.write("Evaluation protocol:\n")
        f.write("  Load images -> pass through any model -> check if predicted malware\n")
        f.write("  Recall on this set = % of malware still detected after attack\n")
        f.write("  Evasion rate = 100 - Recall\n\n")
        for name, count in summary:
            f.write(f"Folder: {name}/  ({count} adversarial images)\n")

    print(f"\n[*] README written: {readme_path}")
    print("\n" + "=" * 65)
    print("  GENERATION SUMMARY")
    print("=" * 65)
    for name, count in summary:
        print(f"  {name:<35} {count} images")
    print(f"\n  Fixed adversarial test set ready at:\n  {out_root}")
    print("  Run evaluate_attacks_fixed.py to evaluate models against this set.")


def main_verify():
    """Verify the fixed set was created correctly."""
    out_root = Path(MALEX_ADV_TEST_SET_DIR_STR)

    if not out_root.exists():
        print(f"[ERROR] {out_root} does not exist. Run generation first.")
        return

    print(f"Verifying fixed adversarial test set at: {out_root}")
    all_ok = True

    for cfg in ATTACK_CONFIGS:
        folder = out_root / cfg["name"]
        img_dir = folder / "images"
        labels_file = folder / "labels.txt"

        if not img_dir.exists():
            print(f"  [FAIL] Missing images directory: {img_dir}")
            all_ok = False
            continue

        png_files = list(img_dir.glob("*.png"))
        n_images = len(png_files)

        if not labels_file.exists():
            print(f"  [FAIL] Missing labels.txt in: {folder}")
            all_ok = False
            continue

        with open(labels_file) as f:
            label_lines = [l for l in f.readlines() if not l.startswith("#")]
        n_labels = len(label_lines)

        # Spot-check a few images
        errors = 0
        for p in png_files[:5]:
            try:
                img = Image.open(p)
                assert img.size == (256, 256), f"Wrong size: {img.size}"
                assert img.mode == "L", f"Wrong mode: {img.mode}"
            except Exception as e:
                print(f"  [FAIL] Image check failed for {p.name}: {e}")
                errors += 1

        status = "OK" if n_images == n_labels and errors == 0 else "FAIL"
        print(f"  [{status}] {cfg['name']}: {n_images} images, {n_labels} labels, {errors} image errors")
        if status == "FAIL":
            all_ok = False

    if all_ok:
        print("\n[PASS] Fixed adversarial test set is valid and complete.")
    else:
        print("\n[FAIL] Issues detected. Re-run generation.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate or verify fixed MaleX adversarial test set")
    p.add_argument("--verify", action="store_true", help="Verify existing set instead of generating")
    args = p.parse_args()

    if args.verify:
        main_verify()
    else:
        main_generate()
