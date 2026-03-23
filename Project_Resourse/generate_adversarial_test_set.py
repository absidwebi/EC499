"""
generate_adversarial_test_set.py
=================================
Generates a fixed adversarial test set by applying FGSM and PGD attacks
to the MALWARE PORTION of the test set using the clean vulnerable model,
then saving the resulting adversarial images as PNG files to disk.

Scientific purpose:
    Both the clean and adversarially trained model can then be evaluated
    against the IDENTICAL set of adversarial pixel values, eliminating
    stochastic variation in attack strength between evaluation runs.
    This is the gold-standard evaluation protocol for comparing defences.

Threat model:
    Only malware images are attacked. The adversary's goal is to make
    malware evade detection (malware → classified as benign).
    Attacking benign images has no real-world motivation.

Output structure:
    adversarial_test_set/
    ├── fgsm_eps0.05/
    │   ├── images/         ← adversarial PNG files (one per malware sample)
    │   └── labels.txt      ← ground truth labels (all 1 = malware)
    ├── pgd_eps0.05_steps40/
    │   ├── images/
    │   └── labels.txt
    └── README.txt
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from config import MALIMG_ARCHIVE_DIR_STR, RESNET_CLEAN_MODEL_PATH_STR, PROJECT_ROOT
from models import get_resnet18_grayscale
from dataset_loader import get_data_loaders

# === OUTPUT CONFIGURATION ===
ADV_SET_ROOT = PROJECT_ROOT / "adversarial_test_set"

ATTACK_CONFIGS = [
    {
        "name":    "fgsm_eps0.05",
        "type":    "fgsm",
        "eps":     0.05,
        "alpha":   None,
        "steps":   1,
    },
    {
        "name":    "pgd_eps0.05_steps40",
        "type":    "pgd",
        "eps":     0.05,
        "alpha":   0.01,
        "steps":   40,
    },
]
# ============================

criterion = nn.BCEWithLogitsLoss()


def fgsm(model, images, labels, eps):
    images = images.clone().detach().requires_grad_(True)
    loss   = criterion(model(images), labels.float().unsqueeze(1))
    model.zero_grad()
    loss.backward()
    return torch.clamp(images + eps * images.grad.sign(), -1.0, 1.0).detach()


def pgd(model, images, labels, eps, alpha, steps):
    labels_f   = labels.float().unsqueeze(1)
    adv        = images.clone().detach()
    adv        = torch.clamp(adv + torch.empty_like(adv).uniform_(-eps, eps), -1.0, 1.0)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = criterion(model(adv), labels_f)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv = adv + alpha * adv.grad.sign()
            adv = torch.clamp(images + torch.clamp(adv - images, -eps, eps), -1.0, 1.0).detach()
    return adv


def tensor_to_png(tensor_img):
    """Convert normalised [-1,1] tensor to uint8 PIL image."""
    arr = tensor_img.squeeze(0).cpu().numpy()   # [H, W]
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L')


def generate_set(model, test_loader, attack_cfg, output_dir, device):
    """Generate and save adversarial images for one attack configuration."""
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    attack_type = attack_cfg["type"]
    eps         = attack_cfg["eps"]
    alpha       = attack_cfg.get("alpha")
    steps       = attack_cfg["steps"]

    label_lines   = []
    saved_count   = 0
    skipped_count = 0

    model.eval()

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        # Only attack malware samples (label == 1)
        malware_mask = labels == 1
        if malware_mask.sum() == 0:
            continue

        malware_imgs   = images[malware_mask]
        malware_labels = labels[malware_mask]

        # Generate adversarial examples
        if attack_type == "fgsm":
            adv_imgs = fgsm(model, malware_imgs, malware_labels, eps)
        else:
            adv_imgs = pgd(model, malware_imgs, malware_labels, eps, alpha, steps)

        # Save each adversarial image as PNG
        for i in range(adv_imgs.size(0)):
            img_idx  = saved_count
            filename = f"adv_malware_{img_idx:05d}.png"
            pil_img  = tensor_to_png(adv_imgs[i])
            pil_img.save(img_dir / filename)
            label_lines.append(f"{filename} 1\n")
            saved_count += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {saved_count} malware adversarial images...", end='\r')

    # Write labels file
    with open(output_dir / "labels.txt", "w") as f:
        f.write("# filename  true_label\n")
        f.writelines(label_lines)

    print(f"\n  Saved {saved_count} adversarial images to: {img_dir}")
    return saved_count


def evaluate_on_fixed_set(model, adv_dir, device, model_name):
    """Evaluate a model on the pre-generated fixed adversarial set."""
    img_dir = adv_dir / "images"
    files   = sorted(img_dir.glob("*.png"))

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    model.eval()
    correct = total = 0

    with torch.no_grad():
        for fpath in files:
            img  = transform(Image.open(fpath)).unsqueeze(0).to(device)
            pred = (model(img).squeeze() > 0).long().item()
            # All adversarial images are malware (label=1)
            # We want to know how many are still correctly predicted as malware
            if pred == 1:
                correct += 1
            total += 1

    recall = 100.0 * correct / total if total > 0 else 0
    evasion = 100.0 - recall
    print(f"  {model_name:<30} Recall: {recall:.2f}%  Evasion: {evasion:.2f}%  ({correct}/{total})")
    return recall, evasion


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  Generating Fixed Adversarial Test Set")
    print(f"  Using clean vulnerable model as the attacker")
    print(f"  Output root: {ADV_SET_ROOT}")
    print("=" * 65)

    # Load test set
    _, _, test_loader, _ = get_data_loaders(
        data_dir=MALIMG_ARCHIVE_DIR_STR, batch_size=32, num_workers=0
    )

    # Load the clean vulnerable model to generate attacks
    print(f"\n[*] Loading clean vulnerable model: {RESNET_CLEAN_MODEL_PATH_STR}")
    clean_model = get_resnet18_grayscale().to(device)
    clean_model.load_state_dict(torch.load(RESNET_CLEAN_MODEL_PATH_STR, map_location=device))
    clean_model.eval()

    # Generate adversarial sets
    for cfg in ATTACK_CONFIGS:
        out_dir = ADV_SET_ROOT / cfg["name"]
        print(f"\n[*] Generating: {cfg['name']}")
        print(f"    eps={cfg['eps']}  steps={cfg['steps']}")
        n = generate_set(clean_model, test_loader, cfg, out_dir, device)
        print(f"    Total adversarial images saved: {n}")

    # Write README
    readme = ADV_SET_ROOT / "README.txt"
    with open(readme, "w") as f:
        f.write("Fixed Adversarial Test Set\n")
        f.write("==========================\n")
        f.write("Generated using the clean vulnerable ResNet-18 model.\n")
        f.write("Contains only MALWARE test samples (label=1) after adversarial perturbation.\n")
        f.write("Threat model: attacker wants malware to evade detection (malware -> benign).\n\n")
        f.write("Evaluation: load each image, pass through model, check if still predicted as malware.\n")
        f.write("Recall on this set = percentage of malware still detected after attack.\n")
        f.write("Evasion rate = 100 - Recall.\n\n")
        for cfg in ATTACK_CONFIGS:
            f.write(f"Folder: {cfg['name']}/\n")
            f.write(f"  Attack: {cfg['type'].upper()}  eps={cfg['eps']}  steps={cfg['steps']}\n\n")

    print(f"\n[*] README written to: {readme}")

    # Now evaluate both models on each fixed set
    from config import RESNET_ADV_TRAINED_MODEL_PATH_STR
    at_model = get_resnet18_grayscale().to(device)
    at_model.load_state_dict(
        torch.load(RESNET_ADV_TRAINED_MODEL_PATH_STR, map_location=device)
    )
    at_model.eval()

    print("\n" + "=" * 65)
    print("  EVALUATION ON FIXED ADVERSARIAL TEST SET")
    print("=" * 65)

    for cfg in ATTACK_CONFIGS:
        adv_dir = ADV_SET_ROOT / cfg["name"]
        print(f"\n[Attack: {cfg['name']}]")
        evaluate_on_fixed_set(clean_model, adv_dir, device, "Clean ResNet-18 (Vulnerable)")
        evaluate_on_fixed_set(at_model,    adv_dir, device, "ResNet-18 (Adversarially Trained)")

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == "__main__":
    main()
