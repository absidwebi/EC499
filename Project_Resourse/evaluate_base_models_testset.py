"""
Comprehensive evaluation of MaleX base models (3C2D and ResNet-18 pretrained) on the test set.
Outputs: Accuracy, Confusion Matrix, F1 (macro/binary), AUC-ROC, logit mean/std by class.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from config import (
    MALEX_DATASET_DIR_STR,
    MALEX_3C2D_CLEAN_MODEL_PATH_STR,
    RESNET_MALEX_PRETRAINED_CLEAN_PATH_STR,
)
from dataset_loader import get_data_loaders
from models import MaleX3C2D, get_resnet18_pretrained_grayscale

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test set
_, _, test_loader, _ = get_data_loaders(MALEX_DATASET_DIR_STR, batch_size=BATCH_SIZE, num_workers=0)

# Helper: Evaluate a model on test set and compute all metrics
def evaluate_model(model, checkpoint_path, loader, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images).squeeze(1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    # Metrics
    acc = (preds == labels).mean() * 100
    cm = confusion_matrix(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_binary = f1_score(labels, preds, pos_label=1)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float('nan')
    # Logit stats by class
    benign_logits = logits[labels == 0]
    malware_logits = logits[labels == 1]
    logit_stats = {
        'benign_mean': float(np.mean(benign_logits)) if len(benign_logits) else float('nan'),
        'benign_std': float(np.std(benign_logits)) if len(benign_logits) else float('nan'),
        'malware_mean': float(np.mean(malware_logits)) if len(malware_logits) else float('nan'),
        'malware_std': float(np.std(malware_logits)) if len(malware_logits) else float('nan'),
    }
    return {
        'accuracy': float(acc),
        'confusion_matrix': cm.tolist(),
        'f1_macro': float(f1_macro),
        'f1_binary': float(f1_binary),
        'auc': float(auc),
        'logit_stats': logit_stats,
    }

def print_results(name, results):
    print(f"\n=== {name} ===")
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    cm = np.array(results['confusion_matrix'])
    print(f"Confusion Matrix:\n{cm}")
    print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
    print(f"F1 Score (Malware): {results['f1_binary']:.4f}")
    print(f"AUC-ROC: {results['auc']:.4f}")
    print(f"Logit Mean/Std (Benign): {results['logit_stats']['benign_mean']:.2f} ± {results['logit_stats']['benign_std']:.2f}")
    print(f"Logit Mean/Std (Malware): {results['logit_stats']['malware_mean']:.2f} ± {results['logit_stats']['malware_std']:.2f}")

if __name__ == "__main__":
    print("\nEvaluating MaleX3C2D (fixed, clean)...")
    results_3c2d = evaluate_model(MaleX3C2D(), MALEX_3C2D_CLEAN_MODEL_PATH_STR, test_loader, DEVICE)
    print_results("MaleX3C2D (Fixed, Clean)", results_3c2d)

    print("\nEvaluating ResNet-18 (pretrained, clean)...")
    results_resnet = evaluate_model(get_resnet18_pretrained_grayscale(), RESNET_MALEX_PRETRAINED_CLEAN_PATH_STR, test_loader, DEVICE)
    print_results("ResNet-18 (Pretrained, Clean)", results_resnet)

    # Save results for markdown table
    import json
    with open("Project_Resourse/base_model_testset_results.json", "w") as f:
        json.dump({
            "3c2d": results_3c2d,
            "resnet18_pretrained": results_resnet
        }, f, indent=2)
    print("\nResults saved to Project_Resourse/base_model_testset_results.json")
