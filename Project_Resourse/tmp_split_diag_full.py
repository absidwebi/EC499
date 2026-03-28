import os, sys, math, random, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

sys.path.insert(0, "/home/alucard-00/EC499/Project_Resourse")
from config import MALIMG_ARCHIVE_DIR_STR, RESNET_CLEAN_MODEL_PATH_STR, LOGS_DIR
from dataset_loader import get_data_loaders
from models import get_resnet18_grayscale

SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 0
MAX_BATCHES = None
SAL_N_PER_CLASS = 6
SAVE_PREFIX = "v3_diag_resnet18"

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

train_loader, val_loader, test_loader, _ = get_data_loaders(
    MALIMG_ARCHIVE_DIR_STR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)

model = get_resnet18_grayscale().to(device)
model.load_state_dict(torch.load(RESNET_CLEAN_MODEL_PATH_STR, map_location=device))
model.eval()
print(f"loaded_model={RESNET_CLEAN_MODEL_PATH_STR}")


def summarize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def split_stats(name, loader):
    sums = {0: 0.0, 1: 0.0}
    sums2 = {0: 0.0, 1: 0.0}
    npx = {0: 0, 1: 0}
    img_means = {0: [], 1: []}
    img_vars = {0: [], 1: []}
    full_pad_rows = {0: [], 1: []}
    full_pad_cols = {0: [], 1: []}

    for bi, (imgs, lbls) in enumerate(loader):
        if MAX_BATCHES is not None and bi >= MAX_BATCHES:
            break
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        flat = imgs.view(imgs.size(0), -1)
        means = flat.mean(dim=1)
        vars_ = flat.var(dim=1, unbiased=False)

        x = imgs.squeeze(1)
        is_neg1 = (x == -1.0)
        pad_rows = is_neg1.all(dim=2).sum(dim=1).detach().cpu().numpy()
        pad_cols = is_neg1.all(dim=1).sum(dim=1).detach().cpu().numpy()

        for cls in (0, 1):
            mask = (lbls == cls)
            if not mask.any():
                continue
            sel = imgs[mask]
            sums[cls] += sel.sum().item()
            sums2[cls] += (sel * sel).sum().item()
            npx[cls] += sel.numel()
            img_means[cls].extend(means[mask].detach().cpu().numpy().tolist())
            img_vars[cls].extend(vars_[mask].detach().cpu().numpy().tolist())
            mnp = mask.detach().cpu().numpy()
            full_pad_rows[cls].extend(pad_rows[mnp].tolist())
            full_pad_cols[cls].extend(pad_cols[mnp].tolist())

    print(f"\n== {name.upper()} stats (normalized tensors) ==")
    for cls, cname in [(0, "benign"), (1, "malware")]:
        mean = sums[cls] / max(1, npx[cls])
        var = (sums2[cls] / max(1, npx[cls])) - (mean * mean)
        std = math.sqrt(max(var, 0.0))
        print(f"{cname}: n_img={len(img_means[cls])}")
        print(f"  pixel_mean_norm={mean:.4f} pixel_std_norm={std:.4f}")
        print(f"  per_image_mean_norm={np.mean(img_means[cls]):.4f} per_image_var_norm={np.mean(img_vars[cls]):.4f}")
        print(f"  pad_rows_all_-1: mean={np.mean(full_pad_rows[cls]):.2f} p95={np.quantile(full_pad_rows[cls],0.95):.2f}")
        print(f"  pad_cols_all_-1: mean={np.mean(full_pad_cols[cls]):.2f} p95={np.quantile(full_pad_cols[cls],0.95):.2f}")

    gap = abs(np.mean(img_means[0]) - np.mean(img_means[1]))
    print(f"per_image_mean_gap_norm={gap:.4f}")


def logits_and_metrics(name, loader, save_tag):
    from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score, roc_curve

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for bi, (imgs, lbls) in enumerate(loader):
            if MAX_BATCHES is not None and bi >= MAX_BATCHES:
                break
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1).detach().cpu()
            all_logits.append(logits)
            all_labels.append(lbls)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    pred = (logits > 0).astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits))
    cm = confusion_matrix(labels, pred, labels=[0, 1])

    print(f"\n== {name.upper()} metrics ==")
    print("confusion_matrix rows=true [benign,malware], cols=pred [benign,malware]")
    print(cm)

    prec, rec, f1, sup = precision_recall_fscore_support(labels, pred, labels=[0, 1], zero_division=0)
    acc = (pred == labels).mean()
    print(f"accuracy={acc:.4f}")
    print(f"benign : precision={prec[0]:.4f} recall={rec[0]:.4f} f1={f1[0]:.4f} support={sup[0]}")
    print(f"malware: precision={prec[1]:.4f} recall={rec[1]:.4f} f1={f1[1]:.4f} support={sup[1]}")

    print("\nclassification_report:")
    print(classification_report(labels, pred, target_names=["benign", "malware"], digits=4, zero_division=0))

    auc = roc_auc_score(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    idx = int(np.argmin(np.abs(fpr - 0.01)))
    print(f"roc_auc={auc:.4f}")
    print(f"tpr_at_fpr_1pct={tpr[idx]:.4f}")

    b = logits[labels == 0]
    m = logits[labels == 1]
    sb = summarize(b)
    sm = summarize(m)
    gap = abs(sb["mean"] - sm["mean"])
    print("\nlogit_summary_benign:", sb)
    print("logit_summary_malware:", sm)
    print(f"logit_gap_mean={gap:.4f}")

    out_hist = Path(LOGS_DIR) / f"{SAVE_PREFIX}_logit_hist_{save_tag}.png"
    plt.figure(figsize=(10, 5))
    plt.hist(b, bins=60, alpha=0.6, density=True, label="benign")
    plt.hist(m, bins=60, alpha=0.6, density=True, label="malware")
    plt.axvline(0, color="black", linestyle="--", label="threshold=0")
    plt.title(f"Logit Distribution ({name}) gap={gap:.2f}")
    plt.xlabel("logit")
    plt.ylabel("density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_hist, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"logit_hist_saved={out_hist}")

    out_cm = Path(LOGS_DIR) / f"{SAVE_PREFIX}_confusion_{save_tag}.png"
    plt.figure(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pred_benign", "pred_malware"],
                yticklabels=["true_benign", "true_malware"])
    plt.title(f"Confusion Matrix ({name})")
    plt.tight_layout()
    plt.savefig(out_cm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"confusion_png_saved={out_cm}")


def save_saliency_grid(loader, out_path):
    benign_img = []
    malware_img = []
    for imgs, lbls in loader:
        for i in range(imgs.size(0)):
            if lbls[i].item() == 0 and len(benign_img) < SAL_N_PER_CLASS:
                benign_img.append(imgs[i:i+1])
            if lbls[i].item() == 1 and len(malware_img) < SAL_N_PER_CLASS:
                malware_img.append(imgs[i:i+1])
            if len(benign_img) >= SAL_N_PER_CLASS and len(malware_img) >= SAL_N_PER_CLASS:
                break
        if len(benign_img) >= SAL_N_PER_CLASS and len(malware_img) >= SAL_N_PER_CLASS:
            break

    def saliency(x):
        x = x.clone().detach().to(device).requires_grad_(True)
        out = model(x).sum()
        model.zero_grad(set_to_none=True)
        out.backward()
        g = x.grad.detach().abs().squeeze(0).squeeze(0)
        return g

    b_imgs = [((x * 0.5) + 0.5).squeeze(0).squeeze(0).cpu().numpy() for x in benign_img]
    m_imgs = [((x * 0.5) + 0.5).squeeze(0).squeeze(0).cpu().numpy() for x in malware_img]
    b_sal = [saliency(x).cpu().numpy() for x in benign_img]
    m_sal = [saliency(x).cpu().numpy() for x in malware_img]

    def norm01(a):
        a = a - a.min()
        d = (a.max() - a.min()) or 1.0
        return a / d

    cols = SAL_N_PER_CLASS
    fig, axes = plt.subplots(4, cols, figsize=(3.2 * cols, 10))
    fig.suptitle("Input-Gradient Saliency (ResNet-18 clean)", fontsize=14)
    for j in range(cols):
        axes[0][j].imshow(b_imgs[j], cmap="gray")
        axes[0][j].axis("off")
        axes[0][j].set_title(f"Benign #{j+1}", fontsize=10)
        axes[1][j].imshow(norm01(b_sal[j]), cmap="hot")
        axes[1][j].axis("off")
        axes[2][j].imshow(m_imgs[j], cmap="gray")
        axes[2][j].axis("off")
        axes[2][j].set_title(f"Malware #{j+1}", fontsize=10)
        axes[3][j].imshow(norm01(m_sal[j]), cmap="hot")
        axes[3][j].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saliency_saved={out_path}")


split_stats("train", train_loader)
split_stats("val", val_loader)
split_stats("test", test_loader)

logits_and_metrics("test", test_loader, "test")
logits_and_metrics("train", train_loader, "train")
logits_and_metrics("val", val_loader, "val")

out_sal = Path(LOGS_DIR) / f"{SAVE_PREFIX}_saliency_test.png"
save_saliency_grid(test_loader, out_sal)

print("\nDONE")
