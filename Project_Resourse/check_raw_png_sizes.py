import argparse
import os
from collections import Counter

from PIL import Image


def iter_png_paths(root):
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".png"):
                yield os.path.join(r, f)


def main(root, max_per_class=None):
    # root is expected to be .../archive/malimg_dataset/{train,val,test}
    counts = {
        "benign": Counter(),
        "malware": Counter(),
    }
    totals = {
        "benign": 0,
        "malware": 0,
    }

    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        label = "benign" if class_name == "benign" else "malware"

        for pi, p in enumerate(iter_png_paths(class_dir)):
            if max_per_class is not None and pi >= max_per_class:
                break
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            counts[label][(w, h)] += 1
            totals[label] += 1

    print(f"root={root}")
    for label in ["benign", "malware"]:
        print(f"\n== {label} ==")
        print(f"n={totals[label]}")
        for (w, h), c in counts[label].most_common(15):
            print(f"  {w}x{h}: {c}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Check raw PNG size distribution")
    p.add_argument("root", help="Split dir, e.g. Project_Resourse/archive/malimg_dataset/train")
    p.add_argument("--max-per-class", type=int, default=None)
    args = p.parse_args()
    main(args.root, max_per_class=args.max_per_class)
