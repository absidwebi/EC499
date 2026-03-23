# Stage 2: CNN Architecture & Robustness Strategy Report

Based on a thorough review of the provided reference documents (`CNN_Architecture_for_Robustness.md` and `Malware_Detection_CNN_Architecture_Recommendations.md`), this report synthesizes the architectural decisions and strict training guidelines required for Stage 2: Model Development.

---

## 1. Architectural Strategy: The Two-Tiered Approach
Our project's ultimate goal is **Adversarial Robustness**. Adversarial training (e.g., generating FGSM/PGD attacks for every batch) is immensely computationally expensive. To balance speed, stability, and state-of-the-art performance, we will adopt the recommended **Two-Tiered Architecture Strategy**.

### Tier 1: Prototyping Baseline (Custom Lightweight CNN)
*   **Structure:** A 4-block custom CNN (Conv2D -> BatchNorm -> ReLU -> MaxPool2D) followed by a fully connected layer with Dropout.
*   **Why I Agree:** This is crucial. A lightweight CNN allows us to prototype the highly complex attack and defense algorithms in Stage 3 rapidly. If there is a bug in the PGD implementation, debugging a model that trains in minutes is vastly superior to debugging ResNet, which takes hours per epoch. Furthermore, its smooth gradients (aided by BatchNorm) make analyzing adversarial texturing straightforward.

### Tier 2: The Final Evaluator (Modified ResNet-18)
*   **Structure:** Standard ResNet-18 with the initial convolutional layer modified to accept 1-channel (grayscale) inputs instead of 3-channel (RGB).
*   **Why I Agree:** ResNet is the undisputed academic standard for benchmarking adversarial defenses (famously used by Madry et al.). The skip connections prevent vanishing gradients during adversarial training. While DenseNet provides marginal accuracy bumps, its heavy memory footprint makes adversarial batch generation prohibitively slow. ViT (Vision Transformers) are inappropriate here due to texture reliance and sample size constraints.

---

## 2. Critical Training Rules & Constraints

The reference documents highlight specific rules unique to malware image classification. I fully agree with these constraints and we must strictly enforce them in our PyTorch scripts:

1.  **NO Geometric Image Augmentations:**
    *   **Rule:** We must disable Random Rotation, Random Cropping, and Random Flipping in our `train_transform`.
    *   **Rationale:** Standard augmentations destroy the structural integrity of a PE file representation. Flipping a malware image changes the byte order, which effectively turns it into meaningless noise rather than an executable structure.
    *   **Permitted Data Changes:** We will perfectly restrict preprocessing to `ToTensor()` (scaling [0, 255] to [0.0, 1.0]) and calculated `Class Weights` to handle dataset imbalances.
2.  **Kernel Sizing:**
    *   Relying on $3 \times 3$ kernels (used in both our Custom CNN and ResNet protocols) is optimal for capturing byte-level sequence patterns.

---

## 3. The Stage 2 to Stage 3 Roadmap

To properly set up the thesis and prove the value of adversarial defense, we must follow this sequence:

1.  **Train the "Vulnerable" Baseline:**
    *   Train the Custom CNN on the clean dataset (the 24,000 merged images).
    *   Use Binary Cross-Entropy (BCEWithLogitsLoss), AdamW optimizer ($LR=1e-3$).
    *   **Goal:** Achieve clean test accuracy > 95%.
    *   **Save:** Export the weights (e.g., `custom_cnn_clean_vulnerable.pth`).
2.  **Initial Attack Phase (Stage 3 Prelude):**
    *   Subject the saved "clean" model to FGSM and PGD attacks.
    *   **Goal:** Demonstrate the catastrophic drop in accuracy (e.g., 95% down to 10%), defining the exact vulnerability we intend to fix.
3.  **Adversarial Training (The Defense):**
    *   Retrain using Adversarial Training (injecting adversarial examples into the training batches).
    *   **Goal:** Restore robust accuracy to acceptable levels (e.g., ~70-80%).
4.  **Repeat for ResNet-18:**
    *   Once the pipeline is perfected, benchmark the entire sequence using ResNet-18 for the final thesis metrics.

---

## Conclusion
The provided documentation is scientifically sound and perfectly aligned with adversarial machine learning best practices. 

**Our immediate next step should be implementing the PyTorch class definition for the `Custom Lightweight CNN` and designing the `train.py` loop according to the constraints listed in Section 2.**
