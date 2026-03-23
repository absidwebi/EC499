File 1: Adversarial Robustness Presentation
Title Slide

University of Tripoli Department of Computer Engineering 



Adversarial Robustness in Deep Learning-Based Malware Detection 


Student: Abdulsalam Ashraf Aldwebi (ID: 2210245306) 


Supervisor: Dr. Suad Elgeder 


Term: Fall 2025 

Presentation Outline
An overview of the sections covered in this project proposal. 

01 Introduction & Motivation 

02 Problem Statement 

03 Project Objectives 

04 Methodology 

05 Evaluation Metrics 

06 Key References 

07 Conclusion 

Introduction & Motivation

Why Trust AI in Cybersecurity? 


The Core Problem: Modern cybersecurity relies heavily on AI, but models are vulnerable to adversarial attacks—subtle, invisible changes to files that cause misclassification as benign. 


The Threat Vector: These attacks exploit the mathematical fragility of deep learning, allowing attackers to evade detection without altering the malware's harmful behavior. 

This is a critical, real-world compromise. 

Our Solution: We develop a robust, image-based malware detector. By converting PE files to grayscale images and using Adversarial Training, we aim for a model that is not just accurate, but demonstrably trustworthy. 

Problem Statement

Adversarial Vulnerability  Deep Learning models are fragile. Attackers use gradient-based optimization to subtly modify malware files. 


These modifications are: 

Invisible to human analysts. 

Preserve malicious functionality. 

Often cause misclassification (Malware → Benign). 


Visualizing the Attack 

Original Malware (Detected) + Noise → Adversarial Malware (Evaded!) 

Project Objectives

Main Goal: Develop a robust malware detection framework using static analysis. 


Representation: Convert Portable Executable (PE) files into grayscale images for CNN processing. 


Vulnerability Check: Investigate gradient-based attacks (FGSM, PGD) to evaluate model susceptibility. 


Defense Strategy: Apply Adversarial Training to harden the model against attacks. 



Evaluation: Measure Clean Accuracy vs. Robust Accuracy and Evasion Rates. 


Methodology Overview
A comprehensive four-phase pipeline. 


Phase 1: Dataset Prep 


Phase 2: Model Development 


Phase 3: Attack & Defense 



Phase 4: Secure Deployment 


Phase 1: Dataset Preparation 


Data Sources 


Malware: Malimg Public Dataset (Various families). 


Benign: Open Source software. 


Preprocessing: Byte-to-Image 

Raw binaries are read as 8-bit integers and reshaped into grayscale images. This allows CNNs to "see" code structure. 


Dataset Composition 

Malware (50%) 

Benign (50%) 

Phase 2: Model Development 


CNN Architecture We utilize a Convolutional Neural Network (CNN) tailored for image classification. 



Framework: PyTorch 


Input: Resized grayscale PE images (e.g., 256x256). 


Simplified Architecture Flow: Input → Conv2D → Pool → Dense → Output 


Phase 3: Adversarial Attacks 


The Attack Vector 

We simulate white-box attacks where the attacker knows the model architecture. 


Goal: Cause misclassification while preserving malicious payload functionality. 


Method: Gradient-based optimization. 


Tool: Torchattacks library. 


Risk: Standard models exhibit 80-90% evasion rates under attack. 

Defense Strategy 


Adversarial Training The most effective defense is to train the model on the attacks themselves. 


Generate adversarial examples during training. 

Inject them into the batch with correct labels. 

Model learns to ignore the adversarial noise. 


Result: Improved Robust Accuracy. 


Active Defense Loop Training set is dynamically expanded with adversarial samples at every epoch.

Phase 4: Secure DeploymentIsolated ArchitectureTo ensure safety, the final model is deployed in a strictly isolated environment.

Isolation: No internet access prevents potential malware leakage.

Secure API: A restricted interface allows file upload and returns prediction results only.

Deployment FlowUser $\rightarrow$ Secure API $\rightarrow$ Isolated Environment

Evaluation Metrics

Performance Measures
Clean Accuracy: Performance on unmodified malware/benign samples.
Robust Accuracy: Accuracy on adversarially perturbed (attacked) samples.Evasion Rate: Rate at which adversarial samples successfully bypass detection.
Impact of Defense Training(Visual comparison of model performance)Standard Model: High Clean Accuracy vs. Low Robust Accuracy.
Defended Model: High Clean Accuracy vs. Improved Robust Accuracy.

Expected Outcomes
Increased Robustness: Significant improvement in model resilience against gradient-based attacks.

Reproducible Pipeline: A standardized research workflow facilitating further studies in adversarial ML.

Secure Prototype: A functional prototype for secure static malware analysis deployment.
 
 
Key References
Malware Images: Visualization and Automatic Classification (2011) Nataraj, L., et al. (2011). ACM International Workshop on Security and Privacy.

Towards Deep Learning Models Resistant to Adversarial Attacks (2018) Madry, A., et al. (2018). International Conference on Learning Representations (ICLR).

Explaining and Harnessing Adversarial Examples (2015) Goodfellow, I. J., et al. (2015). International Conference on Learning Representations (ICLR).

EMBER: An Open Dataset for Training Static PE Malware ML Models (2018) Anderson, H., & Roth, P. (2018). arXiv preprint arXiv:1804.04637.

Malimg Dataset (2017) Larsen, B. (2017). Kaggle.

Adversarial Robustness of Vision-Based Malware Detectors (2022) Ahmadian, S., et al. (2022). Computers & Security, 123, 103009.

MalwareBazaar (2025) abuse.ch. (2025). MalwareBazaar.

Secure ML Model Serving with Docker and Flask (2023) Zhang, Y., et al. (2023). IEEE Security & Privacy Workshops.

Conclusion
Adversarial robustness is not optional—it is essential for trustworthy AI in cybersecurity.

This project bridges the gap between theoretical adversarial machine learning and practical malware detection by delivering a reproducible, end-to-end pipeline that:

Validates the vulnerability of image-based detectors to gradient-based attacks.

Demonstrates quantifiable robustness gains through adversarial training.

Implements safety deployment via static analysis and an isolated environment.