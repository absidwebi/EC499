# Project Proposal

<!-- Page 1 -->

Department of Computer Engineering
Faculty of Engineering
University of Tripoli
Project Proposal Title
Adversarial Robustness in Deep Learning-Based Malware Detection
Student’s Name:Abdulsalam Ashraf Aldwebi
Student’s ID:2210245306
Student’s Email:a.aldwebi@uot.edu.ly
Supervisor’s Name:Dr. Suad Elgeder
Term:Fall 2025
1


<!-- Page 2 -->

1. Problem Statement
Modern cybersecurity systems increasingly rely on deep learning models for malware detection due to their ability
to identify complex patterns in software files. However, these models are vulnerable to adversarial attacks—subtle,
imperceptible modifications to malicious files that cause misclassification while preserving malicious functional-
ity. Such attacks, particularly those based on gradient-based optimization, enable malware to evade detection by
manipulating input features in a way that is invisible to human analysts but highly effective against machine learn-
ing models. This vulnerability is particularly critical as attackers actively develop evasion techniques to bypass
commercial antivirus solutions. This project addresses this challenge by investigating methods to enhance the
robustness of deep learning-based malware detectors against such evasion attacks, ensuring more trustworthy and
resilient cybersecurity solutions through a complete attack-and-defense pipeline.
2. Objective
The primary objective is to develop and evaluate a framework for building adversarially robust malware detection
models using static analysis. Specific aims include:
• Exploring the representation of Portable Executable (PE) files as grayscale images for static analysis.
• Investigating gradient-based adversarial attack methods to evaluate model vulnerability.
• Evaluating adversarial training as a defense mechanism to improve resilience.
• Assessing model performance using standard metrics (clean accuracy, robust accuracy, evasion rate).
This work aims to establish a reproducible pipeline for adversarial robustness in malware detection, balancing
detection performance with resistance to evasion, while operating in a secure, isolated environment.
3. Procedure
The project will be executed in four distinct phases:
Dataset Preparation:Construct a binary dataset comprising malware and benign PE files represented as grayscale
images. Publicly available datasets (e.g., Malimg) will be used for historical malware samples. Benign sam-
ples will be collected from the student’s own Windows System32 directory and open-source software. All
files will be converted to images using standardized byte-to-image methodology in a development environ-
ment with network access.
Model Development and Training:Implement a convolutional neural network (CNN) for binary malware clas-
sification. The model will be trained on the prepared dataset using available computational resources to
accelerate the process.
Attack and Defense Evaluation:Generate adversarial examples using white-box gradient-based methods to as-
sess vulnerability. The model will then be retrained using adversarial training to enhance robustness. All
development and training will occur in the network-enabled development environment.
Secure Deployment and Testing:The final trained model will be deployed in a separate, isolated environment
with no network access. A secure inference API prototype will serve as the interface for this deployment,
accepting PE file uploads, performing static analysis (PE validation and image conversion), and returning
predictions. This phase ensures safety during real-world testing while demonstrating the system’s practical
applicability.
2


<!-- Page 3 -->

4. Resources and Tools
•Software & Libraries:
–Programming Language: Python
–Deep Learning Frameworks: PyTorch
–Adversarial Attack Libraries: Torchattacks
–PE File Analysis:pefilelibrary
–Data Processing: NumPy, Pandas, Pillow
–Visualization and Reporting: Matplotlib, LaTeX (Overleaf)
•Datasets:
–Public malware image datasets (e.g., Malimg)
–Self-collected benign PE files from the student’s licensed Windows installation and open-source soft-
ware
–Optional: Recent malware samples from trusted repositories (e.g., MalwareBazaar) for generalization
testing.
•Other Tools:
–Git/GitHub for version control
–Google Colab or local compute resources for model training
–Tools for creating isolated development and deployment environments
–Framework for a secure inference API prototype
5. Keywords
Adversarial Attacks, Malware Detection, Deep Learning, Convolutional Neural Networks (CNN), Adversarial
Training, PE Files, Image-Based Analysis, Cybersecurity, Secure Inference.
3


<!-- Page 4 -->

I/We understand that all work on my Graduation Project must be my original work. We will clearly cite all re-
sources used in accordance with Department of Computer Engineering guidelines. Any plagiarism will invalidate
my project and jeopardize my graduation credit.
We understand that at the completion of the graduation project, we are required to submit all deliverables and
outputs of the projects (Software, Hardware and data used and produced by the project; source codes) to our
supervisors.
The Department of Computer Engineering is the rightful owner of copyright and all intellectual property rights of
all student’s work.
Student’s Signature: Date:
As the Advisor for this Graduation Project:
I accept the proposal as written.
I do not accept the proposal because
Advisor’s Signature: Date:
4


<!-- Page 5 -->

Committee Decision
Project Examiners
1.
2.
Project Director’s Signature: Date:
5
