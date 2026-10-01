# Tattoo & Street Art Visual Classifier

A transfer-learning computer vision system categorizing body ink, outdoor murals, and background clutter.

- **Author:** Jonathan S. Aldana
- **Course:** ITAI 1378 - Midterm Blueprint Proposal
- **Institution:** Houston City College
- **Tier:** Tier 1: Core (Single-Task CNN)

---

## 1. Problem Statement
In disaster victim identification (DVI), missing persons registries, and forensic intake, automated photo sorting frequently conflates street murals, wall graffiti, and patterned skin with actual tattoos. This project delivers an automated visual classifier to triage permanent body ink markers from background noise and urban murals.

## 2. Solution Pipeline
- **Input:** Raw field imagery / user upload resized to 224x224 RGB.
- **Backbone:** ResNet50 (pre-trained on ImageNet) for feature extraction.
- **Classification Head:** Linear layer (2048 -> 3) with Softmax activation.
- **Output:** Multi-class classification (`tats`, `street_art`, `none`) with confidence scores.

## 3. Technical Approach
- **Framework:** PyTorch & Torchvision
- **Loss Function:** Categorical Cross-Entropy Loss
- **Optimizer:** Adam / AdamW with early stopping (patience = 5)
- **Hardware:** Google Colab T4 GPU (free tier)

## 4. Success Metrics
- **Primary Metric:** Overall Multi-Class Top-1 Accuracy >= 80%
- **Secondary Metric:** Macro F1-Score >= 0.78 (Tattoo Recall >= 85%)
- **Latency Budget:** < 200 ms per image on standard CPU instances

## 5. Milestone Schedule
- **Week 10 (Blueprint):** Midterm pitch and repository proposal submitted.
- **Week 11 (First Working Demo):** End-to-end inference verified on sample batches.
- **Weeks 12–13 (Make It Yours):** Model trained on 2,661-image dataset.
- **Week 14 (Improve & Measure):** Hyperparameter tuning, metrics, and error analysis.
- **Week 15 (Package & Present):** Final model export (< 100 MB), demo video, and documentation.

## 6. Risks & Resources
- **Domain Overfitting:** Mitigated with affine augmentations, color jitter, and deep layer fine-tuning.
- **Class Imbalance:** Mitigated via class-weighted loss and oversampling.
- **Compute:** Google Colab & Kaggle GPU notebooks ($0.00 estimated budget).
