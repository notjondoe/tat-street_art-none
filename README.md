# Tattoo & Street Art Visual Classifier

A transfer-learning computer vision system categorizing body ink, outdoor murals, and background clutter.

- **Author:** Jonathan S. Aldana
- **Course:** ITAI 1378 - Midterm Blueprint Proposal
- **Institution:** Houston City College
- **Tier:** Tier 1: Core (Single-Task CNN) — Selected because the objective focuses on a single three-class image classification task using transfer learning to reliably triage human markings from environmental noise.

---

## 1. Problem Statement
In disaster victim identification (DVI), missing persons registries, and forensic intake, automated photo sorting frequently conflates street murals, wall graffiti, and patterned skin with actual tattoos. Reviewing thousands of unorganized images manually creates critical bottlenecks during recovery operations. This project delivers an automated visual classifier to triage permanent body ink markers from background noise and urban murals.

## 2. Solution Pipeline
- **Input:** Raw field imagery resized and normalized to 224x224 RGB tensors.
- **Backbone:** ResNet50 (pre-trained on ImageNet) with frozen convolutional layers to extract edge and dermis texture features.
- **Classification Head:** Trainable linear layer (2048 -> 3) with Softmax activation.
- **Output:** Categorical predictions (`tats`, `street_art`, `none`) with associated confidence scores.

## 3. Technical Approach
- **CV Technique:** Multi-Class Image Classification
- **Model Architecture:** ResNet50 Backbone + Custom Linear Output Layer
- **Framework & Libraries:** PyTorch, Torchvision, NumPy, Matplotlib
- **Hardware:** Google Colab free-tier NVIDIA T4 GPU (16GB VRAM)

## 4. Data Plan
- **Total Volume:** 2,661 images
- **Classes & Labels:**
  - `none` (1,164 images): Bare skin, textured clothing, billboards, street signs, urban noise
  - `tats` (802 images): Permanent body ink, linework, and dermis pigmentation
  - `street_art` (695 images): Outdoor murals, graffiti, and spray paint on walls
- **Data Sources:** Curated public subsets from Roboflow Universe and original author-captured imagery.
- **Dataset Splits (80 / 10 / 10):**
  - Train: 2,128 images (931 `none`, 641 `tats`, 556 `street_art`)
  - Validation: 265 images (116 `none`, 80 `tats`, 69 `street_art`)
  - Test: 268 images (117 `none`, 81 `tats`, 70 `street_art`)

## 5. Success Metrics
- **Primary Target:** Overall Multi-Class Top-1 Accuracy >= 80% (>= 75% per-class minimum threshold)
- **Secondary Target:** Macro F1-Score >= 0.78; Tattoo Recall >= 85% to minimize missed identification markers
- **Latency & Footprint:** Inference under 200 ms on CPU (< 25 ms on GPU); quantized model size < 100 MB

## 6. Milestone Schedule

| Phase | Project Goal | Milestone | 16-Week Schedule |
| :--- | :--- | :--- | :---: |
| 🧭 **Blueprint** | Architecture plan and proposal approved | Midterm Pitch & Repo Submitted | Week 10 |
| 🥊 **First Working Demo** | Pretrained ResNet50 pipeline runs end-to-end on sample batches | Pipeline verifies inference end-to-end | Week 11 |
| 🛠️ **Make It Yours** | Ingest full 2,661 dataset; fine-tune head on custom classes | Model actively trains on custom classes | Weeks 12–13 |
| 📈 **Improve & Measure** | Evaluate against targets (>= 80% Acc, >= 0.78 F1); tune augmentations | Test metrics & confusion matrix logged | Week 14 |
| 📦 **Package & Present** | Finalize PyTorch weights (< 100MB), README, demo video, and deck | Final deliverables submitted | Week 15 |

## 7. Top Risks & Plan B
- **Domain Overfitting (Line Art Confusion):** Fine tattoo linework resembles intricate graffiti.  
  *Plan B:* Unfreeze deeper ResNet residual layers to learn fine dermis textures, and add heavy color jitter and affine augmentations.
- **Class Imbalance:** Skew toward the `none` class (1,164 vs 695 `street_art`) risks default majority voting.  
  *Plan B:* Apply class-weighted Cross-Entropy Loss to penalize minority-class errors, combined with targeted augmentation sampling.
- **Compute & Cost:** Primary environment on Google Colab T4 GPU with Kaggle GPU backup ($0.00 estimated budget).
