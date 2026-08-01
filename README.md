# Diabatic-B5 — Retinal Disease Classification using EfficientNet-B5

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-EfficientNet--B5-brightgreen)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20notebook-20BEFF?logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

A deep learning project for **binary classification of retinal diseases**
from fundus images, built using a pretrained **EfficientNet-B5** model.
The pipeline is trained and evaluated on two datasets — **APTOS 2019**
(primary) and **RFMiD** (generalization check).

> ⚠️ **Not for clinical use.** This is a research/portfolio project. It
> has not been validated for diagnostic use and should not be used to
> make real medical decisions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Data Augmentation](#data-augmentation)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [ROC Curve & PR Curve](#roc-curve--pr-curve)
- [Overfitting Analysis](#overfitting-analysis)
- [Additional Analysis](#additional-analysis)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Author](#author)
- [License](#license)

---

## Project Overview

| Property | Detail |
|---|---|
| Task | Binary Image Classification (Disease / No Disease) |
| Architecture | EfficientNet-B5 (Pretrained on ImageNet) |
| Primary Dataset | APTOS 2019 (Diabetic Retinopathy) |
| Generalization Dataset | RFMiD (Retinal Fundus Multi-disease Image Dataset) |
| Framework | PyTorch + timm |
| Input Resolution | 456 × 456 px |

## Dataset Description

### APTOS 2019 — Primary Training Dataset
- Fundus images labeled for diabetic retinopathy severity (0–4)
- Converted to **binary labels**: `0 = No DR`, `1 = DR present`
- Used for model training, validation, and core evaluation

### RFMiD — Generalization / Overfitting Check
- Multi-disease retinal fundus dataset
- Used the `Disease_Risk` column as the binary target
- **Purpose:** Evaluate whether the model generalizes beyond the APTOS domain, and to check for overfitting
- A model that performs well on both datasets is considered well-generalized

## Model Architecture

```
EfficientNet-B5 (Pretrained ImageNet Backbone)
        │
        ▼
  Feature Extractor (MBConv Blocks)
        │
        ▼
  Global Average Pooling
        │
        ▼
  Dropout (p = 0.5)
        │
        ▼
  Linear Layer (in_features → 1)
        │
        ▼
  BCEWithLogitsLoss (Binary Output)
```

- Backbone weights initialized from **ImageNet pretrained EfficientNet-B5**
- Classifier head replaced with a custom `Dropout → Linear(1)` layer
- Output passed through `sigmoid` for probability estimation

## Training Pipeline

### Phase 1 — Frozen Backbone (8 Epochs)
- Only the classifier head is trained
- Optimizer: `AdamW` — lr = `3e-4`, weight_decay = `1e-4`
- Scheduler: `CosineAnnealingLR` (T_max = 10)

### Phase 2 — Full Fine-tuning (5 Epochs)
- All layers unfrozen
- Optimizer: `AdamW` — lr = `1e-5`, weight_decay = `1e-4`
- Scheduler: `CosineAnnealingLR` (T_max = 5)

### Phase 3 — Final Refinement (4 Epochs)
- Gradient clipping (`max_norm = 1.0`) applied
- Test-time augmentation: horizontal flip averaging
- Scheduler: `CosineAnnealingLR` (T_max = 4)

## Data Augmentation

| Split | Transforms |
|---|---|
| Training | Resize (456×456), RandomHorizontalFlip, RandomRotation(±15°), ColorJitter, Normalize |
| Validation | Resize (456×456), Normalize |

ImageNet normalization stats used: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

## Evaluation Metrics

The following metrics are computed on the validation set:

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | True positives out of all positive predictions |
| **Recall (Sensitivity)** | True positives out of all actual positives |
| **Specificity** | True negatives out of all actual negatives |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the ROC Curve |
| **PR-AUC** | Area under the Precision-Recall Curve |
| **MCC** | Matthews Correlation Coefficient |
| **Cohen's Kappa** | Agreement beyond chance |
| **Balanced Accuracy** | Average of Sensitivity and Specificity |

## Results

### APTOS 2019 — Validation Set

| Metric | Score |
|---|---|
| Accuracy | ~0.97+ |
| ROC-AUC | ~0.98+ |
| F1 Score | ~0.97+ |
| MCC | ~0.94+ |

> *Exact scores depend on your training run. Update this table after running the notebook.*

### RFMiD — Generalization Check

| Metric | Score |
|---|---|
| Accuracy | *(fill after run)* |
| MCC | *(fill after run)* |

> A strong performance on RFMiD confirms the model generalizes well and is **not overfitting** to the APTOS domain.

## ROC Curve & PR Curve

The notebook generates:

- **ROC Curve** — plots True Positive Rate vs False Positive Rate with AUC score
- **Precision-Recall Curve** — especially useful for imbalanced datasets
- **Calibration Curve** — evaluates how well predicted probabilities match true outcomes
- **Confusion Matrix** — for both training and validation sets

## Overfitting Analysis

Two checks are performed:

1. **Generalization Gap** — compares final train vs. validation accuracy
   - Gap < 2% → Well generalized
   - Gap 2–5% → Slight overfitting
   - Gap > 5% → Possible overfitting

2. **Validation Loss Trend** — if val loss increases after its minimum, overfitting is flagged

3. **Cross-dataset evaluation on RFMiD** — the strongest real-world generalization test

## Additional Analysis

- **Threshold Analysis** — evaluates Recall and Precision at thresholds: `0.3, 0.4, 0.5, 0.6, 0.7`
- **Bootstrap Confidence Interval** — 95% CI on accuracy over 1000 iterations
- **Gaussian Noise Robustness** — utility function to test model stability under noise

## Project Structure

```
Diabatic-B5/
├── epics-project.ipynb       # Main notebook (training + evaluation)
├── README.md                 # Project documentation
└── .gitignore                # Ignores model weights, checkpoints, datasets
```

## How to Run

1. Upload the notebook to [Kaggle](https://www.kaggle.com) (recommended — free GPU)
2. Attach the following datasets:
   - [APTOS 2019](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)
   - [RFMiD](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification)
3. Run all cells in order

## Dependencies

```
torch
torchvision
timm
scikit-learn
pandas
numpy
matplotlib
seaborn
Pillow
tqdm
```

## Author

**AnshumanJ28**
GitHub: [@AnshumanJ28](https://github.com/AnshumanJ28)

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
