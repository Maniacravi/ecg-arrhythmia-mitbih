# ecg-arrhythmia-mitbih

ECG arrhythmia classification on MIT-BIH: class imbalance, domain knowledge, and Grad-CAM.

> **Blog post:** [The Accuracy Trap: Class Imbalance and Domain Knowledge in ECG Arrhythmia Classification](https://maniravi.com/projects/ecg-arrhythmia-mit/)

## The question

Can a simple 1D CNN reliably detect arrhythmias from single ECG beats? It depends entirely on how you define reliable, and accuracy is the wrong metric here. A model that predicts Normal for every beat gets 83% accuracy and catches zero arrhythmias. Mine got 91%. The confusion matrix tells the rest of the story.

## Setup

```bash
uv sync
```

## Reproduce

```bash
# 1. Download MIT-BIH data (run once, ~110MB)
uv run python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='data/raw/')"

# 2. Run notebooks in order
#    01_eda.ipynb               — data exploration, beat morphology, class distribution
#    02_segmentation.ipynb      — beat segmentation, DS1/DS2 patient-wise split
#    03_baseline_cnn.ipynb      — baseline model, the lying accuracy
#    04_weighted_loss.ipynb     — weighted cross-entropy
#    05_focal_loss.ipynb        — focal loss
#    06_smote.ipynb             — SMOTE oversampling
#    07_rr_features.ipynb       — RR interval features, biggest improvement
#    08_gradcam.ipynb           — Grad-CAM heatmaps
```

## Results

All models trained on DS1 (22 records), evaluated on DS2 (22 records). Patient-wise split per AAMI EC57 — no data leakage. Most papers reporting >98% accuracy on MIT-BIH use intra-patient splits.

| Experiment | Accuracy | Macro F1 | S F1 | V F1 | F F1 |
|:---|---:|---:|---:|---:|---:|
| Baseline CNN | 0.91 | 0.29 | 0.01 | 0.51 | 0.00 |
| Weighted CE loss | 0.69 | 0.30 | 0.07 | 0.59 | 0.02 |
| Focal loss (γ=2) | 0.90 | 0.30 | 0.00 | 0.53 | 0.00 |
| SMOTE oversampling | 0.65 | 0.29 | 0.15 | 0.52 | 0.01 |
| **CNN + RR features** | **0.85** | **0.45** | **0.55** | **0.80** | **0.02** |

## Key finding

Loss function approaches (weighted CE, focal loss) and synthetic oversampling (SMOTE) produced marginal gains. The meaningful improvement came from injecting three RR interval features — pre-RR, post-RR, and local ratio — alongside CNN embeddings before the classifier.

S class F1 went from 0.01 to 0.55. Supraventricular beats are premature by definition: a timing property that morphology alone cannot capture. The model didn't need a better loss function. It needed better inputs.

## Grad-CAM

Grad-CAM heatmaps on the final conv layer reveal consistent, focused attention on clinically meaningful signal regions for N and V (high F1 classes), and diffuse, inconsistent attention for F and Q (near-zero F1). Attention consistency correlates directly with classification performance.

## Repo structure

```
notebooks/    one notebook per experiment, pre-run with outputs
src/
  data_loader.py   wfdb loading, AAMI mapping, DS1/DS2 split, RR feature extraction
  model.py         ECGNet and ECGNetRR (CNN + RR branch, Grad-CAM compatible)
  losses.py        FocalLoss
  gradcam.py       GradCAM1D
results/      saved figures
```

## Stack

Python, PyTorch, wfdb, scikit-learn, imbalanced-learn, Plotly, Quarto — managed with uv.