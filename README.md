# ecg-arrhythmia-mitbih

ECG arrhythmia classification on MIT-BIH.

> **Blog post:** [coming soon]

## The question

Can a simple 1D CNN reliably detect arrhythmias from single ECG beats?
Short answer: it depends entirely on how you define "reliable" — and accuracy is the wrong metric.

## Setup

```bash
uv sync
```

## Reproduce

```bash
# 1. Download data and preprocess
uv run python src/data_loader.py

# 2. Train (three experiments)
uv run python src/train.py --config configs/baseline.yaml
uv run python src/train.py --config configs/weighted.yaml
uv run python src/train.py --config configs/focal.yaml

# 3. Open notebooks in order
```

## Results

*Coming once experiments are run.*

| Experiment | Accuracy | Macro F1 | V F1 | F F1 |
|---|---|---|---|---|
| Baseline | — | — | — | — |
| Weighted CE | — | — | — | — |
| Focal Loss | — | — | — | — |

