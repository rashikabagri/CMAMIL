This repository contains the official PyTorch implementation of CMA-MIL, a multi-scale multiple instance learning (MIL) framework designed for whole-slide image (WSI) classification. CMA-MIL integrates cross-magnification attention to effectively model multi-resolution histopathology features.

Overview
CMA-MIL operates on multi-magnification patch graphs (e.g., 5×, 10×, 20×) and jointly learns:
Cross-magnification interactions between patches at different scales
Gated attention pooling within each magnification
Bag-level slide classification
Instance-level supervision using top-k positive and negative patches

CMA_MIL(
  ├─ Cross-Magnification Attention
  ├─ Gated Attention Pooling (per scale)
  ├─ Feature Fusion )

Repository Structure
├── main.py                     
├── models/
│   └── cma_mil.py               # CMA-MIL model definition
├── utils/
    └── data_utils.py    
│   └── train_eval.py            # Training, evaluation, early stopping
│   └── data_utils.py            # Multi-scale data loader
├── README.md

CMA-MIL expects pre-extracted patch-level features at each magnification.

data_root_5x/
  └── fold_id/
      ├── train/
      │   └── class_name/*.npy
      ├── val/
      └── test/

data_root_10x/
data_root_20x/

Training
python main.py \
  --data_root_5x data/5x \
  --data_root_10x data/10x \
  --data_root_20x data/20x \
  --folds 5 \
  --epochs 20 \
  --bag_loss ce \
  --inst_loss svm \
  --early_stopping \

| Argument           | Description                         | Default |
| ------------------ | ----------------------------------- | ------- |
| `--bag_loss`       | Bag-level loss (`ce` or `svm`)      | `ce`   |
| `--inst_loss`      | Instance-level loss (`ce` or `svm`) | `svm`   |
| `--bag_weight`     | Weight for bag-level loss           | `0.7`   |
| `--k_sample`       | Top-k instances for supervision     | `8`     |
| `--drop_out`       | Dropout rate                        | `0.2`   |
| `--embed_dim`      | Feature embedding dimension         | `512`   |
| `--dim`            | Attention hidden dimension          | `128`   |
| `--weight_decay`   | L2 regularization                   | `1e-5`  |
| `--early_stopping` | Enable early stopping               | `False` |
| `--log_data`       | Enable TensorBoard logging          | `False` |


During training and testing, we report:

Accuracy
Precision / Recall / F1-score
Mean multi-class AUC (one-vs-rest)
Cross-validation results are averaged across folds.

Reproducibility
Fixed random seeds
Cross-validation splits saved as CSV (dataset_folds)
All hyperparameters configurable via CLI

If you use this code in your research, please cite:
@article{CMA-MIL,
  title={Cross-Magnification Attention for Multi-Scale Multiple Instance Learning},
  author={...},
  journal={...},
  year={2025}
}
