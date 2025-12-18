This repository contains the official PyTorch implementation of **CMA-MIL**, a multi-scale multiple instance learning (MIL) framework designed for **whole-slide image (WSI) classification**. CMA-MIL integrates **cross-magnification attention** to effectively model **multi-resolution histopathology features**.

---

## Processing Raw WSI Data

This repository includes a **DeepZoom-based WSI patch extraction pipeline** designed for **CMA-MIL training**. The pipeline efficiently extracts informative tissue patches from high-resolution WSIs while filtering out background regions.

Key characteristics of the patch extraction pipeline include:

1. **DeepZoom pyramidal tiling** for efficient large-WSI processing  
2. **Multi-processing** for scalable and fast patch extraction  
3. **Edge-density–based tissue filtering** to remove background patches  
4. **Multi-magnification extraction** (e.g., 5×, 10×, and 20×)  
5. **CMA-MIL–ready bag organization** (class / slide / patches)  

---

## 📂 Expected WSI Directory Structure

Raw WSIs should be organized as follows:

```text
WSI_root/
├── class_1/
│   ├── slide_001.svs
│   ├── slide_002.svs
├── class_2/
│   ├── slide_101.svs
│   ├── slide_102.svs

🗂 Extracted Patch Organization

Extracted patches are automatically organized by magnification, class, and slide, forming CMA-MIL–compatible bags.

5× Magnification
patches_5x/
├── class_1/
│   ├── slide_001/
│   │   ├── 0_0.jpeg
│   │   ├── 0_1.jpeg
│   │   └── ...
│   └── slide_002/
├── class_2/
│   └── slide_101/

10× Magnification
patches_10x/
├── class_1/
│   ├── slide_001/
│   │   ├── 0_0.jpeg
│   │   ├── 0_1.jpeg
│   │   └── ...
│   └── slide_002/
├── class_2/
│   └── slide_101/

20× Magnification
patches_20x/
├── class_1/
│   ├── slide_001/
│   │   ├── 0_0.jpeg
│   │   ├── 0_1.jpeg
│   │   └── ...
│   └── slide_002/
├── class_2/
│   └── slide_101/
```

### Running Patch Extraction

python patch_tiling.py \
  --data_root data/raw_wsi \
  --out_dir patches \
  --tile_size 224 \
  --base_mag 20 \
  --magnifications 0 \
  --workers 4 \
  --threshold 15

## 🧠 Feature Computation (ShuffleNet → `.npy`)

After patch extraction, we compute **patch-level feature embeddings** using a **pretrained ShuffleNet-V2 backbone**. These features are used as input to the CMA-MIL model.

The feature computation step:
- Does **not** use any MIL model
- Operates **independently on each patch**
- Produces **512-dimensional embeddings**
- Saves features in **NumPy (`.npy`) format** for efficient loading

---

### 🔍 Feature Extractor

- Backbone: **ShuffleNet-V2 (x1.0)**
- Pretrained on: **ImageNet**
- Classification head removed
- Projection head: `1024 → 512`
- Output per patch: **512-D feature vector**

---

### 📂 Input (Patch Directory)

The script expects patches organized as:

```text
patches_5x/
├── class_1/
│   ├── slide_001/
│   │   ├── 0_0.jpeg
│   │   ├── 0_1.jpeg
│   │   └── ...
│   └── slide_002/
├── class_2/
│   └── slide_101/

The same structure applies to patches_10x/ and patches_20x/.

Running Feature Extraction
python compute_features.py \
  --patch_root patches_5x \
  --out_dir features \
  --batch_size 128 \
  --num_workers 4

Also run separately for each magnification (5×, 10×, 20×).

Extracted features are saved as one .npy file per slide (bag):

features/
└── patches_5x/
    ├── class_1/
    │   ├── slide_001.npy   # (N_patches × 512)
    │   └── slide_002.npy
    ├── class_2/
    │   └── slide_101.npy
    └── dataset_index.npy

Each .npy file contains all patch features for a single WSI
Shape: (number_of_patches, 512)
```

Patch-level features were extracted using a pretrained ShuffleNet-V2 backbone. The classification layer was removed, and a lightweight projection head was used to obtain 512-dimensional embeddings. For each whole-slide image, patch features were aggregated and stored as NumPy arrays to support efficient multi-scale multiple instance learning.


## Training Pipeline

This repository supports **feature-based representations of whole-slide images (WSIs)** for training the **CMA-MIL** model. Each slide is represented as a **set of patch-level feature vectors**, stored as NumPy (`.npy`) files. These features are loaded directly during training without constructing any graph structures.

---

## Feature Representation

- Each WSI is represented as a **NumPy array (`.npy`)**
- Shape of each file:
(number_of_patches, feature_dimension)

yaml
Copy code
- Feature dimension is typically **512**
- Each row corresponds to **one patch-level feature embedding**

---

## 📂 Feature Directory Structure

Features must be organized by **class label** and **magnification**:

```text
features_5x/
├── class_0/
│   ├── slide_001.npy
│   ├── slide_002.npy
├── class_1/
│   ├── slide_101.npy

features_10x/
├── class_0/
│   ├── slide_001.npy
│   ├── slide_002.npy
├── class_1/
│   ├── slide_101.npy

features_20x/
├── class_0/
│   ├── slide_001.npy
│   ├── slide_002.npy
├── class_1/
│   ├── slide_101.npy
Each .npy file corresponds to one slide (one MIL bag).
```

Multi-Magnification Feature Loading
The training pipeline synchronously loads features from 5×, 10×, and 20× magnifications:

python
Copy code
yield features_5x, features_10x, features_20x, label
This design:

Ensures label consistency across magnifications

Enables cross-magnification attention in CMA-MIL

Uses three feature tensors and one slide-level label per sample

Loss Functions
The pipeline supports flexible loss configurations at both levels.

Bag-Level Loss
Cross-Entropy Loss
Smooth Top-1 SVM (optional)

Instance-Level Loss
Cross-Entropy Loss
Smooth Top-1 SVM (optional)

This enables effective supervision at both bag and instance levels.

Early Stopping
An early stopping strategy is applied during training:
Monitors validation AUC
Saves the best-performing model checkpoint
Stops training when performance does not improve beyond a patience threshold

Training and Evaluation
Training
Optimizer: Adam
Supports weight decay regularization
Combines bag-level and instance-level losses
Trains using fold-wise evaluation

Evaluation Metrics
The following metrics are reported:
Accuracy
Precision
Recall
F1-score

Mean multi-class ROC-AUC (one-vs-rest)

Fold-Wise Training
The train_fold routine:
Trains CMA-MIL on a training split
Validates after each epoch
Applies early stopping
Evaluates final performance on a held-out test split
This setup supports cross-validation experiments.

Key Training Arguments
Argument	Description	Default
--bag_loss	Bag-level loss (ce or svm)	ce
--inst_loss	Instance-level loss (ce or svm)	svm
--bag_weight	Weight for bag-level loss	0.7
--k_sample	Top-k instances for supervision	8
--drop_out	Dropout rate	0.2
--embed_dim	Feature embedding dimension	512
--dim	Attention hidden dimension	128
--weight_decay	L2 regularization	1e-5
--early_stopping	Enable early stopping	False
--log_data	Enable TensorBoard logging	False

📈 Reported Results
During training and testing, we report:

Accuracy

Precision / Recall / F1-score

Mean multi-class ROC-AUC (one-vs-rest)

Cross-validation results are averaged across folds.

Reproducibility
Fixed random seeds
Fold-wise cross-validation splits saved as CSV files


