
---

# Intel Image Classification — CNN Pipeline (PyTorch & TensorFlow)

End-to-end image classification pipeline for the
[Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification),
implementing a CNN model in both **PyTorch** and **TensorFlow/Keras** with a unified CLI.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Saved Models](#saved-models)
- [Results](#results)

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle — Intel Image Classification |
| Classes | `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street` |
| Train samples | 14 034 |
| Test samples | 3 000 |
| Image size | 150 × 150 px (RGB) |

Download the dataset from Kaggle and place it so the structure looks like:

```
data/
└── archive/
    ├── seg_train/
    │   └── seg_train/
    │       ├── buildings/
    │       ├── forest/
    │       ├── glacier/
    │       ├── mountain/
    │       ├── sea/
    │       └── street/
    ├── seg_test/
    │   └── seg_test/
    │       └── <same 6 classes>/
    └── seg_pred/
        └── seg_pred/
```

---

## Project Structure

```
mnist/
├── main.py                  # Entry point — CLI for training & evaluation
├── requirements.txt         # Python dependencies
├── README.md
├── models/
│   ├── cnn.py               # CNN architecture — PyTorch
│   ├── cnn_tf.py            # CNN architecture — TensorFlow/Keras
│   └── train.py             # Trainer (PyTorch) + TFTrainer (TensorFlow)
└── utils/
    └── prep.py              # Data loading, augmentation & normalization
```

---

## Pipeline Overview

### Preprocessing & Augmentation

| Stage | Train | Test |
|---|---|---|
| Resize | 150 × 150 | 150 × 150 |
| Horizontal flip | yes (p=0.5) | no |
| Rotation | ± 15° | no |
| Color jitter | brightness / contrast / saturation | no |
| Translation | ± 5% | no |
| Normalization | ImageNet mean/std (PyTorch) · rescale ÷ 255 (TF) | same |

### Architecture

```
Input (3 × 150 × 150)
  └─ Conv(32)  + BatchNorm + ReLU + MaxPool(2)   → 75 × 75
  └─ Conv(64)  + BatchNorm + ReLU + MaxPool(2)   → 37 × 37
  └─ Conv(128) + BatchNorm + ReLU + MaxPool(2)   → 18 × 18
  └─ Conv(256) + BatchNorm + ReLU + MaxPool(2)   →  9 × 9
  └─ GlobalAveragePooling
  └─ Dropout(0.5) → FC(512) → ReLU → Dropout(0.3) → FC(6)
Output (6 classes)
```

### Training strategy

| Setting | PyTorch | TensorFlow |
|---|---|---|
| Optimizer | Adam + weight decay | Adam |
| Loss | CrossEntropyLoss | CategoricalCrossentropy |
| LR scheduler | — | ReduceLROnPlateau (patience=3) |
| Early stopping | — | EarlyStopping (patience=7) |
| Default epochs | 20 | 20 |
| Batch size | 32 | 32 |

---

## Installation

**Prerequisite:** Python 3.12+

### Option A — Conda (recommended)

```bash
# 1. Create environment (once)
conda create --name aims_cv python=3.12

# 2. Activate
conda activate aims_cv

# 3. Install dependencies
pip install -r requirements.txt
```

### Option B — venv

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Jupyter (optional)

```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=aims_cv
```

---

## Usage

All commands must be run from the `mnist/` directory.

### Train — PyTorch

```bash
# Basic training (CPU)
python main.py --framework pytorch --mode train

# With GPU + custom hyperparameters
python main.py --framework pytorch --mode train --epochs 30 --lr 0.001 --wd 1e-4 --cuda
```

### Train — TensorFlow / Keras

```bash
# Basic training
python main.py --framework tensorflow --mode train

# Custom epochs and learning rate
python main.py --framework tensorflow --mode train --epochs 30 --lr 0.0005
```

### Evaluate a saved model

```bash
# PyTorch  (loads rosly_model.pth)
python main.py --framework pytorch --mode eval --cuda

# TensorFlow  (loads rosly_model.keras)
python main.py --framework tensorflow --mode eval
```

### Full argument reference

| Argument | Values | Default | Description |
|---|---|---|---|
| `--framework` | `pytorch` \| `tensorflow` | **required** | Framework to use |
| `--mode` | `train` \| `eval` | `train` | Train or evaluate only |
| `--epochs` | int | `20` | Number of training epochs |
| `--lr` | float | `0.001` | Learning rate |
| `--wd` | float | `1e-4` | Weight decay (PyTorch only) |
| `--cuda` | flag | off | Use GPU if available (PyTorch only) |

---

## Saved Models

After training the following files are written to `mnist/`:

| File | Framework | Description |
|---|---|---|
| `rosly_model.pth` | PyTorch | Model state dict |
| `rosly_model.keras` | TensorFlow/Keras | Full Keras model |
| `training_with_pytorch.png` | PyTorch | Loss & accuracy curves |
| `training_with_keras.png` | TensorFlow | Loss & accuracy curves |

---

## Results

> Fill in after training.

| Framework | Test Accuracy | Test Loss |
|---|---|---|
| PyTorch | — | — |
| TensorFlow | — | — |

---
# AIMS-Computer-vision
