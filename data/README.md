# Dataset Documentation

## Overview

This directory contains the ultrasound lymph node dataset used for the LNDiff project. The dataset consists of B-mode ultrasound images of lymph nodes classified into two categories: Tuberculous (TB) and Metastatic (Meta).

## Directory Structure

```
data/
├── 结核性淋巴结/                    # Tuberculous lymph nodes (Label: 0)
│   ├── patient_id_1/
│   │   ├── b超/
│   │   │   ├── image_001.jpg
│   │   │   ├── image_002.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── patient_id_2/
│   └── ...
│
└── 转移性淋巴结/                    # Metastatic lymph nodes (Label: 1)
    ├── patient_id_3/
    │   ├── b超/
    │   │   ├── image_001.jpg
    │   │   ├── image_002.jpg
    │   │   └── ...
    │   └── ...
    ├── patient_id_4/
    └── ...
```

## Class Labels

- **0 (结核性淋巴结)**: Tuberculous lymph nodes
- **1 (转移性淋巴结)**: Metastatic lymph nodes

## Image Format

- **formats**:  PNG
- **Resolution**: Variable (automatically resized to 224×224 during preprocessing)
- **Color space**: RGB
- **Modality**: B-mode ultrasound

## Data Loading

The dataset is loaded using the `find_all_patients()` function from `lndiff/data/dataset.py`:

```python
from lndiff.data.dataset import find_all_patients

# Load all patients
all_patients = find_all_patients('/path/to/data')

# Each patient is a dictionary:
# {
#     'id': 'patient_id_1',
#     'label': 0,  # 0 for TB, 1 for Meta
#     'images': ['/path/to/image_001.png', '/path/to/image_002.png', ...]
# }
```

## Data Splitting

The project uses **5-Fold Stratified Cross-Validation** for model training and evaluation:

- **Training set**: 80% of patients (4 folds)
- **Validation set**: 20% of patients (1 fold)
- **Stratification**: Ensures balanced class distribution across folds

### Sampling Strategy

During training, images are sampled to balance computational load:

- **Positive class (Meta)**: Up to 1000 images per patient
- **Negative class (TB)**: Up to 600 images per patient
- **Validation set**: Downsampled to ~100 images per patient

## Data Preprocessing

### Training Transforms
- Resize with aspect ratio preservation (224×224)
- Random horizontal flip
- Random vertical flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Random erasing (p=0.3)
- Normalization: ImageNet statistics

### Validation Transforms
- Resize with aspect ratio preservation (224×224)
- Normalization: ImageNet statistics

## Usage Example

```python
from lndiff.data.dataset import UltrasoundDataset, find_all_patients
from lndiff.data.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader

# Load patients
all_patients = find_all_patients('/data/processed_frame')

# Create dataset
train_ds = UltrasoundDataset(
    image_paths=train_paths,
    labels=train_labels,
    transform=get_train_transforms(img_size=224)
)

# Create dataloader
train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
```