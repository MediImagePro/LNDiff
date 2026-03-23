# LNDiff: Lymph Node Differentiation using Swin Transformer

Automated classification of lymph nodes in ultrasound images using Swin Transformer with Attention Pooling.

## Project Structure

```
LNDiff_Ultrasound_Diagnosis/
├── lndiff/                    # Core package
│   ├── core/                 # Training engine and metrics
│   ├── data/                 # Data loading and preprocessing
│   ├── models/               # Model architectures
│   └── scripts/              # Utility functions
├── scripts/                  # Main scripts
│   ├── train.py             # 5-Fold training
│   ├── inference_oof.py      # OOF predictions
│   └── visualize_gradcam.py  # GradCAM visualization
├── tools/                    # Data preprocessing
│   ├── frame_extractor.py
│   ├── roi_cropper.py
│   └── preprocessing_pipeline.py
├── data/                     # Dataset directory
├── weights/                  # Pretrained weights
└── requirements.txt
```

## Installation

```bash
conda create -n lymph python=3.9
conda activate lymph
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python scripts/train.py \
    --data_dir /path/to/data/processed_frame \
    --output_root /path/to/outputs \
    --batch_size 128 \
    --epochs 80
```

### Inference

```bash
python scripts/inference_oof.py \
    --data_dir /path/to/data/processed_frame \
    --run_root /path/to/checkpoints \
    --output_csv /path/to/oof_predictions.csv
```

### GradCAM Visualization

```bash
python scripts/visualize_gradcam.py \
    --data_dir /path/to/data/processed_frame \
    --run_root /path/to/checkpoints \
    --oof_csv /path/to/oof_predictions.csv
```

## Dataset Format

Organize images as:
```
data/processed_frame/
├── 结核性淋巴结/          # Tuberculous (Label: 0)
│   ├── patient_001/
│   │   └── b超/
│   │       ├── image_001.png
│   │       └── ...
└── 转移性淋巴结/          # Metastatic (Label: 1)
    ├── patient_100/
    │   └── b超/
    │       └── ...
```

See `data/README.md` for details.

## Model Architecture

- **Backbone**: Swin Transformer Tiny (ImageNet-1K pretrained)
- **Pooling**: Attention Pooling with learnable query
- **Head**: Linear classifier with BatchNorm and GELU

## Training Configuration

- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.05)
- **Loss**: Focal Loss (α=0.7, γ=3.0)
- **Scheduler**: Cosine Annealing
- **Validation**: 5-Fold Stratified Cross-Validation
- **Early Stopping**: Patience=10

## Metrics

- AUC (Area Under ROC Curve)
- Accuracy
- Sensitivity
- Specificity

## Output

Results saved in checkpoint directory:
- `best_model.pth` - Model weights
- `checkpoint.pth` - Full checkpoint (for resuming)
- `summary.json` - Mean metrics across folds

## References

- Swin Transformer: [arXiv:2105.01601](https://arxiv.org/abs/2105.01601)
- Pretrained weights: [Hugging Face](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
