# Pretrained Weights

## Model Architecture

This project uses **Swin Transformer Tiny** as the backbone architecture for feature extraction.

## Pretrained Weights Source

The pretrained weights are obtained from **Hugging Face Model Hub**:

- **Model Name**: `swin_tiny_patch4_window7_224`
- **Source**: [Hugging Face - Swin Tiny](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
- **Framework**: PyTorch
- **Pretraining Dataset**: ImageNet-1K

## Model Specifications

- **Architecture**: Swin Transformer
- **Variant**: Tiny
- **Patch Size**: 4×4
- **Window Size**: 7×7
- **Input Resolution**: 224×224
- **Number of Parameters**: ~28M
- **Embedding Dimension**: 96

## Loading Pretrained Weights

The weights are automatically loaded during model initialization:

```python
from lndiff.models.swin_apdam import Swin_APDAM_Model

# Load model with pretrained weights
model = Swin_APDAM_Model(
    num_classes=2,
    swin_path='/path/to/pretrained/weights.pth'
)
```

## Fine-tuning Strategy

The pretrained weights are fine-tuned on the lymph node ultrasound dataset using:

- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Loss Function**: Focal Loss (α=0.7, γ=3.0)
- **Scheduler**: Cosine Annealing
- **Training Epochs**: 80
- **Early Stopping**: Patience=10

## Citation

If you use these pretrained weights, please cite:

```bibtex
@article{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutao and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}
```

## License

The Swin Transformer weights are released under the MIT License by Microsoft Research.

## References

- [Swin Transformer Paper](https://arxiv.org/abs/2105.01601)
- [Hugging Face Model Card](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
- [Official GitHub Repository](https://github.com/microsoft/Swin-Transformer)
