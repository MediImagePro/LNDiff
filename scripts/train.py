"""
LNDiff Training - 5-Fold Cross Validation
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import json
import datetime
import random
import argparse

from lndiff.models.swin_apdam import Swin_APDAM_Model
from lndiff.core.losses import FocalLoss
from lndiff.core.engine import train_epoch, validate_epoch, save_checkpoint, EarlyStopping
from lndiff.core.metrics import calculate_metrics
from lndiff.data.dataset import UltrasoundDataset, find_all_patients
from lndiff.data.transforms import get_train_transforms, get_val_transforms
from lndiff.scripts.utils import set_seed


def get_config():
    parser = argparse.ArgumentParser(description='LNDiff 5-Fold Training')
    parser.add_argument('--exp_name', type=str, default='5Fold_LNDiff_Swin-APDAM')
    parser.add_argument('--data_dir', type=str, default='/data/processed_frame')
    parser.add_argument('--output_root', type=str, default='/outputs/checkpoints/runs_lndiff')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--focal_alpha', type=float, default=0.7)
    parser.add_argument('--focal_gamma', type=float, default=3.0)
    parser.add_argument('--valid_threshold', type=float, default=0.35)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--swin_checkpoint', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=224)
    
    args = parser.parse_args()
    return vars(args)


def flatten_data_for_fold(patients, is_train=True):
    """Flatten patient list to image paths and labels."""
    img_paths = []
    labels = []

    for p in patients:
        p_imgs = p['images']
        label = p['label']
        
        if is_train:
            limit = 1000 if label == 1 else 600
            limit = min(limit, len(p_imgs))
            current_imgs = random.sample(p_imgs, limit)
        elif len(p_imgs) > 100:
            step = len(p_imgs) // 100
            current_imgs = p_imgs[::step]
        else:
            current_imgs = p_imgs

        img_paths.extend(current_imgs)
        labels.extend([label] * len(current_imgs))
    
    return img_paths, labels


def run_fold(fold_idx, train_patients, val_patients, run_root, device, config):
    """Run training for a single fold."""
    fold_dir = os.path.join(run_root, f"Fold_{fold_idx+1}")
    os.makedirs(fold_dir, exist_ok=True)

    train_paths, train_labels = flatten_data_for_fold(train_patients, is_train=True)
    val_paths, val_labels = flatten_data_for_fold(val_patients, is_train=False)

    train_ds = UltrasoundDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=get_train_transforms(config['img_size'])
    )
    val_ds = UltrasoundDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=get_val_transforms(config['img_size'])
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    model = Swin_APDAM_Model(
        num_classes=2,
        swin_path=config["swin_checkpoint"]
    ).to(device)
    
    criterion = FocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"])
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    scaler = GradScaler('cuda')
    early_stopping = EarlyStopping(patience=config["patience"])

    best_auc = 0.0
    best_metrics = {}
    best_model_path = os.path.join(fold_dir, 'best_model.pth')
    checkpoint_path = os.path.join(fold_dir, 'checkpoint.pth')

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=True
        )
        scheduler.step()

        val_loss, y_true, y_pred, y_scores = validate_epoch(
            model, val_loader, criterion, device
        )
        
        metrics = calculate_metrics(y_true, y_pred, y_scores)
        
        print(f"Fold {fold_idx+1} Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"AUC: {metrics['AUC']:.4f} | Acc: {metrics['Accuracy']:.4f}")

        if metrics['AUC'] > best_auc:
            best_auc = metrics['AUC']
            best_metrics = metrics
            torch.save(model.state_dict(), best_model_path)
            save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, scaler, best_metrics)

        early_stopping(metrics['AUC'])
        if early_stopping.early_stop:
            break

    return best_metrics


def main(config):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(config["output_root"], f"{timestamp}_{config['exp_name']}")
    os.makedirs(run_root, exist_ok=True)
    
    with open(os.path.join(run_root, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    all_patients = find_all_patients(config["data_dir"])
    labels = [p['label'] for p in all_patients]

    skf = StratifiedKFold(
        n_splits=config["n_splits"],
        shuffle=True,
        random_state=config["seed"]
    )

    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(all_patients, labels)):
        train_patients = [all_patients[i] for i in train_indices]
        val_patients = [all_patients[i] for i in val_indices]
        
        metrics = run_fold(fold_idx, train_patients, val_patients, run_root, device, config)
        fold_results.append(metrics)

    mean_auc = np.mean([m['AUC'] for m in fold_results])
    mean_acc = np.mean([m['Accuracy'] for m in fold_results])
    mean_sens = np.mean([m['Sensitivity'] for m in fold_results])
    mean_spec = np.mean([m['Specificity'] for m in fold_results])
    
    summary = {
        "fold_results": fold_results,
        "mean_metrics": {
            "AUC": mean_auc,
            "Accuracy": mean_acc,
            "Sensitivity": mean_sens,
            "Specificity": mean_spec
        }
    }
    
    with open(os.path.join(run_root, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == '__main__':
    config = get_config()
    main(config)
