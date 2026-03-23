"""
LNDiff OOF Inference
Generate out-of-fold predictions
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
from tqdm import tqdm
import argparse

from lndiff.models.swin_apdam import Swin_APDAM_Model
from lndiff.core.metrics import calculate_metrics
from lndiff.data.dataset import find_all_patients
from lndiff.data.transforms import get_val_transforms
from lndiff.scripts.utils import set_seed


def get_config():
    parser = argparse.ArgumentParser(description='LNDiff OOF Inference')
    parser.add_argument('--data_dir', type=str, default='/data/processed_frame')
    parser.add_argument('--run_root', type=str, default='/outputs/checkpoints/runs_lndiff')
    parser.add_argument('--output_csv', type=str, default='/outputs/csv_results/oof_predictions.csv')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.35)
    
    args = parser.parse_args()
    return vars(args)


def predict_patient(model, p_imgs, transform, device, batch_size=64, threshold=100):
    """Patient-level prediction."""
    from PIL import Image
    
    target_imgs = p_imgs
    if len(p_imgs) > threshold:
        step = len(p_imgs) // threshold
        target_imgs = p_imgs[::step]

    if len(target_imgs) == 0:
        return 0.0

    batch_tensors = []
    for img_path in target_imgs:
        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            batch_tensors.append(img)
        except:
            pass

    if not batch_tensors:
        return 0.0

    input_imgs = torch.stack(batch_tensors)
    probs_list = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(input_imgs), batch_size):
            chunk = input_imgs[i: i + batch_size].to(device)
            out = model(chunk)
            prob = torch.softmax(out, dim=1)[:, 1]
            probs_list.append(prob.cpu())

    return torch.mean(torch.cat(probs_list)).item() if probs_list else 0.0


def main(config):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_patients = find_all_patients(config["data_dir"])
    labels = [p['label'] for p in all_patients]

    skf = StratifiedKFold(
        n_splits=config["n_splits"],
        shuffle=True,
        random_state=config["seed"]
    )

    all_predictions = []
    fold_metrics_list = []
    transform = get_val_transforms(config["img_size"])
    class_names = {0: 'Tuberculosis', 1: 'Metastasis'}

    for fold_idx, (train_index, val_index) in enumerate(skf.split(all_patients, labels)):
        val_patients = [all_patients[i] for i in val_index]

        model_path = os.path.join(config["run_root"], f"Fold_{fold_idx + 1}", "best_model.pth")
        if not os.path.exists(model_path):
            continue

        model = Swin_APDAM_Model(num_classes=2, swin_path=None)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)

        fold_y_true = []
        fold_y_scores = []

        for p in tqdm(val_patients, desc=f"Fold {fold_idx + 1}", leave=False):
            prob = predict_patient(model, p['images'], transform, device, config["batch_size"])
            pred_label = 1 if prob > config["threshold"] else 0

            all_predictions.append({
                "Patient_ID": p['id'],
                "Fold": fold_idx + 1,
                "Probability": round(prob, 4),
                "Predicted_Label": pred_label,
                "Predicted_Class": class_names[pred_label],
                "Actual_Label": p['label'],
                "Actual_Class": class_names[p['label']],
                "Correct": "Yes" if pred_label == p['label'] else "No"
            })

            fold_y_true.append(p['label'])
            fold_y_scores.append(prob)

        metrics = calculate_metrics(np.array(fold_y_true), np.array([1 if p > config["threshold"] else 0 for p in fold_y_scores]), np.array(fold_y_scores))
        fold_metrics_list.append(metrics)

    result_df = pd.DataFrame(all_predictions)
    result_df = result_df.sort_values(by=["Fold", "Patient_ID"])
    result_df.to_csv(config["output_csv"], index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    config = get_config()
    main(config)
