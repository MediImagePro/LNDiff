"""
LNDiff GradCAM Visualization
Generate Grad-CAM heatmaps
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import re
import argparse
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm

from lndiff.models.swin_apdam import Swin_APDAM_Model
from lndiff.data.transforms import get_val_transforms


def get_config():
    parser = argparse.ArgumentParser(description='LNDiff GradCAM Visualization')
    parser.add_argument('--data_dir', type=str, default='/data/processed_frame')
    parser.add_argument('--run_root', type=str, default='/outputs/checkpoints/runs_lndiff')
    parser.add_argument('--oof_csv', type=str, default='/outputs/csv_results/oof_predictions.csv')
    parser.add_argument('--output_dir', type=str, default='/results/gradcam_gallery')
    parser.add_argument('--imgs_per_patient', type=int, default=300)
    parser.add_argument('--img_size', type=int, default=224)
    
    args = parser.parse_args()
    return vars(args)


TARGET_CASES = {
    'Meta_Large': ['97535', '104227'],
    'Meta_Medium': ['158671', '160386'],
    'Meta_Small': ['123445', '104574'],
    'TB_Large': ['91189', '102874'],
    'TB_Medium': ['77303', '117118'],
    'TB_Small': ['72220', '93865']
}


def swin_reshape_transform(tensor, height=7, width=7):
    """Reshape Swin Transformer feature maps."""
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 3:
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        return result.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def get_patient_fold_map(csv_path):
    """Build Patient_ID -> Fold mapping from OOF CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'Patient_ID' not in df.columns or 'Fold' not in df.columns:
        raise ValueError(f"CSV must contain 'Patient_ID' and 'Fold' columns")

    mapping = {}
    for idx, row in df.iterrows():
        full_id = str(row['Patient_ID']).strip()
        match = re.match(r'^(\d+)', full_id)
        if match:
            short_id = match.group(1)
            fold = int(row['Fold'])
            mapping[short_id] = fold
            mapping[full_id] = fold

    return mapping


def load_models(run_root, device):
    """Load 5-fold models and GradCAM."""
    models = {}
    cams = {}

    for fold in range(1, 6):
        weight_path = os.path.join(run_root, f"Fold_{fold}", "best_model.pth")
        if not os.path.exists(weight_path):
            continue

        model = Swin_APDAM_Model(num_classes=2, swin_path=None)
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        target_layers = [model.vision_encoder.backbone.layers[-1].blocks[-1].norm1]
        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=swin_reshape_transform
        )

        models[fold] = model
        cams[fold] = cam

    return models, cams


def get_patient_image_list(root_dir, patient_id, label_class, max_imgs):
    """Get patient image list."""
    folder_name = '结核性淋巴结' if label_class == 0 else '转移性淋巴结'
    search_path = os.path.join(root_dir, folder_name, f"*{patient_id}*")
    candidates = glob.glob(search_path)
    
    if not candidates:
        return []

    p_path = candidates[0]
    b_mode = os.path.join(p_path, 'b超')
    if not os.path.exists(b_mode):
        b_mode = os.path.join(p_path, 'B超')
    if not os.path.exists(b_mode):
        b_mode = p_path

    images = glob.glob(os.path.join(b_mode, "*.[jpJP][pnPN]*"))
    images += glob.glob(os.path.join(b_mode, "*.bmp"))

    def natural_key(string_):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

    images.sort(key=natural_key)

    if not images:
        return []

    total = len(images)
    if total <= max_imgs:
        return images
    
    indices = np.linspace(0, total - 1, max_imgs, dtype=int)
    return [images[i] for i in np.unique(indices)]


def save_image_safe(file_path, img_bgr):
    """Save image with Chinese path support."""
    try:
        cv2.imencode('.jpg', img_bgr)[1].tofile(file_path)
    except Exception:
        pass


def clean_filename(pid):
    """Extract numeric ID."""
    match = re.match(r'^(\d+)', str(pid))
    return match.group(1) if match else str(pid)


def main(config):
    os.makedirs(config['output_dir'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id_to_fold = get_patient_fold_map(config['oof_csv'])
    models, cams = load_models(config['run_root'], device)
    transform = get_val_transforms(config['img_size'])

    for group_name, pid_list in TARGET_CASES.items():
        label_val = 0 if group_name.startswith('TB') else 1
        group_dir = os.path.join(config['output_dir'], group_name)
        os.makedirs(group_dir, exist_ok=True)

        for pid in pid_list:
            pid_str = str(pid)
            if pid_str not in id_to_fold:
                continue

            target_fold = id_to_fold[pid_str]
            if target_fold not in models:
                continue

            model = models[target_fold]
            cam = cams[target_fold]

            img_list = get_patient_image_list(
                config['data_dir'], pid, label_val, config['imgs_per_patient']
            )

            if not img_list:
                continue

            patient_dir = os.path.join(group_dir, str(pid))
            os.makedirs(patient_dir, exist_ok=True)

            for idx, img_path in enumerate(img_list):
                try:
                    rgb_img_raw = Image.open(img_path).convert('RGB')
                    rgb_img = transform(rgb_img_raw)
                    
                    img_tensor = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])(rgb_img).unsqueeze(0).to(device)
                    
                    rgb_img_np = np.float32(rgb_img) / 255

                    with torch.no_grad():
                        out = model(img_tensor)
                        prob = torch.softmax(out, dim=1)[0, 1].item()

                    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
                    visualization = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)

                    short_id = clean_filename(pid)
                    fname = f"{short_id}_seq{idx:03d}_F{target_fold}_P{prob:.2f}.jpg"
                    save_path = os.path.join(patient_dir, fname)

                    vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                    orig_bgr = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
                    combined = np.hstack([orig_bgr, vis_bgr])

                    save_image_safe(save_path, combined)

                except Exception:
                    pass


if __name__ == '__main__':
    config = get_config()
    main(config)
