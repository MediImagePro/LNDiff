"""
Ultrasound image dataset module.
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class UltrasoundDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        assert len(image_paths) == len(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        return img, label


def find_all_patients(root_dir):
    """Find all patients and their images from directory structure."""
    class_map = {'结核性淋巴结': 0, '转移性淋巴结': 1}
    all_patients = []

    for class_folder, label in class_map.items():
        class_path = os.path.join(root_dir, class_folder)
        if not os.path.exists(class_path):
            continue

        patient_dirs = [d for d in os.listdir(class_path) 
                       if os.path.isdir(os.path.join(class_path, d))]
        patient_dirs.sort()

        for p_dir in patient_dirs:
            p_path = os.path.join(class_path, p_dir)
            
            b_mode_path = os.path.join(p_path, 'b超')
            if not os.path.exists(b_mode_path):
                b_mode_path = os.path.join(p_path, 'B超')
            if not os.path.exists(b_mode_path):
                b_mode_path = p_path

            images = glob.glob(os.path.join(b_mode_path, "*.[jpJP][pnPN]*"))
            images += glob.glob(os.path.join(b_mode_path, "*.bmp"))
            images.sort()

            if len(images) > 0:
                all_patients.append({
                    'id': p_dir,
                    'label': label,
                    'images': images
                })

    return all_patients
