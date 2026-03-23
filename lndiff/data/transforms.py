"""
Data transforms and augmentation.
"""

from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import cv2

class ResizeKeepRatio:
    """Resize image while maintaining aspect ratio with black padding."""
    
    def __init__(self, size=224):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img = img.resize((new_w, new_h), Image.BICUBIC)
        new_img = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        paste_x = (self.size - new_w) // 2
        paste_y = (self.size - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img

class EnhanceContrast:
    """CLAHE contrast enhancement for ultrasound images."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_np = np.array(img)

        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(img_enhanced)


def get_train_transforms(img_size=224):
    """Training transforms with augmentation."""
    return transforms.Compose([
        ResizeKeepRatio(img_size),
        #EnhanceContrast(clip_limit=3.0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        #transforms.RandomAffine(degrees=15,translate=(0.1,0.1), scale=（0.9,1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3)
    ])


def get_val_transforms(img_size=224):
    """Validation transforms without augmentation."""
    return transforms.Compose([
        ResizeKeepRatio(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

