"""
Frame extraction from video and DICOM files with SSIM deduplication.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from skimage.metrics import structural_similarity as ssim


def cv2_imread_chinese(filepath):
    """Read image with Chinese path support."""
    try:
        n = np.fromfile(filepath, np.uint8)
        img = cv2.imdecode(n, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode: {filepath}")
        return img
    except Exception as e:
        return None


def cv2_imwrite_chinese(filepath, img):
    """Write image with Chinese path support."""
    try:
        Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
        ret, buf = cv2.imencode('.png', img)
        if ret:
            with open(filepath, 'wb') as f:
                f.write(buf)
            return True
        return False
    except Exception:
        return False


def calculate_ssim(img1, img2):
    """Calculate structural similarity index between two images."""
    try:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        return ssim(gray1, gray2)
    except Exception:
        return 1.0


def extract_frames_from_video(video_path, output_dir, ssim_threshold=0.95, frame_interval=2):
    """Extract frames from video with SSIM deduplication."""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps if fps > 0 else 0
    
    if duration > 60:
        frame_interval = 5
    elif duration > 30:
        frame_interval = 3
    
    frame_count = 0
    saved_count = 0
    last_saved_frame = None
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            break
        
        if frame_count % frame_interval == 0:
            if last_saved_frame is not None:
                similarity = calculate_ssim(frame, last_saved_frame)
                if similarity > ssim_threshold:
                    frame_count += 1
                    continue
            
            save_name = f"{base_name}_frame_{saved_count:04d}.png"
            save_path = os.path.join(output_dir, save_name)
            if cv2_imwrite_chinese(save_path, frame):
                saved_count += 1
                last_saved_frame = frame.copy()
        
        frame_count += 1
    
    cap.release()
    return saved_count


def extract_frames_from_dicom(dicom_path, output_dir, ssim_threshold=0.95):
    """Extract frames from DICOM file with SSIM deduplication."""
    try:
        import pydicom
    except ImportError:
        return 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(dicom_path, 'rb') as f:
            ds = pydicom.dcmread(f)
        
        if 'PixelData' not in ds:
            return 0
        
        arr = ds.pixel_array
        
        photometric = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
        if 'YBR' in photometric:
            try:
                from pydicom.pixel_data_handlers.util import convert_color_space
                arr = convert_color_space(arr, photometric, 'RGB')
            except:
                pass
        
        def convert_to_bgr(img):
            if img.dtype != np.uint8:
                img = img.astype(float)
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        
        frames = []
        if arr.ndim == 4:
            for i in range(arr.shape[0]):
                frames.append(convert_to_bgr(arr[i]))
        elif arr.ndim == 3:
            num_frames = getattr(ds, 'NumberOfFrames', 1)
            if num_frames > 1 and arr.shape[0] == num_frames:
                for i in range(arr.shape[0]):
                    frames.append(convert_to_bgr(arr[i]))
            else:
                frames.append(convert_to_bgr(arr))
        elif arr.ndim == 2:
            frames.append(convert_to_bgr(arr))
        
        saved_count = 0
        last_saved_frame = None
        base_name = os.path.splitext(os.path.basename(dicom_path))[0]
        
        for i, frame in enumerate(frames):
            if last_saved_frame is not None:
                similarity = calculate_ssim(frame, last_saved_frame)
                if similarity > ssim_threshold:
                    continue
            
            save_name = f"{base_name}_frame_{saved_count:04d}.png"
            save_path = os.path.join(output_dir, save_name)
            if cv2_imwrite_chinese(save_path, frame):
                saved_count += 1
                last_saved_frame = frame.copy()
        
        return saved_count
    
    except Exception:
        return 0
