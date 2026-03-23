"""
Interactive ROI cropping tool for ultrasound images.
"""

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from pathlib import Path
import json

from frame_extractor import cv2_imread_chinese, cv2_imwrite_chinese


class RoiSelector:
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.original_image = image.copy()
        self.image = image.copy()
        self.clone = image.copy()
        self.roi_pts = []
        self.drawing = False
        self.scale_factor = 1.0

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_pts = [(x, y)]
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.image = self.clone.copy()
            cv2.rectangle(self.image, self.roi_pts[0], (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and len(self.roi_pts) == 1:
                self.roi_pts.append((x, y))
                self.drawing = False
                self._finalize_roi()

    def _finalize_roi(self):
        p1, p2 = self.roi_pts
        x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
        x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
        self.roi = (x1, y1, x2 - x1, y2 - y1)
        self._redraw_roi()

    def _redraw_roi(self):
        self.image = self.clone.copy()
        x, y, w, h = self.roi
        cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(self.image, "Press ENTER to confirm, ESC to redraw", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def select(self):
        h, w = self.image.shape[:2]
        max_display_width = 1200
        
        if w > max_display_width:
            scale = max_display_width / w
            new_w = max_display_width
            new_h = int(h * scale)
            self.image = cv2.resize(self.image, (new_w, new_h))
            self.clone = self.image.copy()
            self.scale_factor = scale
        else:
            self.scale_factor = 1.0
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self.roi = None
        
        while True:
            display_img = self.image.copy()
            if self.roi is None:
                cv2.putText(display_img, "Draw ROI: Click and drag to select region", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(display_img, "Press ENTER to confirm, ESC to redraw", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 and self.roi is not None:
                break
            elif key == 27:
                self.roi = None
                self.roi_pts = []
                self.image = self.clone.copy()
        
        cv2.destroyWindow(self.window_name)
        
        if self.roi is None:
            return None
        
        x, y, w, h = self.roi
        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)
        orig_w = int(w / self.scale_factor)
        orig_h = int(h / self.scale_factor)
        
        orig_x = max(0, min(orig_x, self.original_image.shape[1] - 1))
        orig_y = max(0, min(orig_y, self.original_image.shape[0] - 1))
        orig_w = max(1, min(orig_w, self.original_image.shape[1] - orig_x))
        orig_h = max(1, min(orig_h, self.original_image.shape[0] - orig_y))
        
        return (orig_x, orig_y, orig_w, orig_h)


class InteractiveCropper:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('0x0+9999+9999')
        self.root.withdraw()
        self.output_root = None

    def _select_category(self):
        categories = ['结核性淋巴结', '转移性淋巴结']
        dialog = tk.Toplevel(self.root)
        dialog.title("Select lymph node type")
        dialog.geometry("300x150")
        
        var = tk.StringVar(value=categories[0])
        tk.Label(dialog, text="Select lymph node type:", font=('Arial', 10)).pack(pady=10)
        
        for cat in categories:
            tk.Radiobutton(dialog, text=cat, variable=var, value=cat).pack(anchor='w', padx=20)
        
        selection = None
        def on_ok():
            nonlocal selection
            selection = var.get()
            dialog.destroy()
        
        tk.Button(dialog, text="OK", command=on_ok).pack(pady=10)
        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)
        
        return selection

    def _select_frames_dir(self):
        messagebox.showinfo("Select directory", "Select directory with extracted frames")
        frames_dir = filedialog.askdirectory(title="Select frames directory")
        return frames_dir if frames_dir else None

    def _get_output_dir(self, category):
        if self.output_root is None:
            messagebox.showinfo("Select output", "Select output root directory")
            self.output_root = filedialog.askdirectory(title="Select output root directory")
            if not self.output_root:
                return None
        
        output_dir = os.path.join(self.output_root, category, 'b超')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _find_image_files(self, directory):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        images = []
        
        for file in sorted(os.listdir(directory)):
            if file.lower().endswith(image_extensions):
                images.append(os.path.join(directory, file))
        
        return images

    def _crop_images(self, images, roi, output_dir):
        if roi is None:
            return 0
        
        x, y, w, h = roi
        saved_count = 0
        
        for img_path in images:
            img = cv2_imread_chinese(img_path)
            if img is None:
                continue
            
            if x >= 0 and y >= 0 and x + w <= img.shape[1] and y + h <= img.shape[0]:
                crop_img = img[y:y+h, x:x+w]
                
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                save_name = f"{base_name}_crop.png"
                save_path = os.path.join(output_dir, save_name)
                
                if cv2_imwrite_chinese(save_path, crop_img):
                    saved_count += 1
        
        return saved_count

    def _save_roi_info(self, roi, output_dir):
        if roi is None:
            return
        
        roi_info = {
            "x": int(roi[0]),
            "y": int(roi[1]),
            "width": int(roi[2]),
            "height": int(roi[3])
        }
        
        json_path = os.path.join(output_dir, "roi_info.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(roi_info, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def run(self):
        while True:
            category = self._select_category()
            if not category:
                break
            
            frames_dir = self._select_frames_dir()
            if not frames_dir:
                break
            
            images = self._find_image_files(frames_dir)
            if not images:
                messagebox.showerror("Error", f"No images found in {frames_dir}")
                continue
            
            first_img = cv2_imread_chinese(images[0])
            if first_img is None:
                messagebox.showerror("Error", "Cannot read first image")
                continue
            
            selector = RoiSelector("Select ROI Region", first_img)
            roi = selector.select()
            
            if roi is None:
                messagebox.showwarning("Warning", "No valid ROI selected")
                continue
            
            output_dir = self._get_output_dir(category)
            if not output_dir:
                break
            
            saved_count = self._crop_images(images, roi, output_dir)
            self._save_roi_info(roi, output_dir)
            
            messagebox.showinfo("Complete", f"Processed {saved_count} images\nOutput: {output_dir}")
            
            if not messagebox.askyesno("Continue?", "Process next batch?"):
                break
        
        messagebox.showinfo("Complete", "All operations completed")
        self.root.destroy()


def main():
    cropper = InteractiveCropper()
    cropper.run()


if __name__ == '__main__':
    main()
