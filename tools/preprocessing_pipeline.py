"""
Data preprocessing pipeline with frame extraction and ROI cropping.
"""

import os
import tkinter as tk
from tkinter import messagebox, filedialog
from frame_extractor import extract_frames_from_video, extract_frames_from_dicom
from roi_cropper import InteractiveCropper


class PreprocessingPipeline:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LNDiff Data Preprocessing")
        self.root.geometry("400x300")
        self._center_window()
        self._create_ui()

    def _center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _create_ui(self):
        title_label = tk.Label(
            self.root,
            text="LNDiff Data Preprocessing",
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=20)
        
        desc_label = tk.Label(
            self.root,
            text="Select operation:",
            font=('Arial', 10)
        )
        desc_label.pack(pady=10)
        
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        extract_btn = tk.Button(
            button_frame,
            text="1. Extract Frames",
            command=self._extract_frames,
            width=20,
            height=2,
            font=('Arial', 10)
        )
        extract_btn.pack(pady=10)
        
        crop_btn = tk.Button(
            button_frame,
            text="2. Crop ROI",
            command=self._crop_roi,
            width=20,
            height=2,
            font=('Arial', 10)
        )
        crop_btn.pack(pady=10)
        
        pipeline_btn = tk.Button(
            button_frame,
            text="3. Full Pipeline",
            command=self._run_full_pipeline,
            width=20,
            height=2,
            font=('Arial', 10)
        )
        pipeline_btn.pack(pady=10)
        
        exit_btn = tk.Button(
            self.root,
            text="Exit",
            command=self.root.quit,
            width=10,
            font=('Arial', 10)
        )
        exit_btn.pack(pady=20)

    def _extract_frames(self):
        self.root.withdraw()
        
        try:
            file_path = filedialog.askopenfilename(
                title="Select video or DICOM file",
                filetypes=[
                    ("Video files", "*.avi *.mp4"),
                    ("DICOM files", "*.dcm"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                self.root.deiconify()
                return
            
            output_dir = filedialog.askdirectory(title="Select output directory")
            if not output_dir:
                self.root.deiconify()
                return
            
            if file_path.lower().endswith(('.avi', '.mp4')):
                extract_frames_from_video(file_path, output_dir)
            elif file_path.lower().endswith('.dcm'):
                extract_frames_from_dicom(file_path, output_dir)
            
            messagebox.showinfo("Complete", f"Frame extraction complete!\nOutput: {output_dir}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Frame extraction failed: {str(e)}")
        
        finally:
            self.root.deiconify()

    def _crop_roi(self):
        self.root.withdraw()
        
        try:
            cropper = InteractiveCropper()
            cropper.run()
        except Exception as e:
            messagebox.showerror("Error", f"Cropping failed: {str(e)}")
        
        finally:
            self.root.deiconify()

    def _run_full_pipeline(self):
        self.root.withdraw()
        
        try:
            messagebox.showinfo("Step 1", "Select video or DICOM file for frame extraction")
            
            file_path = filedialog.askopenfilename(
                title="Select video or DICOM file",
                filetypes=[
                    ("Video files", "*.avi *.mp4"),
                    ("DICOM files", "*.dcm"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                self.root.deiconify()
                return
            
            temp_frames_dir = os.path.join(
                os.path.dirname(file_path),
                "temp_frames"
            )
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            if file_path.lower().endswith(('.avi', '.mp4')):
                extract_frames_from_video(file_path, temp_frames_dir)
            elif file_path.lower().endswith('.dcm'):
                extract_frames_from_dicom(file_path, temp_frames_dir)
            
            messagebox.showinfo("Step 1 Complete", f"Frame extraction complete!\nTemp dir: {temp_frames_dir}")
            
            messagebox.showinfo("Step 2", "Now perform ROI cropping")
            cropper = InteractiveCropper()
            cropper.run()
            
            messagebox.showinfo("Complete", "Full pipeline completed!")
        
        except Exception as e:
            messagebox.showerror("Error", f"Pipeline failed: {str(e)}")
        
        finally:
            self.root.deiconify()

    def run(self):
        self.root.mainloop()


def main():
    app = PreprocessingPipeline()
    app.run()


if __name__ == '__main__':
    main()
