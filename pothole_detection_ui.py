"""
YOLOv11 Pothole & Crack Detection UI
A simple graphical interface for detecting potholes and cracks in images
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
import os

# Try to import YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO


class PotholeDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv11 Pothole & Crack Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Model variables
        self.model = None
        self.current_image_path = None
        self.original_image = None
        self.detected_image = None
        
        # Class names and colors
        self.class_names = ['Pothole', 'Crack']
        self.colors = {
            0: (255, 0, 0),    # Red for Pothole
            1: (0, 255, 0)     # Green for Crack
        }
        
        # Detection settings
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#1e1e1e', height=60)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = tk.Label(
            title_frame, 
            text="🚗 YOLOv11 Pothole & Crack Detection", 
            font=('Arial', 20, 'bold'),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        title_label.pack(pady=10)
        
        # Control Panel
        control_frame = tk.Frame(self.root, bg='#2b2b2b')
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Left controls
        left_controls = tk.Frame(control_frame, bg='#2b2b2b')
        left_controls.pack(side=tk.LEFT, padx=5)
        
        # Import button
        self.import_btn = tk.Button(
            left_controls,
            text="📁 Import Image",
            command=self.import_image,
            font=('Arial', 12, 'bold'),
            bg='#0078d4',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.FLAT
        )
        self.import_btn.pack(side=tk.LEFT, padx=5)
        
        # Detect button
        self.detect_btn = tk.Button(
            left_controls,
            text="🔍 Detect",
            command=self.run_detection,
            font=('Arial', 12, 'bold'),
            bg='#28a745',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        # Save button
        self.save_btn = tk.Button(
            left_controls,
            text="💾 Save Result",
            command=self.save_result,
            font=('Arial', 12, 'bold'),
            bg='#6c757d',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_btn = tk.Button(
            left_controls,
            text="🗑️ Clear",
            command=self.clear_all,
            font=('Arial', 12, 'bold'),
            bg='#dc3545',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            relief=tk.FLAT
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Right controls - Confidence slider
        right_controls = tk.Frame(control_frame, bg='#2b2b2b')
        right_controls.pack(side=tk.RIGHT, padx=5)
        
        conf_label = tk.Label(
            right_controls,
            text="Confidence:",
            font=('Arial', 10),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        conf_label.pack(side=tk.LEFT, padx=5)
        
        self.conf_slider = tk.Scale(
            right_controls,
            from_=0.1,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.confidence_threshold,
            length=200,
            bg='#3b3b3b',
            fg='#ffffff',
            highlightthickness=0,
            troughcolor='#1e1e1e'
        )
        self.conf_slider.pack(side=tk.LEFT, padx=5)
        
        self.conf_value_label = tk.Label(
            right_controls,
            textvariable=self.confidence_threshold,
            font=('Arial', 10, 'bold'),
            bg='#2b2b2b',
            fg='#00ff00',
            width=5
        )
        self.conf_value_label.pack(side=tk.LEFT, padx=5)
        
        # Image display area
        display_frame = tk.Frame(self.root, bg='#2b2b2b')
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Original image
        original_frame = tk.LabelFrame(
            display_frame,
            text="Original Image",
            font=('Arial', 12, 'bold'),
            bg='#1e1e1e',
            fg='#ffffff',
            relief=tk.RIDGE,
            borderwidth=2
        )
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_canvas = tk.Canvas(
            original_frame,
            bg='#1e1e1e',
            highlightthickness=0
        )
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Detected image
        detected_frame = tk.LabelFrame(
            display_frame,
            text="Detection Result",
            font=('Arial', 12, 'bold'),
            bg='#1e1e1e',
            fg='#ffffff',
            relief=tk.RIDGE,
            borderwidth=2
        )
        detected_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.detected_canvas = tk.Canvas(
            detected_frame,
            bg='#1e1e1e',
            highlightthickness=0
        )
        self.detected_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#1e1e1e', height=100)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.status_text = tk.Text(
            status_frame,
            height=4,
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='#00ff00',
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_status("✅ Application started. Click 'Import Image' to begin.")
    
    def load_model(self):
        """Load the YOLOv11 model"""
        try:
            # Try to find the model file
            possible_models = ['best.pt', 'final_model.pt', 'last.pt']
            model_path = None
            
            for model_name in possible_models:
                if os.path.exists(model_name):
                    model_path = model_name
                    break
            
            if model_path:
                self.model = YOLO(model_path)
                self.log_status(f"✅ Model loaded successfully: {model_path}")
            else:
                messagebox.showwarning(
                    "Model Not Found",
                    "No trained model found. Please ensure best.pt, final_model.pt, or last.pt exists in the current directory."
                )
                self.log_status("⚠️ No trained model found!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.log_status(f"❌ Error loading model: {str(e)}")
    
    def log_status(self, message):
        """Log a message to the status bar"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
    
    def import_image(self):
        """Import an image for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load image
                self.current_image_path = file_path
                self.original_image = cv2.imread(file_path)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                
                # Display original image
                self.display_image(self.original_image, self.original_canvas)
                
                # Clear previous detection
                self.detected_canvas.delete("all")
                self.detected_image = None
                
                # Enable detect button
                self.detect_btn.config(state=tk.NORMAL)
                self.save_btn.config(state=tk.DISABLED)
                
                self.log_status(f"✅ Image loaded: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.log_status(f"❌ Error loading image: {str(e)}")
    
    def display_image(self, img, canvas):
        """Display image on canvas with proper scaling"""
        # Get canvas size
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Calculate scaling to fit canvas
        img_height, img_width = img.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(resized_img)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Display on canvas
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            anchor=tk.CENTER,
            image=img_tk
        )
        canvas.image = img_tk  # Keep a reference
    
    def run_detection(self):
        """Run YOLOv11 detection on the loaded image"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded!")
            return
        
        try:
            self.log_status("🔍 Running detection...")
            
            # Run detection
            conf_threshold = self.confidence_threshold.get()
            results = self.model(self.original_image, conf=conf_threshold, verbose=False)
            
            # Draw detections
            self.detected_image = self.original_image.copy()
            detection_count = {'Pothole': 0, 'Crack': 0}
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Count detections
                        class_name = self.class_names[class_id]
                        detection_count[class_name] += 1
                        
                        # Draw bounding box
                        color = self.colors[class_id]
                        cv2.rectangle(
                            self.detected_image,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color,
                            3
                        )
                        
                        # Draw label background
                        label = f'{class_name}: {confidence:.2f}'
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            2
                        )
                        
                        cv2.rectangle(
                            self.detected_image,
                            (int(x1), int(y1) - label_height - 10),
                            (int(x1) + label_width, int(y1)),
                            color,
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            self.detected_image,
                            label,
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2
                        )
            
            # Display result
            self.display_image(self.detected_image, self.detected_canvas)
            
            # Enable save button
            self.save_btn.config(state=tk.NORMAL)
            
            # Log results
            total_detections = sum(detection_count.values())
            if total_detections > 0:
                self.log_status(
                    f"✅ Detection complete! Found {total_detections} objects: "
                    f"{detection_count['Pothole']} Potholes, {detection_count['Crack']} Cracks"
                )
            else:
                self.log_status("ℹ️ No objects detected. Try lowering the confidence threshold.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.log_status(f"❌ Detection error: {str(e)}")
    
    def save_result(self):
        """Save the detection result"""
        if self.detected_image is None:
            messagebox.showerror("Error", "No detection result to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Detection Result",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Convert RGB to BGR for cv2.imwrite
                img_to_save = cv2.cvtColor(self.detected_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, img_to_save)
                self.log_status(f"✅ Result saved: {Path(file_path).name}")
                messagebox.showinfo("Success", "Detection result saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
                self.log_status(f"❌ Error saving image: {str(e)}")
    
    def clear_all(self):
        """Clear all images and reset"""
        self.original_canvas.delete("all")
        self.detected_canvas.delete("all")
        self.current_image_path = None
        self.original_image = None
        self.detected_image = None
        self.detect_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.log_status("🗑️ Cleared all images")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = PotholeDetectionUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
