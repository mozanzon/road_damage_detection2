# 🚗 YOLOv11 Pothole & Crack Detection UI

A simple graphical user interface for detecting potholes and cracks in road images using your trained YOLOv11 model.

## 📋 Features

- **Easy Image Import**: Load any image file (JPG, PNG, BMP, TIFF)
- **Real-time Detection**: Detect potholes and cracks with adjustable confidence threshold
- **Visual Results**: Side-by-side comparison of original and detected images
- **Color-coded Detection**: 
  - 🔴 Red boxes = Potholes
  - 🟢 Green boxes = Cracks
- **Save Results**: Export detection results with bounding boxes
- **Confidence Adjustment**: Slider to control detection sensitivity (0.1 - 1.0)
- **Detection Summary**: See count of potholes and cracks detected

## 🚀 Quick Start

### 1. Install Required Libraries

```powershell
pip install ultralytics pillow opencv-python
```

### 2. Run the Application

```powershell
python pothole_detection_ui.py
```

Or simply double-click the `pothole_detection_ui.py` file.

## 📖 How to Use

1. **Launch the Application**
   - Run `python pothole_detection_ui.py`
   - The UI will automatically load your trained model (`best.pt`, `final_model.pt`, or `last.pt`)

2. **Import an Image**
   - Click the **"📁 Import Image"** button
   - Select an image file from your computer
   - The original image will appear on the left panel

3. **Adjust Confidence (Optional)**
   - Use the **Confidence slider** to set detection threshold (default: 0.5)
   - Lower values = more detections but may include false positives
   - Higher values = fewer but more confident detections

4. **Run Detection**
   - Click the **"🔍 Detect"** button
   - Wait for the model to process the image
   - Results will appear on the right panel with colored bounding boxes
   - Detection summary will show in the status bar

5. **Save Results**
   - Click the **"💾 Save Result"** button
   - Choose a location and filename
   - The annotated image will be saved

6. **Clear and Start Over**
   - Click the **"🗑️ Clear"** button to reset and load a new image

## 🎨 UI Layout

```
┌─────────────────────────────────────────────────────────┐
│           🚗 YOLOv11 Pothole & Crack Detection          │
├─────────────────────────────────────────────────────────┤
│ [Import] [Detect] [Save] [Clear]    Confidence: [====] │
├──────────────────────────┬──────────────────────────────┤
│   Original Image         │   Detection Result           │
│                          │                              │
│   [Your Image Here]      │   [Detected Image Here]      │
│                          │                              │
├──────────────────────────┴──────────────────────────────┤
│ Status: ✅ Detection complete! Found 5 objects...        │
└─────────────────────────────────────────────────────────┘
```

## 🎯 Detection Classes

- **Class 0 - Pothole** 🔴 (Red bounding box)
- **Class 1 - Crack** 🟢 (Green bounding box)

## ⚙️ Technical Details

- **Model**: YOLOv11 (nano/small/medium/large)
- **Framework**: Ultralytics
- **GUI**: Tkinter (built-in with Python)
- **Image Processing**: OpenCV, PIL
- **Supported Formats**: JPG, JPEG, PNG, BMP, TIFF

## 🔧 Troubleshooting

### Model Not Found
- Ensure `best.pt`, `final_model.pt`, or `last.pt` is in the same directory as the script
- The UI will show a warning if no model is found

### No Objects Detected
- Try lowering the confidence threshold using the slider
- Ensure the image contains potholes or cracks
- Check that the model was trained properly

### Import Error
- Make sure all dependencies are installed: `pip install ultralytics pillow opencv-python`

### Image Display Issues
- The UI automatically scales images to fit the window
- Try resizing the window for better visibility

## 📊 Model Files in Your Folder

Your folder contains these trained models:
- `best.pt` - Best performing model during training
- `last.pt` - Final epoch model
- `final_model.pt` - Manually saved final model
- `best.onnx` - ONNX format (for deployment)

The UI will automatically use the first available model in this priority order.

## 💡 Tips

1. **Best Results**: Use confidence threshold between 0.3 - 0.7
2. **Speed**: Lower resolution images process faster
3. **Accuracy**: Higher confidence = fewer false positives
4. **Batch Processing**: For multiple images, use the notebook or CLI instead

## 📝 Example Workflow

```powershell
# 1. Navigate to your folder
cd "c:\Users\Unilever\Downloads\New folder (2)\boda's_data"

# 2. Run the UI
python pothole_detection_ui.py

# 3. Use the GUI to:
#    - Import test images
#    - Adjust confidence
#    - Run detection
#    - Save results
```

## 🎓 Model Training Results

Based on your `results.csv`:
- **mAP50**: ~0.65 (65% accuracy at 50% IoU)
- **mAP50-95**: ~0.38 (38% average accuracy)
- **Precision**: ~0.72 (72% correct predictions)
- **Recall**: ~0.61 (61% detection rate)

These are good results for pothole/crack detection!

## 📞 Support

If you encounter issues:
1. Check the status bar for error messages
2. Ensure all dependencies are installed
3. Verify your model files exist
4. Try a different image

---

**Enjoy detecting potholes and cracks! 🚗💨**
