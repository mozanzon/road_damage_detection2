# YOLOv11 Pothole Detection for Raspberry Pi 5

This script provides optimized pothole and crack detection using ONNX Runtime on Raspberry Pi 5.

## 🎯 Features

- **ONNX Runtime** for efficient inference on Raspberry Pi 5
- **Real-time camera detection** with FPS monitoring
- **Image detection** with visualization
- **Optimized for ARM architecture** (4-core CPU utilization)
- **Low latency** inference suitable for edge deployment
- **Headless mode** support for deployment without display

## 📋 Requirements

### Hardware
- Raspberry Pi 5 (4GB or 8GB RAM recommended)
- USB Camera or Raspberry Pi Camera Module (optional, for real-time detection)
- MicroSD card (32GB+ recommended)

### Software
- Raspberry Pi OS (64-bit recommended)
- Python 3.8 or higher

## 🚀 Installation

### 1. Update System
```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install System Dependencies
```bash
# Install OpenCV dependencies
sudo apt install -y python3-pip python3-opencv libopencv-dev

# Install camera support (if using Pi Camera)
sudo apt install -y python3-picamera2

# Install additional libraries
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev \
    libjasper-dev libqtgui4 libqt4-test
```

### 3. Install Python Packages
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install numpy opencv-python onnxruntime

# If onnxruntime installation fails, try:
pip install onnxruntime --no-cache-dir
```

### 4. Copy Model File
Make sure `best.onnx` is in the same directory as the script, or specify the path with `--model` argument.

```bash
# Copy the model to your Raspberry Pi (if not already there)
# From your computer:
scp best.onnx pi@raspberrypi:~/pothole_detection/

# Or download from your storage
wget <your-model-url> -O best.onnx
```

## 💻 Usage

### Basic Image Detection
```bash
python raspberry_pi_detection.py --source test_image.jpg
```

### Image Detection with Custom Confidence
```bash
python raspberry_pi_detection.py --source test_image.jpg --conf 0.6 --output result.jpg
```

### Real-time Camera Detection
```bash
# USB Camera (default)
python raspberry_pi_detection.py --source 0

# With custom resolution
python raspberry_pi_detection.py --source 0 --width 1280 --height 720

# Save video output
python raspberry_pi_detection.py --source 0 --output detection_video.mp4
```

### Headless Mode (No Display)
```bash
# Process image without display
python raspberry_pi_detection.py --source image.jpg --output result.jpg

# Camera with video recording (no window)
python raspberry_pi_detection.py --source 0 --output video.mp4
```

## 🎛️ Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `best.onnx` | Path to ONNX model file |
| `--source` | str | None | Image path or camera index (0 for default) |
| `--conf` | float | 0.5 | Confidence threshold (0.0-1.0) |
| `--iou` | float | 0.45 | IoU threshold for NMS |
| `--output` | str | None | Output file path |
| `--show` | flag | False | Display results window |
| `--width` | int | 640 | Camera frame width |
| `--height` | int | 480 | Camera frame height |

## 📊 Performance

Expected performance on Raspberry Pi 5:

| Resolution | Model Input | FPS (approx) | Latency |
|------------|-------------|--------------|---------|
| 640x480 | 640x640 | 8-12 FPS | 80-120 ms |
| 1280x720 | 640x640 | 6-10 FPS | 100-160 ms |

*Performance may vary based on:*
- Number of detections in frame
- Raspberry Pi temperature/throttling
- Other running processes
- Camera interface (USB vs. CSI)

## 🎥 Camera Controls (During Detection)

When running camera detection:
- Press `q` - Quit and close
- Press `s` - Save current frame as JPEG

## 📝 Examples

### Example 1: Batch Process Images
```bash
# Process multiple images in a directory
for img in images/*.jpg; do
    python raspberry_pi_detection.py --source "$img" --conf 0.6
done
```

### Example 2: Auto-start on Boot
Create a systemd service to run detection on boot:

```bash
# Create service file
sudo nano /etc/systemd/system/pothole-detection.service
```

Add this content:
```ini
[Unit]
Description=Pothole Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/pothole_detection
ExecStart=/home/pi/pothole_detection/venv/bin/python raspberry_pi_detection.py --source 0 --output /home/pi/detections/video.mp4
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable pothole-detection.service
sudo systemctl start pothole-detection.service
```

### Example 3: Remote Access with Flask
Create a simple web interface to view detections remotely:

```python
# flask_viewer.py
from flask import Flask, Response
import cv2

app = Flask(__name__)
detector = PotholeDetectorONNX('best.onnx')

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        detections, _ = detector.detect(frame)
        result = detector.draw_detections(frame, detections)
        
        _, buffer = cv2.imencode('.jpg', result)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Run with:
```bash
pip install flask
python flask_viewer.py
# Access from browser: http://raspberrypi.local:5000/video
```

## 🔧 Troubleshooting

### Issue: Low FPS / High Latency
**Solutions:**
- Reduce camera resolution: `--width 640 --height 480`
- Increase confidence threshold: `--conf 0.6`
- Ensure Raspberry Pi is not throttling (check temperature)
- Close other applications
- Use heatsink or active cooling

### Issue: Camera Not Detected
**Solutions:**
```bash
# Check available cameras
ls /dev/video*

# Test camera
libcamera-hello

# For USB camera, try different index
python raspberry_pi_detection.py --source 1
```

### Issue: ONNX Runtime Import Error
**Solutions:**
```bash
# Reinstall with no cache
pip uninstall onnxruntime
pip install onnxruntime --no-cache-dir

# Or install specific version
pip install onnxruntime==1.16.0
```

### Issue: Out of Memory
**Solutions:**
- Close other applications
- Reduce camera resolution
- Disable desktop environment (run in terminal only)
- Use swap file:
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## 🌡️ Monitoring Performance

### Check Temperature
```bash
# Check CPU temperature
vcgencmd measure_temp

# Monitor continuously
watch -n 1 vcgencmd measure_temp
```

### Check CPU Usage
```bash
htop
```

### Test Model Performance
```bash
# Time a single detection
time python raspberry_pi_detection.py --source test.jpg
```

## 📦 Model Information

- **Model Type**: YOLOv11 Nano
- **Input Size**: 640x640 pixels
- **Classes**: 2 (Pothole, Crack)
- **Model Size**: ~10 MB
- **Format**: ONNX

## 🔗 Integration Examples

### Use as Python Module
```python
from raspberry_pi_detection import PotholeDetectorONNX

# Initialize detector
detector = PotholeDetectorONNX('best.onnx', conf_threshold=0.5)

# Detect from image
import cv2
image = cv2.imread('test.jpg')
detections, timing = detector.detect(image)

# Print results
for det in detections:
    x1, y1, x2, y2, conf, class_id = det
    print(f"Detected: {detector.class_names[int(class_id)]} ({conf:.2%})")
```

### Log Detections to File
```python
import json
from datetime import datetime

# Run detection
detections, timing = detector.detect(image)

# Save to JSON
log_entry = {
    'timestamp': datetime.now().isoformat(),
    'detections': [
        {
            'class': detector.class_names[int(det[5])],
            'confidence': float(det[4]),
            'bbox': [float(x) for x in det[:4]]
        }
        for det in detections
    ],
    'inference_time_ms': timing['inference'] * 1000
}

with open('detections_log.json', 'a') as f:
    f.write(json.dumps(log_entry) + '\n')
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Check Raspberry Pi system logs: `sudo journalctl -xe`

## 📄 License

This script is provided as-is for the YOLOv11 pothole detection model.

---

**Happy detecting! 🚗💨**
