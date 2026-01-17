"""
YOLOv11 Pothole & Crack Detection for Raspberry Pi 5
ONNX Runtime implementation for efficient inference on Raspberry Pi hardware
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
import argparse


class PotholeDetectorONNX:
    """
    ONNX-based pothole and crack detector optimized for Raspberry Pi 5
    """
    
    def __init__(self, model_path='best.onnx', conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize the detector with ONNX model
        
        Args:
            model_path: Path to the ONNX model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = ['Pothole', 'Crack']
        self.colors = {
            0: (0, 0, 255),    # Red (BGR) for Pothole
            1: (0, 255, 0)     # Green (BGR) for Crack
        }
        
        # Initialize ONNX Runtime session with optimizations for Raspberry Pi
        print(f"Loading ONNX model from: {model_path}")
        
        # Use CPU execution provider optimized for ARM
        providers = ['CPUExecutionProvider']
        
        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Raspberry Pi 5 has 4 cores
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {self.input_shape}")
        print(f"Input name: {self.input_name}")
        
    def preprocess(self, image):
        """
        Preprocess image for YOLO model
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image tensor and scale factors
        """
        # Get original image dimensions
        img_height, img_width = image.shape[:2]
        
        # Resize image to model input size with padding
        # Calculate scale to fit within input size
        scale = min(self.input_width / img_width, self.input_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded_img = np.ones((self.input_height, self.input_width, 3), dtype=np.uint8) * 114
        
        # Calculate padding
        pad_left = (self.input_width - new_width) // 2
        pad_top = (self.input_height - new_height) // 2
        
        # Place resized image in center
        padded_img[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = resized_img
        
        # Convert to RGB and normalize
        padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and transpose to CHW format
        input_tensor = padded_img.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)   # Add batch dimension
        
        return input_tensor, scale, pad_left, pad_top
    
    def postprocess(self, outputs, scale, pad_left, pad_top, orig_width, orig_height):
        """
        Postprocess model outputs to get bounding boxes
        
        Args:
            outputs: Model output
            scale: Scale factor used in preprocessing
            pad_left: Left padding used in preprocessing
            pad_top: Top padding used in preprocessing
            orig_width: Original image width
            orig_height: Original image height
            
        Returns:
            List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        output = outputs[0]
        
        # YOLO output format: [batch, num_classes + 4, num_boxes] or [batch, num_boxes, num_classes + 4]
        # Handle both formats
        if len(output.shape) == 3:
            if output.shape[1] > output.shape[2]:
                output = np.transpose(output[0], (1, 0))  # Transpose to [num_boxes, features]
            else:
                output = output[0]
        
        detections = []
        
        for detection in output:
            # Extract box coordinates and scores
            if len(detection) >= 6:  # [x, y, w, h, conf, class_scores...]
                x, y, w, h = detection[:4]
                confidence = detection[4]
                class_scores = detection[5:]
            else:  # Alternative format [x, y, w, h, class_scores...]
                x, y, w, h = detection[:4]
                class_scores = detection[4:]
                confidence = np.max(class_scores)
            
            class_id = np.argmax(class_scores)
            
            # Filter by confidence
            if confidence >= self.conf_threshold:
                # Convert from center format to corner format
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                
                # Scale back to original image coordinates
                x1 = (x1 - pad_left) / scale
                y1 = (y1 - pad_top) / scale
                x2 = (x2 - pad_left) / scale
                y2 = (y2 - pad_top) / scale
                
                # Clip to image boundaries
                x1 = max(0, min(x1, orig_width))
                y1 = max(0, min(y1, orig_height))
                x2 = max(0, min(x2, orig_width))
                y2 = max(0, min(y2, orig_height))
                
                detections.append([x1, y1, x2, y2, confidence, class_id])
        
        # Apply Non-Maximum Suppression
        if len(detections) > 0:
            detections = self.non_max_suppression(detections)
        
        return detections
    
    def non_max_suppression(self, detections):
        """
        Apply Non-Maximum Suppression to remove overlapping boxes
        
        Args:
            detections: List of detections
            
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []
        
        detections = np.array(detections)
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]
        
        return detections[keep].tolist()
    
    def detect(self, image):
        """
        Run detection on an image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of detections and inference time
        """
        orig_height, orig_width = image.shape[:2]
        
        # Preprocess
        start_time = time.time()
        input_tensor, scale, pad_left, pad_top = self.preprocess(image)
        preprocess_time = time.time() - start_time
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        
        # Postprocess
        start_time = time.time()
        detections = self.postprocess(outputs, scale, pad_left, pad_top, orig_width, orig_height)
        postprocess_time = time.time() - start_time
        
        total_time = preprocess_time + inference_time + postprocess_time
        
        return detections, {
            'total': total_time,
            'preprocess': preprocess_time,
            'inference': inference_time,
            'postprocess': postprocess_time
        }
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det
            class_id = int(class_id)
            
            # Get color and class name
            color = self.colors.get(class_id, (255, 255, 255))
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
            
            # Draw bounding box
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_image, (int(x1), int(y1) - label_height - 10), 
                         (int(x1) + label_width, int(y1)), color, -1)
            
            # Draw label text
            cv2.putText(result_image, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image


def detect_from_image(detector, image_path, output_path=None, show=False):
    """
    Run detection on a single image
    
    Args:
        detector: PotholeDetectorONNX instance
        image_path: Path to input image
        output_path: Path to save output image (optional)
        show: Whether to display the image (not recommended for headless Pi)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    print(f"\nProcessing image: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Run detection
    detections, timing = detector.detect(image)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Detection Results:")
    print(f"{'='*50}")
    print(f"Total detections: {len(detections)}")
    print(f"\nTiming breakdown:")
    print(f"  Preprocessing:  {timing['preprocess']*1000:.2f} ms")
    print(f"  Inference:      {timing['inference']*1000:.2f} ms")
    print(f"  Postprocessing: {timing['postprocess']*1000:.2f} ms")
    print(f"  Total:          {timing['total']*1000:.2f} ms")
    print(f"  FPS:            {1.0/timing['total']:.2f}")
    
    if detections:
        print(f"\nDetected objects:")
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2, confidence, class_id = det
            class_name = detector.class_names[int(class_id)]
            print(f"  {i}. {class_name}: {confidence:.2%} at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    else:
        print("\nNo objects detected above confidence threshold.")
    
    # Draw detections
    result_image = detector.draw_detections(image, detections)
    
    # Save output
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"\nOutput saved to: {output_path}")
    
    # Display (only if not headless)
    if show:
        cv2.imshow('Detection Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_image, detections


def detect_from_camera(detector, camera_index=0, width=640, height=480, save_video=None):
    """
    Run real-time detection from camera
    
    Args:
        detector: PotholeDetectorONNX instance
        camera_index: Camera device index (0 for default)
        width: Camera frame width
        height: Camera frame height
        save_video: Path to save output video (optional)
    """
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    print(f"\nStarting camera detection (Press 'q' to quit, 's' to save frame)")
    print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Video writer setup
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        video_writer = cv2.VideoWriter(save_video, fourcc, fps, (width, height))
        print(f"Recording to: {save_video}")
    
    frame_count = 0
    fps_list = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame_count += 1
            
            # Run detection
            detections, timing = detector.detect(frame)
            
            # Draw detections
            result_frame = detector.draw_detections(frame, detections)
            
            # Calculate FPS
            fps = 1.0 / timing['total']
            fps_list.append(fps)
            avg_fps = np.mean(fps_list[-30:])  # Average over last 30 frames
            
            # Draw FPS and detection info
            info_text = f"FPS: {avg_fps:.1f} | Detections: {len(detections)}"
            cv2.putText(result_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save to video if enabled
            if video_writer:
                video_writer.write(result_frame)
            
            # Display frame
            cv2.imshow('Pothole Detection', result_frame)
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"Frame saved: {filename}")
            
            # Print stats every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {avg_fps:.1f} FPS, {len(detections)} detections")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nCamera session ended.")
        print(f"Total frames: {frame_count}")
        if fps_list:
            print(f"Average FPS: {np.mean(fps_list):.2f}")


def main():
    """
    Main function with command-line interface
    """
    parser = argparse.ArgumentParser(description='YOLOv11 Pothole Detection for Raspberry Pi 5')
    parser.add_argument('--model', type=str, default='best.onnx',
                       help='Path to ONNX model file')
    parser.add_argument('--source', type=str, default=None,
                       help='Image file path or camera index (0 for default camera)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (for images) or video path (for camera)')
    parser.add_argument('--show', action='store_true',
                       help='Display results (disable for headless mode)')
    parser.add_argument('--width', type=int, default=640,
                       help='Camera frame width')
    parser.add_argument('--height', type=int, default=480,
                       help='Camera frame height')
    
    args = parser.parse_args()
    
    # Initialize detector
    print("\n" + "="*60)
    print("YOLOv11 Pothole & Crack Detection for Raspberry Pi 5")
    print("="*60)
    
    detector = PotholeDetectorONNX(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run detection based on source
    if args.source is None:
        print("\nNo source specified. Use --source <image_path> or --source 0 for camera")
        print("Example: python raspberry_pi_detection.py --source test_image.jpg")
        print("Example: python raspberry_pi_detection.py --source 0")
    elif args.source.isdigit():
        # Camera mode
        camera_index = int(args.source)
        detect_from_camera(
            detector,
            camera_index=camera_index,
            width=args.width,
            height=args.height,
            save_video=args.output
        )
    else:
        # Image mode
        if not Path(args.source).exists():
            print(f"Error: Image file not found: {args.source}")
            return
        
        output_path = args.output
        if output_path is None:
            # Generate output filename
            input_path = Path(args.source)
            output_path = str(input_path.parent / f"{input_path.stem}_detected{input_path.suffix}")
        
        detect_from_image(detector, args.source, output_path, args.show)


if __name__ == '__main__':
    main()
