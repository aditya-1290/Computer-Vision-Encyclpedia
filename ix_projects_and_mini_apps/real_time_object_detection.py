"""
Real-time Object Detection: Detect objects in webcam feed using YOLO.

Implementation uses OpenCV and YOLOv3 for real-time detection.

Theory:
- YOLO: Single-shot detector that predicts bounding boxes and class probabilities.
- Real-time: Process video frames at high FPS.

Math: YOLO divides image into grid, predicts B boxes per cell with confidence scores.

Reference:
- Redmon et al., YOLOv3: An Incremental Improvement, arXiv 2018
"""

import cv2
import numpy as np
import urllib.request
import os
import time

def download_yolo_files(config_url, weights_url, classes_url, config_path, weights_path, classes_path):
    """Download YOLO files if they don't exist"""
    if not os.path.exists(config_path):
        print(f"Downloading config file from {config_url}")
        urllib.request.urlretrieve(config_url, config_path)
        
    if not os.path.exists(weights_path):
        print(f"Downloading weights file from {weights_url}")
        urllib.request.urlretrieve(weights_url, weights_path)
        
    if not os.path.exists(classes_path):
        print(f"Downloading classes file from {classes_url}")
        urllib.request.urlretrieve(classes_url, classes_path)

def load_yolo_model(config_path, weights_path, classes_path, use_gpu=False):
    """
    Load YOLO model with GPU support
    """
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    if use_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, classes

def detect_objects(net, frame, classes, conf_threshold=0.5, nms_threshold=0.4):
    """
    Enhanced object detection with performance tracking
    """
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Time the forward pass
    start = time.time()
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)
    inference_time = time.time() - start

    boxes, confidences, class_ids = [], [], []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Prepare detection results with additional info
    detections = []
    if indices is not None:
        for i in indices.flatten():
            detections.append({
                'box': boxes[i],
                'confidence': confidences[i],
                'class': classes[class_ids[i]],
                'class_id': class_ids[i]
            })
    
    return detections, inference_time

def run_real_time_detection(config_path, weights_path, classes_path, 
                           video_source=0, use_gpu=False, show_fps=True):
    """
    Enhanced real-time detection with multiple improvements
    """
    # Download files if needed
    config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
    weights_url = "https://pjreddie.com/media/files/yolov3.weights"
    classes_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    
    download_yolo_files(config_url, weights_url, classes_url, 
                       config_path, weights_path, classes_path)
    
    net, classes = load_yolo_model(config_path, weights_path, classes_path, use_gpu)
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source {video_source}")
    
    # For FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Perform detection
        detections, inference_time = detect_objects(net, frame, classes)
        
        # Calculate FPS
        if frame_count % 10 == 0:
            fps = frame_count / (time.time() - start_time)
        
        # Draw detections with different colors for different classes
        for detection in detections:
            x, y, w, h = detection['box']
            conf = detection['confidence']
            cls = detection['class']
            
            # Generate color based on class
            color_hash = hash(cls) % 360
            color = tuple(int(c) for c in cv2.cvtColor(
                np.uint8([[[color_hash, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with background
            label = f"{cls}: {conf:.2f}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - label_size[1] - 5), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display FPS and inference time
        if show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Real-time Object Detection', frame)
        
        # Exit on 'q', save on 's'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"detection_{int(time.time())}.jpg", frame)
            print("Image saved!")
    
    cap.release()
    cv2.destroyAllWindows()
