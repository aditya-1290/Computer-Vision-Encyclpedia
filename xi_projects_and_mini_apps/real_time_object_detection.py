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

def load_yolo_model(config_path, weights_path, classes_path):
    """
    Load YOLO model.
    """
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

def detect_objects(net, frame, classes, conf_threshold=0.5, nms_threshold=0.4):
    """
    Detect objects in frame.
    """
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return [(boxes[i], confidences[i], classes[class_ids[i]]) for i in indices.flatten()]

def run_real_time_detection(config_path, weights_path, classes_path):
    """
    Run real-time detection on webcam.
    """
    net, classes = load_yolo_model(config_path, weights_path, classes_path)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect_objects(net, frame, classes)
        for (box, conf, cls) in detections:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls}: {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Real-time Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Assume YOLO files are available
    # run_real_time_detection('yolov3.cfg', 'yolov3.weights', 'coco.names')
    print("Real-time object detection function defined.")
