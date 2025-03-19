# Object Detection menggunakan YOLOv5
# package
# pip install torch torchvision
# pip install opencv-python
# pip install numpy
# Untuk YOLOv5: pip install -U ultralytics

import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time

def detect_objects(image_path, conf_threshold=0.25):
    """
    Melakukan deteksi objek pada gambar menggunakan YOLOv5
    
    Args:
        image_path (str): Path ke file gambar
        conf_threshold (float): Threshold keyakinan untuk deteksi (0-1)
        
    Returns:
        image_with_boxes (numpy.ndarray): Gambar dengan bounding box
        results (list): Hasil deteksi
    """
    # Load model YOLOv5
    model = YOLO('yolov5s.pt')  # Model YOLOv5 small
    
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Tidak dapat membaca gambar: {image_path}")
    
    # Lakukan deteksi
    results = model(image, conf=conf_threshold)
    
    # Tampilkan hasil
    annotated_img = results[0].plot()
    
    # Ambil informasi deteksi (class, confidence, bounding box)
    detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Koordinat x1, y1, x2, y2
            conf = float(box.conf)  # Confidence
            cls = int(box.cls)  # Class ID
            class_name = model.names[cls]  # Class name
            
            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
    
    return annotated_img, detections

def detect_from_webcam():
    """
    Menjalankan deteksi objek pada webcam secara real-time
    """
    # Load model YOLOv5
    model = YOLO('yolov5s.pt')
    
    # Akses webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Deteksi objek pada frame
        start_time = time.time()
        results = model(frame)
        end_time = time.time()
        
        # Hitung FPS
        fps = 1 / (end_time - start_time)
        
        # Gambar hasil deteksi
        annotated_frame = results[0].plot()
        
        # Tampilkan FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Tampilkan frame
        cv2.imshow("YOLOv5 Detection", annotated_frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Contoh penggunaan untuk deteksi pada gambar
if __name__ == "__main__":
    
    # 1. Deteksi pada gambar
    # image_path = "lokasi/image.jpg"  # path gambar
    # annotated_img, detections = detect_objects(image_path, conf_threshold=0.25)
    
    # # Tampilkan gambar dengan bounding box
    # cv2.imshow("YOLOv5 Detection", annotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # # Cetak hasil deteksi
    # for det in detections:
    #     print(f"Class: {det['class']}, Confidence: {det['confidence']:.2f}, BBox: {det['bbox']}")
    
    # 2. Deteksi pada webcam (real-time)
    detect_from_webcam()