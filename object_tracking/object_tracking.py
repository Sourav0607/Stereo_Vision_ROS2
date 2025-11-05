import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort


#CONFIG

CAM_INDEX = 2
CONF_THRESHOLD = 0.6
MODEL_PATH = 'yolov8n.pt'

# Initialize YOLO and Tracker

model = YOLO(MODEL_PATH)
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Open Camera

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()
print("Camera opened successfully. Object Tracking initialised.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Failed to read frame")
        break
    
    # YOLO DETECTION
    results = model(frame, conf=CONF_THRESHOLD,verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.cpu().numpy())
            cls = int(box.cls.cpu().numpy())
            detections.append([x1, y1, x2, y2, conf, cls])
            
    if len(detections)>0:
        dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections])
        tracked_objects = tracker.update(dets)
        
    else:
        tracked_objects = np.empty((0,5))
        
    #DRAW Detections and Tracked Objects
    
    # Draw detections (green boxes)
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = model.names[cls]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)    
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    # Draw tracked objects (blue boxes with IDs)
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 3)
        cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y2)+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        
    cv2.imshow("Object Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()