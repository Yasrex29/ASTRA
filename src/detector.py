from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_people(frame):

    results = model(frame)

    centers = []
    boxes = []

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])

            if cls == 0:

                x1,y1,x2,y2 = map(int, box.xyxy[0])

                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                centers.append((cx,cy))
                boxes.append((x1,y1,x2,y2))

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    return boxes, centers
