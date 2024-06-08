import cvzone
import numpy as np
from ultralytics import YOLO
import cv2
import math

# For video capture
#cap = cv2.VideoCapture(0)  # for Webcam
#cap.set(3, 1280)
#cap.set(4, 720)
cap = cv2.VideoCapture("../videos/demo1.gif")

# Load YOLO models
violenceDetect_model = YOLO("best.pt")
person_model = YOLO("yolov8n.pt")

classNames = ['NonViolence', 'Violence']

# Set confidence thresholds
PERSON_CONFIDENCE_THRESHOLD = 0.5
VIOLENCE_CONFIDENCE_THRESHOLD = 0.5

while True:
    success, frame = cap.read()
    if not success:
        break

    # Perform person detection
    person_results = person_model(frame)
    persons = []

    for result in person_results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0]
            if conf > PERSON_CONFIDENCE_THRESHOLD and int(box.cls[0]) == 0:  # Class '0' for 'person'
                x1, y1, x2, y2 = box.xyxy[0]
                persons.append((int(x1), int(y1), int(x2), int(y2)))

    # Perform violence detection if a person is detected
    if persons:
        violence_results = violenceDetect_model(frame)
        for result in violence_results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0]
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if conf > VIOLENCE_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw rectangles based on the detection class
                    if currentClass == 'Violence':
                        # Check if the violence box overlaps with any person box
                        for px1, py1, px2, py2 in persons:
                            if (x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cvzone.putTextRect(frame, f'Violence {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                                break

                    elif currentClass == 'NonViolence':
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(frame, f'NonViolence {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
