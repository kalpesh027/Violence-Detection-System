import cvzone
import numpy as np
from ultralytics import YOLO
import cv2
import math
from sort import *

#cap = cv2.VideoCapture(0)  # for Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("../videos/people.mp4")  # For Video

model = YOLO("../YOLO-Weight/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask-1.png")

limitUP =[103,161,296,161]  # line cross then detect as count
limitDown = [527, 489, 735, 489]
totalCountUp = []
totalCountDown = []


#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    success, frame = cap.read()
    if mask.shape[:2] != frame.shape[:2]:
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    else:
        mask_resized = mask
    countRegion = cv2.bitwise_and(frame, mask_resized)
    results = model(countRegion, stream = True)

    detections = np.empty((0, 5))

    for result in results:
        boxes = result.boxes
        for box in boxes:

            # Box around subject
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)

            # confidence
            conf = math.ceil((box.conf[0]*100))/100
            # class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == 'person' and conf > 0.3:
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    cv2.line(frame, (limitUP[0], limitUP[1]), (limitUP[2], limitUP[3]), (0, 0, 255), 2)
    cv2.line(frame, (limitDown[0], limitDown[1]), (limitDown[2], limitDown[3]), (0, 0, 255), 2)

    for result in resultTracker:
        x1,y1,x2,y2,Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1,
                           offset=5)
        cx, cy = x1 + w // 2, y1 + h // 2
        #cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if limitUP[0] < cx < limitUP [2] and limitUP[1] -15 < cy < limitUP[3] + 15:
            if totalCountUp.count(Id) == 0:
                totalCountUp.append(Id)
        cvzone.putTextRect(frame, f'Up-{len(totalCountUp)}', (1000, 620))

        if limitDown[0] < cx < limitDown [2] and limitDown[1] - 15 < cy < limitDown[3] + 15:
            if totalCountDown.count(Id) == 0:
                totalCountDown.append(Id)
        cvzone.putTextRect(frame, f'Down-{len(totalCountDown)}', (1000, 680))

    cv2.imshow('Video', frame)
    cv2.waitKey(1)