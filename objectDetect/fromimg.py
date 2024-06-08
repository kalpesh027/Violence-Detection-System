from ultralytics import YOLO
import cv2

model = YOLO('../YOLO-Weight/yolov8n.pt')
results = model("../images/public.png", show=True)

cv2.waitKey(0)
