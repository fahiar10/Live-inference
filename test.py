from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor



if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    results = model.predict(source="0", show=True)
    print(results)