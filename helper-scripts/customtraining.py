from ultralytics import YOLO

model = YOLO("../yolov8m.pt")  # Or yolov8s.pt / yolov8m.pt etc.
model.train(data="Vehicle_classification.v2-train_test.yolov8\\data.yaml", epochs=50, imgsz=640)