from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # or yolov8s.yaml for more capacity
model.train(data="patos_dataset_improved_yolo_format/data.yaml", epochs=50, imgsz=640)
