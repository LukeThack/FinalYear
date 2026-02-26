
from ultralytics import YOLO
model = YOLO("runs/obb/train/weights/best.pt")


def test_YOLO_model_accuracy_DSSDD_and_SSDD():
    metrics = model.val(data="ssdd_offshore.yaml")
    map_50_95 = metrics.box.map
    assert(map_50_95>0.8)


