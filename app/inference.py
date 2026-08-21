from dataclasses import dataclass, field
from ultralytics import YOLO

@dataclass
class Inference:
    weights: str = "yolo26n.pt"
    conf: float = 0.25
    imgsz: int = 640
    device: int = 0
    model: YOLO = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.model = YOLO(self.weights)

    def infer(self, frame):
        results = self.model(frame, conf=self.conf, imgsz=self.imgsz,device=self.device, verbose=False)
        return results[0].plot()


if __name__ == "__main__":
    from app.camera import Camera

    camera = Camera()
    detector = Inference()

    for frame in camera.stream():
        if not camera.show(detector.infer(frame)):
            break