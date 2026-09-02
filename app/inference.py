from dataclasses import dataclass, field

from ultralytics import YOLO


@dataclass
class Inference:
    
    weights: str
    conf: float
    imgsz: int
    device: int
    model: YOLO = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.model = YOLO(self.weights)

    def infer(self, frame, draw=True):
        results = self.model(frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
                             verbose=False, rect=False)
        return results[0].plot() if draw else results[0]


if __name__ == "__main__":
    import cv2

    from app.camera import Camera

    camera_configs = {"source": 0, "width": 640, "height": 480, "delay": 1}
    inference_configs = {"weights": "yolo26n.pt", "conf": 0.25, "imgsz": 640, "device": 0}

    camera = Camera(**camera_configs)
    detector = Inference(**inference_configs)

    camera.open()
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        cv2.imshow("Detection", detector.infer(frame))
        if cv2.waitKey(camera.delay) & 0xFF == ord("q"):
            break
    camera.close()