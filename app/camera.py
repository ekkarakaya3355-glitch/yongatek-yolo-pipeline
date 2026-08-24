from dataclasses import dataclass, field

import cv2


@dataclass
class Camera:
    
    source: int
    width: int
    height: int
    delay: int
    cap: cv2.VideoCapture = field(default=None, init=False)

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera açılamadı: {self.source}")

    def read(self):
        if self.cap is None:
            raise RuntimeError("Kamera açılmadan okuma yapılamaz. Önce open() çağırın.")
        ret, frame = self.cap.read()
        return ret, frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()


if __name__ == "__main__":
    configs = {"source": 0, "width": 640, "height": 480, "delay": 1}
    camera = Camera(**configs)

    camera.open()
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        cv2.imshow("Camera", frame)
        if cv2.waitKey(camera.delay) & 0xFF == ord("q"):
            break
    camera.close()