from dataclasses import dataclass, field

import cv2

@dataclass
class Camera:
    source: int = 0
    width: int = 640
    height: int = 480
    delay: int = 1
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

    def stream(self):
        self.open()
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame
        self.close()

    def show(self, frame) -> bool:
        cv2.imshow("Detection", frame)
        return cv2.waitKey(self.delay) & 0xFF != ord("q")


if __name__ == "__main__":
    camera = Camera()
    for frame in camera.stream():
        if not camera.show(frame):
            break