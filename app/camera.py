from dataclasses import dataclass, field

import cv2

@dataclass
class Camera:
    """
    Kamera kaynağını yöneten sınıf.

    Args
    source(int|str) : Kamera indeksi (0 = varsayılan kamera) veya video dosya yolu.
    width(int)      : İstenen kare genişliği.
    height(int)     : İstenen kare yüksekliği.
    """

    source: int = 0
    width: int = 640
    height: int = 480
    cap: cv2.VideoCapture = field(default=None, init=False)

    def open(self) -> None:
        """Kamerayı açar ve çözünürlüğü ayarlar."""
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            raise RuntimeError(f("Kamera açılamadı: {self.source}"))

    def read(self) -> None:
        """
        Kameradan bir kare okur.
        
        Returns
        (bool, ndarray) : Okuma başarılıysa True ve kare, değilse False ve None.
        """
        if self.cap is None:
            raise RuntimeError("Kamera açılmadan okuma yapılamaz. Önce open() çağırın.")

        ret, frame= self.cap.read()
        return ret, frame

    def close(self) -> None:
        """Kamerayı kapatır ve işlemi sonlandırır."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindow()