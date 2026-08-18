import cv2
from app.camera import Camera


def run(configs, logger):
    logger.info("Çıkarım modu başlatıldı.")

    camera = Camera(**configs["camera"])
    camera.open()
    logger.info(f"Kamera Açıldı: {configs['camera']['source']}")

    try:
        while True:
            ret, frame =camera.read()
            if not ret:
                logger.warning("Kare okunamadı, döngü sonlandırılıyor")
                break
            
            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0XFF == ord("q"):
                logger.info("Kullanıcı çıkış yaptı")
                break
    finally:
        camera.close()
        logger.info("Kamera Kapatıldı")
    