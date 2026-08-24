import cv2

from app.config import Configs
from app.logger import Logger
from app.camera import Camera
from app.inference import Inference


def main(args, configs):
    logger = Logger(**configs["logger"])
    logger.debug("############ CONFIGURATIONS ############")
    logger.debug(configs)

    mode = configs.get("mode")

    if mode == "train":
        pass

    elif mode == "predict":
        camera = Camera(**configs["camera"])
        detector = Inference(**configs["inference"])

        camera.open()
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        logger.info("Çıkarım başlatıldı")

        while True:
            ret, frame = camera.read()
            if not ret:
                logger.warning("Kare okunamadı")
                break
            cv2.imshow("Detection", detector.infer(frame))
            if cv2.waitKey(camera.delay) & 0xFF == ord("q"):
                logger.info("Kullanıcı çıkış yaptı")
                break
        camera.close()
        logger.info("Kamera kapatıldı")

    else:
        logger.error(f"Geçersiz mod: '{mode}'")
        raise ValueError(f"Geçersiz mod: '{mode}'. Beklenen: 'train' veya 'predict'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Computer Vision Pipeline")
    parser.add_argument("-e", "--env", type=str, default="local")
    args = parser.parse_args()

    configs = Configs().load(config_name=args.env)
    main(args, configs)