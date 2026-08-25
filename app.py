from pathlib import Path

import cv2

from app.config import Configs
from app.logger import Logger
from app.camera import Camera
from app.inference import Inference
from app.trt_inference import TRTInference


def main(args, configs):
    logger = Logger(**configs["logger"])
    logger.debug("############ CONFIGURATIONS ############")
    logger.debug(configs)

    camera = Camera(**configs["camera"])

    cfg = configs["inference"]
    if Path(cfg["weights"]).suffix == ".engine":
        detector = TRTInference(**cfg)
        logger.info(f"TensorRT engine yüklendi: {cfg['weights']}")
    else:
        detector = Inference(**cfg)
        logger.info(f"PyTorch modeli yüklendi: {cfg['weights']}")

    camera.open()
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    logger.info("Çıkarım başlatıldı")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                logger.warning("Kare okunamadı")
                break

            cv2.imshow("Detection", detector.infer(frame))

            if cv2.waitKey(camera.delay) & 0xFF == ord("q"):
                logger.info("Kullanıcı çıkış yaptı")
                break
    finally:
        camera.close()
        logger.info("Kamera kapatıldı")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Computer Vision Pipeline")
    parser.add_argument("-e", "--env", type=str, default="local")
    args = parser.parse_args()

    configs = Configs().load(config_name=args.env)
    main(args, configs)