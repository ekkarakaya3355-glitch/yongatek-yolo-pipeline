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
        logger.info("Çıkarım başlatıldı")

        for frame in camera.stream():
            if not camera.show(detector.infer(frame)):
                logger.info("Kullanıcı çıkış yaptı")
                break

        logger.info("Çıkarım sonlandı")

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