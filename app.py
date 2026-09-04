import os
import time
from pathlib import Path

os.environ.setdefault("QT_LOGGING_RULES", "*=false")

import cv2

from app.config import Configs
from app.logger import Logger
from app.camera import Camera
from app.inference import Inference
from app.trt_inference import TRTInference


def smooth(current, period):
    fps = 1.0 / period
    return fps if current == 0 else current * 0.9 + fps * 0.1


def overlay(frame, text):
    """FPS yazisini kare boyutuna oranlar; 1080p'de eski gorunumu verir."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = max(6, round(h * 0.02))
    scale = max(0.35, h / 900.0)
    thickness = max(1, round(scale * 1.7))

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw > w - 2 * margin:                    # dar karede genisligi de gozet
        scale *= (w - 2 * margin) / tw
        thickness = max(1, round(scale * 1.7))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    cv2.putText(frame, text, (margin, margin + th), font, scale, (0, 255, 0), thickness)


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

    # gosterim kapaliyken app.py yapisal olarak benchmark.py ile ayni yolu kosar
    display = not args.no_display

    camera.open()
    if display:
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    else:
        logger.info("Gösterim kapalı, çıkış için Ctrl+C")

    logger.info(f"Kaynak FPS: {camera.cap.get(cv2.CAP_PROP_FPS):.2f}")
    logger.info("Çıkarım başlatıldı")

    # benchmark.py ile ayni sayiyi atsin diye isinma turu oradan okunuyor
    warmup = configs.get("benchmark", {}).get("warmup", 20)

    # finally icinde okundugu icin try'dan once tanimli olmali (ornegin Ctrl+C)
    infer_times = []

    try:
        infer_fps = 0.0
        loop_fps = 0.0
        last = time.perf_counter()
        while True:
            ret, frame = camera.read()
            if not ret:
                logger.warning("Kare okunamadı")
                break

            start = time.perf_counter()
            output = detector.infer(frame)
            period = time.perf_counter() - start

            infer_times.append(period)
            infer_fps = smooth(infer_fps, period)

            if display:
                overlay(output, f"Inference {infer_fps:.0f} | "
                                f"Pipeline (decode+inference+display) {loop_fps:.0f} FPS")
                cv2.imshow("Detection", output)

            now = time.perf_counter()
            loop_fps = smooth(loop_fps, now - last)
            last = now

            if display and cv2.waitKey(camera.delay) & 0xFF == ord("q"):
                logger.info("Kullanıcı çıkış yaptı")
                break
    finally:
        camera.close()
        logger.info("Kamera kapatıldı")

        # soguk baslangic ortalamayi bozuyor, benchmark.py de ilk kareleri atiyor
        atilan = warmup if len(infer_times) > warmup else 0
        olculen = infer_times[atilan:]

        if olculen:
            # benchmark.py ile ayni hesap: once ms ortalamasi, FPS ondan tureniyor
            mean_ms = sum(olculen) / len(olculen) * 1000
            ozet = (f"Çıkarım ortalaması: {mean_ms:.2f} ms  "
                    f"{1000 / mean_ms:.1f} FPS  "
                    f"({len(olculen)} kare, ilk {atilan} ısınma atıldı)")
            logger.info(ozet)
            print(f"\n{ozet}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Computer Vision Pipeline")
    parser.add_argument("-e", "--env", type=str, default="local")
    parser.add_argument("--no-display", action="store_true",
                        help="pencereyi hiç açma; benchmark.py ile aynı yolu ölçmek için")
    args = parser.parse_args()

    configs = Configs().load(config_name=args.env)
    main(args, configs)