import time
from pathlib import Path

import cv2
import numpy as np
import torch

from app.config import Configs
from app.inference import Inference
from app.trt_inference import TRTInference


def load_detector(weights, cfg):
    params = {
        "weights": weights,
        "conf": cfg["conf"],
        "imgsz": cfg["imgsz"],
        "device": cfg["device"],
    }
    if Path(weights).suffix == ".engine":
        return TRTInference(**params)
    return Inference(**params)


def gpu_memory_mb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024 ** 2)


def benchmark(detector, source, warmup, runs, mem_before):
    # her model videoyu bastan okur, yani ucu de ayni kareleri gorur
    cap = cv2.VideoCapture(source)

    for _ in range(warmup):
        ok, frame = cap.read()
        if not ok:
            break
        detector.infer(frame)
    torch.cuda.synchronize()

    memory_mb = gpu_memory_mb() - mem_before

    timings = []
    for _ in range(runs):
        # decode olcumun disinda; app.py'de de cikarim suresine dahil degil
        ok, frame = cap.read()
        if not ok:
            break

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        detector.infer(frame)

        torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)

    cap.release()

    if not timings:
        raise RuntimeError(f"Olculecek kare kalmadi: {source}")

    timings = np.array(timings)

    return {
        "mean_ms": timings.mean(),
        "median_ms": np.median(timings),
        "p95_ms": np.percentile(timings, 95),
        "fps": 1000 / timings.mean(),
        "memory_mb": memory_mb,
    }


def print_table(results):
    header = f"{'Model':<28}{'Mean(ms)':>10}{'Median':>10}{'P95':>10}{'FPS':>10}{'GPU(MB)':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for name, r in results.items():
        print(f"{name:<28}{r['mean_ms']:>10.2f}{r['median_ms']:>10.2f}"
              f"{r['p95_ms']:>10.2f}{r['fps']:>10.1f}{r['memory_mb']:>10.0f}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT")
    parser.add_argument("-e", "--env", type=str, default="local")
    args = parser.parse_args()

    configs = Configs().load(config_name=args.env)
    bench = configs["benchmark"]
    infer_cfg = configs["inference"]

    source = configs["camera"]["source"]

    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not ret:
        raise RuntimeError(f"Video okunamadı: {source}")

    needed = bench["warmup"] + bench["runs"]
    if 0 < total_frames < needed:
        print(f"Uyarı: video {total_frames} kare, {needed} gerekiyor; ölçüm kısa kesilecek\n")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"Video       : {source}  {frame.shape[1]}x{frame.shape[0]}  {total_frames} kare")
    print(f"imgsz={infer_cfg['imgsz']}  warmup={bench['warmup']}  runs={bench['runs']}")
    print(f"Ölçülen: app.py ile aynı yol (çizim dahil), kareler videodan sırayla okunuyor, "
          f"her backend letterbox ile {infer_cfg['imgsz']}x{infer_cfg['imgsz']} girdi görüyor")
    print("FP32 satırları saf FP32'dir, TF32 hem TensorRT hem PyTorch tarafında kapalı\n")

    models = {
        "PyTorch FP32": bench["pytorch_weights"],
        "TensorRT FP32": bench["fp32_weights"],
        "TensorRT FP16": bench["fp16_weights"],
    }

    results = {}
    for name, weights in models.items():
        print(f"Ölçülüyor: {name}")
        torch.cuda.empty_cache()
        mem_before = gpu_memory_mb()
        detector = load_detector(weights, infer_cfg)
        results[name] = benchmark(detector, source, bench["warmup"], bench["runs"], mem_before)
        del detector
        torch.cuda.empty_cache()

    print_table(results)