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


def benchmark(detector, frame, warmup, runs, mem_before):
    for _ in range(warmup):
        detector.infer(frame.copy())
    torch.cuda.synchronize()

    memory_mb = gpu_memory_mb() - mem_before

    timings = []
    for _ in range(runs):
        img = frame.copy()  
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        detector.infer(img)

        torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)

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

    cap = cv2.VideoCapture(configs["camera"]["source"])
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Test karesi okunamadı")

    print(f"Test karesi : {frame.shape}")
    print(f"imgsz={infer_cfg['imgsz']}  warmup={bench['warmup']}  runs={bench['runs']}")


    models = {
        "PyTorch": (bench["pytorch_weights"], True),
        "TensorRT FP32": (bench["fp32_weights"], False),
        "TensorRT FP16": (bench["fp16_weights"], True),
    }

    results = {}
    for name, (weights, tf32) in models.items():
        print(f"Ölçülüyor: {name}")
        torch.backends.cudnn.allow_tf32 = tf32
        torch.cuda.empty_cache()
        mem_before = gpu_memory_mb()
        detector = load_detector(weights, infer_cfg)
        results[name] = benchmark(detector, frame, bench["warmup"], bench["runs"], mem_before)
        del detector
        torch.cuda.empty_cache()

    print_table(results)