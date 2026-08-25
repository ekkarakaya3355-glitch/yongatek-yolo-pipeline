import torch
from pathlib import Path
from ultralytics import YOLO

from app.config import Configs


def export_onnx(configs, fp16=False):

    cfg = configs["export"]
    device = cfg.get("device", 0)

    onnx_path = cfg["onnx_fp16_path"] if fp16 else cfg["onnx_path"]
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(cfg["weights"])
    model = yolo.model.fuse()
    model.eval()
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    dummy_input = torch.zeros(1, 3, cfg["imgsz"], cfg["imgsz"]).to(device)

    if fp16:
        model = model.half()
        dummy_input = dummy_input.half()

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=cfg["opset"],
            input_names=["images"],
            output_names=["output0"],
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )

    precision = "FP16" if fp16 else "FP32"
    size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"{precision} ONNX oluşturuldu: {onnx_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("-e", "--env", type=str, default="local")
    args = parser.parse_args()

    configs = Configs().load(config_name=args.env)
    export_onnx(configs, fp16=False)
    export_onnx(configs, fp16=True)