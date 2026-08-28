from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO

from app.config import Configs


@dataclass
class ONNXExporter:

    weights: str
    onnx_path: str
    onnx_fp16_path: str
    imgsz: int
    opset: int
    device: int

    def load_model(self, fp16: bool):
        yolo = YOLO(self.weights)
        model = yolo.model.fuse()
        model.eval()
        model = model.to(self.device)

        head = model.model[-1]
        head.export = True
        head.format = "onnx"

        for p in model.parameters():
            p.requires_grad = False

        if fp16:
            model = model.half()

        return model

    def export(self, fp16: bool = False) -> str:
        onnx_path = self.onnx_fp16_path if fp16 else self.onnx_path
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

        model = self.load_model(fp16)

        dummy_input = torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device)
        if fp16:
            dummy_input = dummy_input.half()  # Tracing

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                opset_version=self.opset,
                input_names=["images"],
                output_names=["output0"],
                do_constant_folding=True,
                export_params=True,
                dynamo=False,
            )

        precision = "FP16" if fp16 else "FP32"
        size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
        print(f"{precision} ONNX oluşturuldu: {onnx_path} ({size_mb:.1f} MB)")

        return onnx_path

    def export_all(self) -> None:
        self.export(fp16=False)
        self.export(fp16=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("-e", "--env", type=str, default="local")
    args = parser.parse_args()

    configs = Configs().load(config_name=args.env)
    ONNXExporter(**configs["export"]).export_all()
