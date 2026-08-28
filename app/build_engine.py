from dataclasses import dataclass
from pathlib import Path

import tensorrt as trt

from app.config import Configs


@dataclass
class EngineBuilder:

    onnx_path: str
    onnx_fp16_path: str
    fp32_path: str
    fp16_path: str
    workspace: int

    def build(self, onnx_path: str, engine_path: str) -> str:
        engine_path = Path(engine_path)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            ok = parser.parse(f.read())

        if not ok:
            print(f"Parser hata sayısı: {parser.num_errors}")
            for i in range(parser.num_errors):
                print(f"  [{i}] {parser.get_error(i)}")
            raise RuntimeError(f"ONNX parse edilemedi: {onnx_path}")

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace * (1 << 30))

        print(f"Engine derleniyor: {engine_path}")

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Engine derlenemedi")

        with open(engine_path, "wb") as f:
            f.write(serialized)

        size_mb = engine_path.stat().st_size / (1024 * 1024)
        print(f"Engine oluşturuldu: {engine_path} ({size_mb:.1f} MB)")

        return str(engine_path)

    def build_all(self) -> None:
        self.build(self.onnx_path, self.fp32_path)
        self.build(self.onnx_fp16_path, self.fp16_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model")
    parser.add_argument("-e", "--env", type=str, default="local")
    args = parser.parse_args()

    configs = Configs().load(config_name=args.env)
    EngineBuilder(**configs["engine"]).build_all()
