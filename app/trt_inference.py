from dataclasses import dataclass, field
from pathlib import Path
import json

import cv2
import torch
import tensorrt as trt


COCO_PATH = Path(__file__).resolve().parent.parent / "coco.json"

with open(COCO_PATH, encoding="utf-8") as f:
    _coco = json.load(f)

COCO_NAMES = tuple(_coco["names"])
COCO_COLORS = tuple(tuple(color) for color in _coco["colors"])


@dataclass
class TRTInference:

    weights: str
    conf: float
    imgsz: int
    device: int
    engine: object = field(default=None, init=False)
    context: object = field(default=None, init=False)

    def __post_init__(self) -> None:
        torch.cuda.set_device(self.device)
        self.torch_device = torch.device("cuda", self.device)

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(self.weights, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_name = "images"
        self.output_name = "output0"

        self.stream = torch.cuda.Stream(device=self.torch_device)

        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = torch.float16 if dtype == trt.DataType.HALF else torch.float32
            self.buffers[name] = torch.empty(shape, dtype=torch_dtype, device=self.torch_device)
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())

        self.input_dtype = self.buffers[self.input_name].dtype

        engine_h, engine_w = self.buffers[self.input_name].shape[-2:]
        if (engine_h, engine_w) != (self.imgsz, self.imgsz):
            raise ValueError(
                f"Engine girdi boyutu {engine_h}x{engine_w}, config imgsz={self.imgsz}. "
                f"Engine'i yeniden derleyin veya config'i düzeltin."
            )

    def preprocess(self, frame):
        
        h, w = frame.shape[:2]
        ratio = min(self.imgsz / h, self.imgsz / w)
        new_h, new_w = round(h * ratio), round(w * ratio)
        pad_x, pad_y = (self.imgsz - new_w) // 2, (self.imgsz - new_h) // 2

        img = torch.from_numpy(frame).to(self.torch_device)
        img = img.permute(2, 0, 1)                           
        img = img[[2, 1, 0]]                                 
        img = img.unsqueeze(0).float()
        img = torch.nn.functional.interpolate(
            img, size=(new_h, new_w),
            mode="bilinear", align_corners=False
        )

        canvas = torch.full((1, 3, self.imgsz, self.imgsz), 114 / 255,
                            dtype=self.input_dtype, device=self.torch_device)
        canvas[:, :, pad_y:pad_y + new_h, pad_x:pad_x + new_w] = (img / 255.0).to(self.input_dtype)

        return canvas, ratio, pad_x, pad_y

    def infer(self, frame, draw=True):
      
        with torch.cuda.stream(self.stream):
            canvas, ratio, pad_x, pad_y = self.preprocess(frame)
            self.buffers[self.input_name].copy_(canvas)
            ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

        if not ok:
            raise RuntimeError(f"TensorRT çıkarımı başarısız: {self.weights}")

        detections = self.buffers[self.output_name].cpu().numpy()[0]

        if not draw:
            return detections, ratio, pad_x, pad_y

        return self.draw(frame.copy(), detections, ratio, pad_x, pad_y)

    def draw(self, frame, detections, ratio, pad_x, pad_y):
        h, w = frame.shape[:2]
        detections = detections[detections[:, 4] >= self.conf]

        for x1, y1, x2, y2, score, cls in detections:

            p1 = (min(max(int((x1 - pad_x) / ratio), 0), w - 1), min(max(int((y1 - pad_y) / ratio), 0), h - 1))
            p2 = (min(max(int((x2 - pad_x) / ratio), 0), w - 1), min(max(int((y2 - pad_y) / ratio), 0), h - 1))

            index = int(cls)
            name = COCO_NAMES[index] if index < len(COCO_NAMES) else str(index)
            color = COCO_COLORS[index % len(COCO_COLORS)]

            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(frame, f"{name} %{score * 100:.0f}",
                        (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 1)
        return frame


if __name__ == "__main__":
    from app.camera import Camera

    camera_configs = {"source": 0, "width": 640, "height": 480, "delay": 1}
    inference_configs = {
        "weights": "models/engine/yolo26x_fp16.engine",
        "conf": 0.30,
        "imgsz": 640,
        "device": 0,
    }

    camera = Camera(**camera_configs)
    detector = TRTInference(**inference_configs)

    camera.open()
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        cv2.imshow("Detection", detector.infer(frame))
        if cv2.waitKey(camera.delay) & 0xFF == ord("q"):
            break
    camera.close()
