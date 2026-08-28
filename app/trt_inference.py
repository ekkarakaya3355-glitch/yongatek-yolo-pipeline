from dataclasses import dataclass, field

import cv2
import torch
import tensorrt as trt


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

        engine_imgsz = self.buffers[self.input_name].shape[-1]
        if engine_imgsz != self.imgsz:
            raise ValueError(
                f"Engine girdi boyutu {engine_imgsz}, config imgsz={self.imgsz}. "
                f"Engine'i yeniden derleyin veya config'i düzeltin."
            )

    def preprocess(self, frame):
        img = torch.from_numpy(frame).to(self.torch_device)  
        img = img.permute(2, 0, 1)                           
        img = img[[2, 1, 0]]                                
        img = img.unsqueeze(0).float()                       
        img = torch.nn.functional.interpolate(
            img, size=(self.imgsz, self.imgsz),
            mode="bilinear", align_corners=False
        )
        return (img / 255.0).to(self.input_dtype)

    def infer(self, frame):
        h, w = frame.shape[:2]

        with torch.cuda.stream(self.stream):
            self.buffers[self.input_name].copy_(self.preprocess(frame))
            ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

        if not ok:
            raise RuntimeError(f"TensorRT çıkarımı başarısız: {self.weights}")

        detections = self.buffers[self.output_name].cpu().numpy()[0]

        return self.draw(frame, detections, w / self.imgsz, h / self.imgsz)

    def draw(self, frame, detections, scale_x, scale_y):
        detections = detections[detections[:, 4] >= self.conf]
        for x1, y1, x2, y2, score, cls in detections:
            p1 = (int(x1 * scale_x), int(y1 * scale_y))
            p2 = (int(x2 * scale_x), int(y2 * scale_y))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(frame, f"{int(cls)} %{score * 100:.0f}",
                        (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 1)
        return frame


if __name__ == "__main__":
    from app.camera import Camera

    camera_configs = {"source": 0, "width": 640, "height": 480, "delay": 1}
    inference_configs = {
        "weights": "models/engine/yolo26s_fp16.engine",
        "conf": 0.30,
        "imgsz": 960,
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
