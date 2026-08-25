from dataclasses import dataclass, field

import cv2
import numpy as np
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

    COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",]

    def __post_init__(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(self.weights, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_name = "images"
        self.output_name = "output0"

        in_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        out_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.output_shape = out_shape

        self.d_input = torch.empty(in_shape, dtype=torch.float16, device="cuda")
        self.d_output = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        self.buffers = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = torch.float16 if dtype == trt.DataType.HALF else torch.float32
            self.buffers[name] = torch.empty(shape, dtype=torch_dtype, device="cuda")

    def preprocess(self, frame):
        """Kareyi GPU'da modelin beklediği formata çevirir."""
        img = torch.from_numpy(frame).cuda()              # ham kare GPU'ya
        img = img.permute(2, 0, 1)                        # HWC → CHW
        img = img[[2, 1, 0]]                              # BGR → RGB
        img = img.unsqueeze(0).float()                    # batch ekle
        img = torch.nn.functional.interpolate(
            img, size=(self.imgsz, self.imgsz),
            mode="bilinear", align_corners=False
        )
        return (img / 255.0).half().squeeze(0)

    
    def infer(self, frame):
        h, w = frame.shape[:2]

        self.buffers[self.input_name].copy_(self.preprocess(frame).unsqueeze(0))

        for name, buf in self.buffers.items():
            self.context.set_tensor_address(name, buf.data_ptr())

        ok = self.context.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream
        )
        torch.cuda.synchronize()

        detections = self.buffers[self.output_name].cpu().numpy()[0]

        return self.draw(frame, detections, w / self.imgsz, h / self.imgsz)

    
    def draw(self, frame, detections, scale_x, scale_y):
        detections = detections[detections[:, 4] >= self.conf]
        for x1, y1, x2, y2, score, cls in detections:
            p1 = (int(x1 * scale_x), int(y1 * scale_y))
            p2 = (int(x2 * scale_x), int(y2 * scale_y))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(frame, f"{self.COCO_NAMES[int(cls)]} {score:.2f}",
                        (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame