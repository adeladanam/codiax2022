import logging
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import sys
import torch
import cv2 as cv

from imagecontentextraction.modelling.yolov7.models.experimental import attempt_load
from imagecontentextraction.modelling.yolov7.utils.datasets import letterbox
from imagecontentextraction.modelling.yolov7.utils.general import non_max_suppression

FILE = Path(__file__).resolve()
ROOT = str(FILE.parents[1])  # YOLOv7 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

logger = logging.getLogger(__name__)


def load_yolov7_model(weights, processor_type):
    model = attempt_load(weights, map_location=processor_type)
    return model


def process_image_for_yolov7_from_cv_image(original_image, img_size, stride):
    # Padded resize
    processed_image = letterbox(original_image, img_size, stride=stride)[0]
    # Convert
    processed_image = processed_image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    processed_image = np.ascontiguousarray(processed_image)
    return processed_image


def detect_yolov7(model, processor_type, processed_image,
                  conf_thres=0.3,  # confidence threshold
                  iou_thres=0.45,  # NMS IOU threshold
                  classes=None,  # filter by class: --class 0, or --class 0 2 3
                  agnostic_nms=False  # class-agnostic NMS
                  ):
    detections = []
    img = torch.from_numpy(processed_image).to(processor_type)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            for *xyxy, conf, cls in reversed(det):
                p_x1, p_y1, p_x2, p_y2 = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                p1, p2 = (p_x1, p_y1), (p_x2, p_y2)
                confidence = float(conf)
                label_index = int(cls)
                label_name = model.names[label_index]
                if logger.isEnabledFor(logging.NOTSET):
                    to_draw_image = processed_image[0].copy()
                    cv.rectangle(to_draw_image, p1, p2, (0, 255, 0), 2)
                    cv.putText(to_draw_image, f"{label_name} with {confidence} confidence", (int(abs(p2[0] - p1[0]) / 2), int(abs(p2[1] - p1[1]) / 2)),
                                cv.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 255, 0), 2)
                    display_image(to_draw_image, convert_to_rgb=True)
                if p_x1 >= 0 and p_y1 >= 0 and p_x2 >= 0 and p_y2 >= 0:
                    detections.append(
                        {"coordinates": (p_x1, p_y1, p_x2, p_y2), "label_index": label_index, "label_name": label_name,
                         "confidence": confidence})
    return detections


def display_image(image):
    cv.imshow("Results", image)
    cv.waitKey(0)