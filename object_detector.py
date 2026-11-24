"""
Object Detection Module
Pure Vision Transformer (ViT) based detection: YOLOS-tiny
Outputs: bounding boxes, class labels, confidence scores.
"""

import time
import logging
import torch
from PIL import Image, ImageDraw
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class YOLOSDetector:
    """
    Pure-ViT object detector.  No CNN backbone.
    model: hustvl/yolos-tiny
    """
    MODEL_NAME = "hustvl/yolos-tiny"

    def __init__(self, device: Optional[str] = None, threshold: float = 0.25):
        from transformers import YolosImageProcessor, YolosForObjectDetection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        logger.info("Loading YOLOS detector on %s …", self.device)
        self.processor = YolosImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = YolosForObjectDetection.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def detect(self, image: Image.Image, threshold: Optional[float] = None) -> Dict:
        thr = threshold if threshold is not None else self.threshold
        t0 = time.time()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=thr, target_sizes=target_sizes
        )[0]

        detections: List[Dict] = []
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            box_coords = [round(float(c), 1) for c in box.tolist()]
            label_name = self.id2label.get(label_idx.item(), f"obj_{label_idx.item()}")
            detections.append({
                "label": label_name,
                "confidence": round(float(score), 3),
                "box": box_coords,   # [xmin, ymin, xmax, ymax]
            })
        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return {
            "detections": detections,
            "latency_s": round(time.time() - t0, 3),
            "model": "YOLOS-tiny (ViT)",
        }

    def annotate_image(self, image: Image.Image, detections: List[Dict]) -> Image.Image:
        """Draw bounding boxes and labels on a copy of the image."""
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        palette = [
            (56, 189, 248), (129, 140, 248), (16, 185, 129),
            (244, 63, 94), (245, 158, 11), (234, 179, 8),
        ]
        for i, det in enumerate(detections):
            color = palette[i % len(palette)]
            box = det["box"]
            label = det["label"].upper()
            conf = int(det["confidence"] * 100)
            draw.rectangle(box, outline=color, width=3)
            tag = f"{label} {conf}%"
            xmin, ymin = box[0], box[1]
            draw.rectangle(
                [xmin, max(0, ymin - 20), min(annotated.width, xmin + len(tag) * 7 + 8), ymin],
                fill=color,
            )
            draw.text((xmin + 4, max(0, ymin - 18)), tag, fill=(10, 10, 10))
        return annotated

