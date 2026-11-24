"""
Attribute Classifier — CLIP ViT Zero-Shot Semantic Projection
Predicts visual attributes per detected object ROI without random noise.
"""

import time
import logging
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Optional

from configs.config import ATTRIBUTE_VOCAB

logger = logging.getLogger(__name__)


class CLIPAttributeClassifier:
    """
    Uses CLIP ViT text-image matching to predict multi-label attributes
    for each detected bounding box.
    No random weights. No noise. All predictions are semantic.
    """
    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self, device: Optional[str] = None):
        from transformers import CLIPProcessor, CLIPModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CLIP attribute classifier on %s …", self.device)
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()

    def _crop_roi(self, image: Image.Image, box: List[float]) -> Image.Image:
        xmin, ymin, xmax, ymax = [max(0, int(b)) for b in box]
        w, h = image.size
        xmax = min(w, max(xmin + 10, xmax))
        ymax = min(h, max(ymin + 10, ymax))
        crop = image.crop((xmin, ymin, xmax, ymax))
        return crop if crop.size[0] >= 10 and crop.size[1] >= 10 else image

    def classify(
        self,
        image: Image.Image,
        box: List[float],
        entity_label: str,
        top_k: int = 3,
    ) -> List[Dict]:
        crop = self._crop_roi(image, box)
        prompts = [f"a photo of a {attr} {entity_label}" for attr in ATTRIBUTE_VOCAB]
        inputs = self.processor(
            text=prompts, images=crop,
            return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)
        with torch.no_grad():
            probs = self.model(**inputs).logits_per_image.softmax(dim=1).squeeze(0).cpu().numpy()
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [
            {"attribute": ATTRIBUTE_VOCAB[i], "confidence": round(float(probs[i]), 3)}
            for i in top_indices
        ]

    def classify_all(
        self,
        image: Image.Image,
        detections: List[Dict],
        top_k: int = 3,
    ) -> List[Dict]:
        """Augment a list of detections with predicted attributes."""
        t0 = time.time()
        enhanced = []
        for idx, det in enumerate(detections):
            attrs = self.classify(image, det["box"], det["label"], top_k=top_k)
            enhanced.append({**det, "id": idx + 1, "attributes": attrs})
        return enhanced, round(time.time() - t0, 3)

