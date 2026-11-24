"""
Relationship Prediction Module
Uses CLIP ViT to semantically reason over (subject, predicate, object) triplets.
Works for multi-object scenes and handles single-object scenes gracefully.
"""

import time
import logging
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple

from configs.config import RELATION_VOCAB

logger = logging.getLogger(__name__)

ENV_CANDIDATES = ["grass", "lawn", "road", "floor", "table", "carpet", "water", "pavement", "sky", "shelf"]
POSTURE_FOR_ENTITY = {
    "dog": ["sitting on", "lying on", "standing on", "running on"],
    "cat": ["sitting on", "lying on", "standing on"],
    "person": ["sitting on", "standing on", "walking on", "lying on"],
    "default": ["on", "near", "next to", "in front of"],
}


class CLIPRelationshipPredictor:
    """
    Predicts relational triplets using CLIP-based semantic scoring.
    - Multi-object: pairwise CLIP reasoning across RELATION_VOCAB.
    - Single-object: grounds entity against environmental context.
    """
    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self, device: Optional[str] = None, clip_model=None, clip_processor=None):
        """
        Accepts pre-loaded CLIP model/processor to avoid duplicate downloads.
        """
        if clip_model is not None:
            self.model = clip_model
            self.processor = clip_processor
            self.device = next(clip_model.parameters()).device
        else:
            from transformers import CLIPProcessor, CLIPModel
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
            self.model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self.device)
            self.model.eval()

    def _score_prompts(self, image: Image.Image, prompts: List[str]) -> np.ndarray:
        inputs = self.processor(
            text=prompts, images=image,
            return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)
        with torch.no_grad():
            return self.model(**inputs).logits_per_image.softmax(dim=1).squeeze(0).cpu().numpy()

    def predict_single_entity_context(
        self, image: Image.Image, entity: Dict
    ) -> Tuple[str, str, float]:
        """
        When only one object is detected, ground it to its environment.
        Returns (predicate, env_object, confidence).
        """
        # 1. Identify environment
        env_probs = self._score_prompts(image, [f"a photo of {e}" for e in ENV_CANDIDATES])
        best_env = ENV_CANDIDATES[int(np.argmax(env_probs))]
        env_conf = float(np.max(env_probs))

        # 2. Identify entity posture on that environment
        label = entity.get("label", "object").lower()
        postures = POSTURE_FOR_ENTITY.get(label, POSTURE_FOR_ENTITY["default"])
        posture_probs = self._score_prompts(
            image, [f"a {label} {p} {best_env}" for p in postures]
        )
        best_posture = postures[int(np.argmax(posture_probs))]
        posture_conf = float(np.max(posture_probs))

        return best_posture, best_env, round((env_conf + posture_conf) / 2, 3)

    def predict_pairwise(
        self, image: Image.Image, enhanced_detections: List[Dict], max_edges: int = 8
    ) -> Tuple[List[Dict], float]:
        """
        Multi-object pairwise relationship prediction.
        Returns sorted list of triplets with confidence scores.
        """
        t0 = time.time()
        candidates = []
        n = len(enhanced_detections)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s = enhanced_detections[i]
                o = enhanced_detections[j]
                prompts = [f"a {s['label']} {r} a {o['label']}" for r in RELATION_VOCAB]
                probs = self._score_prompts(image, prompts)
                best_idx = int(np.argmax(probs))
                candidates.append({
                    "subject": s["label"],
                    "subject_id": s.get("id", i + 1),
                    "predicate": RELATION_VOCAB[best_idx],
                    "object": o["label"],
                    "object_id": o.get("id", j + 1),
                    "confidence": round(float(probs[best_idx]), 3),
                })

        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        # Deduplicate: keep only strongest edge per pair (undirected)
        seen_pairs = set()
        triplets = []
        for c in candidates[:max_edges * 2]:
            pair = frozenset({(c["subject"], c["subject_id"]), (c["object"], c["object_id"])})
            if pair not in seen_pairs:
                triplets.append(c)
                seen_pairs.add(pair)
            if len(triplets) >= max_edges:
                break

        return triplets, round(time.time() - t0, 3)

    def predict(
        self, image: Image.Image, enhanced_detections: List[Dict], max_edges: int = 8
    ) -> Tuple[List[Dict], float]:
        """
        Unified entry point. Dispatches to single- or multi-object logic.
        """
        if len(enhanced_detections) == 0:
            return [], 0.0
        if len(enhanced_detections) == 1:
            det = enhanced_detections[0]
            pred, env, conf = self.predict_single_entity_context(image, det)
            triplet = {
                "subject": det["label"],
                "subject_id": det.get("id", 1),
                "predicate": pred,
                "object": env,
                "object_id": -1,
                "confidence": conf,
            }
            return [triplet], 0.0
        return self.predict_pairwise(image, enhanced_detections, max_edges)

