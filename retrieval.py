"""
Image-Text Retrieval Module
Uses CLIP ViT to build a searchable image embedding index.
Embeddings are cached to avoid repeated computation.
"""

import os
import time
import logging
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CLIPRetrievalIndex:
    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self, device: Optional[str] = None, clip_model=None, clip_processor=None):
        if clip_model is not None:
            self.model     = clip_model
            self.processor = clip_processor
            self.device    = next(clip_model.parameters()).device
        else:
            from transformers import CLIPProcessor, CLIPModel
            self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
            self.model     = CLIPModel.from_pretrained(self.MODEL_NAME).to(self.device)
            self.model.eval()

        self._image_embeddings: Optional[np.ndarray] = None
        self._image_paths: List[str] = []
        self._image_meta: List[Dict] = []

    # ── Indexing ──────────────────────────────────────────────────────────────
    def _embed_images(self, images: List[Image.Image]) -> np.ndarray:
        embeddings = []
        for img in images:
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.get_image_features(**inputs)
                emb = out.pooler_output if hasattr(out, "pooler_output") else out
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.squeeze(0).cpu().numpy())
        return np.stack(embeddings, axis=0)

    def index(self, catalog: List[Dict]) -> None:
        """
        Build embedding index from a list of catalog entries.
        Each entry must have 'image_path'.
        """
        images = []
        for entry in catalog:
            try:
                img = Image.open(entry["image_path"]).convert("RGB")
            except Exception as e:
                logger.warning("Cannot open %s: %s", entry["image_path"], e)
                img = Image.new("RGB", (224, 224), (128, 128, 128))
            images.append(img)

        self._image_embeddings = self._embed_images(images)
        self._image_paths = [e["image_path"] for e in catalog]
        self._image_meta  = catalog
        logger.info("Indexed %d images for retrieval.", len(catalog))

    # ── Querying ──────────────────────────────────────────────────────────────
    def search(self, query: str, catalog: Optional[List[Dict]] = None, top_k: int = 5) -> List[Dict]:
        if self._image_embeddings is None:
            if catalog:
                self.index(catalog)
            else:
                logger.warning("No image index available for retrieval search.")
                return []
        t0 = time.time()
        inputs = self.processor(text=[query], return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_out = self.model.get_text_features(**inputs)
            text_emb = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_emb_np = text_emb.squeeze(0).cpu().numpy()

        similarities = self._image_embeddings @ text_emb_np
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            entry = self._image_meta[idx]
            results.append({
                "rank": rank + 1,
                "image_path": self._image_paths[idx],
                "title": entry.get("title", entry.get("image_caption", "Scene")),
                "similarity": round(float(similarities[idx]), 4),
            })
        logger.info("Retrieval query '%s' completed in %.3fs", query, time.time() - t0)
        return results

