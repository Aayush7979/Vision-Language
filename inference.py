"""
Unified Inference Engine
Single entry point that orchestrates all pipeline components.
Loads models once; exposes a process(image, question) interface.
"""

import os
import sys
import time
import logging
from PIL import Image
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# Ensure src/ is on path when imported as a module
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_HERE, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class MultimodalInferenceEngine:
    """
    Lazy-loads every model component and exposes a single
    `process(image, question)` call that returns a fully populated results dict.
    """

    def __init__(self, device: Optional[str] = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Inference Engine on %s", self.device)

        # Deferred imports so only used models are loaded
        self._blip      = None
        self._baseline  = None
        self._detector  = None
        self._attr      = None
        self._rel       = None
        self._vqa       = None
        self._retrieval = None
        self._clip_model     = None
        self._clip_processor = None

    # ── Lazy loaders ──────────────────────────────────────────────────────────
    def _load_clip(self):
        if self._clip_model is None:
            from transformers import CLIPProcessor, CLIPModel
            logger.info("Loading CLIP …")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self._clip_model.eval()

    def _load_blip(self):
        if self._blip is None:
            from src.caption_model import BLIPCaptioner
            self._blip = BLIPCaptioner(device=self.device)

    def _load_baseline(self):
        if self._baseline is None:
            from src.caption_model import ViTGPT2Captioner
            self._baseline = ViTGPT2Captioner(device=self.device)

    def _load_detector(self):
        if self._detector is None:
            from src.object_detector import YOLOSDetector
            self._detector = YOLOSDetector(device=self.device)

    def _load_attr(self):
        self._load_clip()
        if self._attr is None:
            from src.attribute_classifier import CLIPAttributeClassifier
            self._attr = CLIPAttributeClassifier.__new__(CLIPAttributeClassifier)
            self._attr.model     = self._clip_model
            self._attr.processor = self._clip_processor
            self._attr.device    = self.device

    def _load_rel(self):
        self._load_clip()
        if self._rel is None:
            from src.relationship_model import CLIPRelationshipPredictor
            self._rel = CLIPRelationshipPredictor(
                clip_model=self._clip_model,
                clip_processor=self._clip_processor,
            )

    def _load_vqa(self):
        if self._vqa is None:
            from src.vqa import BLIPVQAModel
            self._vqa = BLIPVQAModel(device=self.device)

    def _load_retrieval(self, catalog: Optional[List[Dict]] = None):
        self._load_clip()
        if self._retrieval is None:
            from src.retrieval import CLIPRetrievalIndex
            self._retrieval = CLIPRetrievalIndex(
                clip_model=self._clip_model,
                clip_processor=self._clip_processor,
            )
        if catalog and (self._retrieval._image_embeddings is None or len(self._retrieval._image_meta) != len(catalog)):
            self._retrieval.index(catalog)

    # ── Main inference pipeline ───────────────────────────────────────────────
    def process(
        self,
        image: Image.Image,
        question: Optional[str] = None,
        retrieval_catalog: Optional[List[Dict]] = None,
        det_threshold: float = 0.25,
        caption_beams: int = 4,
    ) -> Dict:
        results = {"timestamp": time.time()}
        t_total = time.time()

        # 1. BLIP Captioning
        self._load_blip()
        blip_out = self._blip.caption(image, num_beams=caption_beams, max_length=48)
        results["blip_caption"] = blip_out

        # 2. ViT-GPT2 Baseline Caption (comparative)
        self._load_baseline()
        gpt2_out = self._baseline.caption(image, num_beams=1, max_length=24)
        results["baseline_caption"] = gpt2_out

        # 3. YOLOS Object Detection
        self._load_detector()
        det_out = self._detector.detect(image, threshold=det_threshold)
        detections = det_out["detections"]
        results["detection"] = det_out

        # 4. Annotated image
        results["annotated_image"] = self._detector.annotate_image(image, detections)

        # 5. Fast Batch Attribute Classification per ROI
        self._load_attr()
        from configs.config import ATTRIBUTE_VOCAB as _VOCAB
        import numpy as np
        import torch

        enhanced_detections = []
        if detections:
            # Prepare all crops
            crops = []
            for det in detections:
                xmin, ymin, xmax, ymax = [max(0, int(b)) for b in det["box"]]
                w, h = image.size
                xmax = min(w, max(xmin + 10, xmax))
                ymax = min(h, max(ymin + 10, ymax))
                crop = image.crop((xmin, ymin, xmax, ymax))
                crops.append(crop if crop.size[0] >= 10 and crop.size[1] >= 10 else image)

            # Precompute text embeddings once for the top attributes
            top_attrs = _VOCAB[:20]
            txt_prompts = [f"a photo of a {a}" for a in top_attrs]
            txt_inputs = self._clip_processor(text=txt_prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                txt_feats = self._clip_model.get_text_features(**txt_inputs)
                txt_feats = txt_feats.pooler_output if hasattr(txt_feats, "pooler_output") else txt_feats
                txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

                # Batch embed all crops at once
                img_inputs = self._clip_processor(images=crops, return_tensors="pt").to(self.device)
                crop_feats = self._clip_model.get_image_features(**img_inputs)
                crop_feats = crop_feats.pooler_output if hasattr(crop_feats, "pooler_output") else crop_feats
                crop_feats = crop_feats / crop_feats.norm(dim=-1, keepdim=True)

                # Cosine similarities: (num_crops, num_attrs)
                sim_matrix = (crop_feats @ txt_feats.T).softmax(dim=-1).cpu().numpy()

            for idx, det in enumerate(detections):
                top3_idx = np.argsort(sim_matrix[idx])[::-1][:3]
                attrs = [{"attribute": top_attrs[i], "confidence": round(float(sim_matrix[idx][i]), 3)} for i in top3_idx]
                enhanced_detections.append({**det, "id": idx + 1, "attributes": attrs})

        results["enhanced_detections"] = enhanced_detections

        # 6. Relationship Prediction
        self._load_rel()
        triplets, rel_lat = self._rel.predict(image, enhanced_detections)
        results["triplets"] = triplets

        # 7. Scene Graph
        from src.scene_graph import build_scene_graph, render_scene_graph, figure_to_pil, scene_graph_to_json
        G = build_scene_graph(enhanced_detections, triplets)
        sg_fig = render_scene_graph(G)
        sg_pil = figure_to_pil(sg_fig)
        results["scene_graph"] = G
        results["scene_graph_figure"] = sg_pil   # PIL Image
        results["scene_graph_image"]  = sg_pil   # alias used in app.py
        results["scene_graph_json"] = scene_graph_to_json(G)
        import matplotlib.pyplot as plt
        plt.close(sg_fig)

        # 8. Visual Question Answering
        if question and question.strip():
            self._load_vqa()
            vqa_out = self._vqa.answer(image, question)
        else:
            vqa_out = {"question": "", "answer": "No question provided.", "latency_s": 0}
        results["vqa"] = vqa_out

        # 9. Text-Image Retrieval
        if retrieval_catalog:
            self._load_retrieval(retrieval_catalog)
            results["retrieval"] = self._retrieval.search(
                blip_out["caption"], catalog=retrieval_catalog, top_k=min(5, len(retrieval_catalog))
            )
        else:
            results["retrieval"] = []

        results["total_latency_s"] = round(time.time() - t_total, 3)
        return results
