"""
Automated test suite for the Vision-Language Framework.
Run:  python tests/run_tests.py
"""

import os
import sys
import unittest
from PIL import Image

# Ensure src is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)


class TestDataLoader(unittest.TestCase):
    def test_synthetic_catalog_builds(self):
        from src.data_loader import build_synthetic_catalog
        img_dir = os.path.join(_ROOT, "data", "test_catalog_images")
        catalog = build_synthetic_catalog(img_dir)
        self.assertGreater(len(catalog), 0, "Catalog must have at least one entry.")
        for k, v in catalog.items():
            self.assertTrue(os.path.exists(v["image_path"]), f"Image missing: {v['image_path']}")

    def test_manifest_split(self):
        from src.data_loader import split_manifest
        manifest = {i: {} for i in range(100)}
        train_ids, val_ids, test_ids = split_manifest(manifest)
        self.assertEqual(len(train_ids) + len(val_ids) + len(test_ids), 100)
        # No overlap
        all_ids = set(train_ids) | set(val_ids) | set(test_ids)
        self.assertEqual(len(all_ids), 100)


class TestDetector(unittest.TestCase):
    def test_yolos_loads_and_detects(self):
        from src.object_detector import YOLOSDetector
        detector = YOLOSDetector()
        img = Image.new("RGB", (640, 480), (200, 100, 50))
        result = detector.detect(img)
        self.assertIn("detections", result)
        self.assertIn("latency_s", result)

    def test_annotation(self):
        from src.object_detector import YOLOSDetector
        detector = YOLOSDetector()
        img = Image.new("RGB", (320, 240), (100, 100, 200))
        detections = [{"label": "cat", "confidence": 0.8, "box": [10, 10, 100, 100]}]
        ann = detector.annotate_image(img, detections)
        self.assertEqual(ann.size, (320, 240))


class TestCaptioner(unittest.TestCase):
    def test_blip_captions(self):
        from src.caption_model import BLIPCaptioner
        captioner = BLIPCaptioner()
        img = Image.new("RGB", (224, 224), (50, 150, 50))
        result = captioner.caption(img, num_beams=2, max_length=16)
        self.assertIn("caption", result)
        self.assertIsInstance(result["caption"], str)
        self.assertGreater(len(result["caption"]), 0)

    def test_vit_gpt2_captions(self):
        from src.caption_model import ViTGPT2Captioner
        captioner = ViTGPT2Captioner()
        img = Image.new("RGB", (224, 224), (50, 50, 200))
        result = captioner.caption(img, num_beams=2, max_length=12)
        self.assertIn("caption", result)


class TestVQA(unittest.TestCase):
    def test_blip_vqa_answers(self):
        from src.vqa import BLIPVQAModel
        vqa = BLIPVQAModel()
        img = Image.new("RGB", (224, 224), (200, 200, 100))
        result = vqa.answer(img, "What color is the background?")
        self.assertIn("answer", result)
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 0)


class TestSceneGraph(unittest.TestCase):
    def test_builds_and_renders(self):
        from src.scene_graph import build_scene_graph, render_scene_graph, figure_to_pil
        detections = [
            {"id": 1, "label": "dog", "confidence": 0.9, "box": [10, 10, 100, 100],
             "attributes": [{"attribute": "white", "confidence": 0.8}]},
        ]
        triplets = [{"subject": "dog", "subject_id": 1,
                     "predicate": "sitting on", "object": "grass", "object_id": -1,
                     "confidence": 0.75}]
        G = build_scene_graph(detections, triplets)
        self.assertGreater(len(G.nodes), 0)
        self.assertGreater(len(G.edges), 0)
        fig = render_scene_graph(G)
        pil_img = figure_to_pil(fig)
        self.assertIsNotNone(pil_img)


class TestEvaluation(unittest.TestCase):
    def test_bleu(self):
        from src.evaluation import compute_bleu
        preds = ["a white dog sitting in the grass"]
        refs  = [["a white fluffy dog sitting in the lawn"]]
        result = compute_bleu(preds, refs)
        self.assertIsInstance(result, dict)

    def test_attribute_metrics(self):
        from src.evaluation import compute_attribute_metrics
        preds = [["white", "fluffy"]]
        gts   = [["white", "sitting", "fluffy"]]
        m = compute_attribute_metrics(preds, gts)
        self.assertIn("f1", m)

    def test_detection_metrics(self):
        from src.evaluation import compute_detection_metrics
        preds = [[{"label": "dog", "box": [10, 10, 100, 100]}]]
        gts   = [[{"names": ["dog"], "box": [10, 10, 90, 90]}]]
        m = compute_detection_metrics(preds, gts)
        self.assertIn("precision", m)


if __name__ == "__main__":
    unittest.main(verbosity=2)

