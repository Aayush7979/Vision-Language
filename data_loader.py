"""
Visual Genome Dataset Loader
Handles: image metadata, region descriptions, objects, attributes,
         relationships, QA, manifest generation, splits.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Low-level JSON helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe_load_json(path: str) -> Optional[list]:
    """Load a JSON file, return None on any error."""
    if not os.path.exists(path):
        logger.warning("File not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load %s: %s", path, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Manifest builder
# ─────────────────────────────────────────────────────────────────────────────
def build_manifest(
    vg_root: str,
    images_dir: str,
    image_data_file: str,
    region_desc_file: str,
    objects_file: str,
    attributes_file: str,
    relationships_file: str,
    qa_file: str,
    subset_size: Optional[int],
    seed: int = 42,
) -> Dict:
    """
    Build a dataset manifest dict:
      { image_id: { "file_name", "captions", "objects", "attributes",
                    "relationships", "qa" } }
    Only keeps entries that have an accessible image file.
    """
    logger.info("Building Visual Genome manifest from: %s", vg_root)

    # ── Image metadata ────────────────────────────────────────────────────────
    image_data = _safe_load_json(image_data_file)
    if image_data is None:
        raise FileNotFoundError(f"image_data.json not found at {image_data_file}")

    id_to_meta: Dict[int, dict] = {}
    for entry in image_data:
        iid = entry.get("image_id") or entry.get("id")
        if iid is None:
            continue
        fname = entry.get("file_name") or entry.get("url", "").rsplit("/", 1)[-1]
        id_to_meta[int(iid)] = {"file_name": fname, "url": entry.get("url", "")}

    # ── Region descriptions → image-level captions ────────────────────────────
    region_data = _safe_load_json(region_desc_file) or []
    id_to_captions: Dict[int, List[str]] = {}
    for item in region_data:
        iid = int(item["id"])
        phrases = [r["phrase"] for r in item.get("regions", []) if r.get("phrase")]
        if phrases:
            id_to_captions[iid] = phrases

    # ── Objects ───────────────────────────────────────────────────────────────
    objects_data = _safe_load_json(objects_file) or []
    id_to_objects: Dict[int, list] = {}
    for item in objects_data:
        iid = int(item["image_id"])
        id_to_objects[iid] = item.get("objects", [])

    # ── Attributes ────────────────────────────────────────────────────────────
    attrs_data = _safe_load_json(attributes_file) or []
    id_to_attrs: Dict[int, dict] = {}  # object_id → attribute list
    for item in attrs_data:
        for obj in item.get("attributes", []):
            oid = obj.get("object_id")
            if oid:
                id_to_attrs[oid] = obj.get("attributes", [])

    # ── Relationships ─────────────────────────────────────────────────────────
    rel_data = _safe_load_json(relationships_file) or []
    id_to_rels: Dict[int, list] = {}
    for item in rel_data:
        iid = int(item["image_id"])
        id_to_rels[iid] = item.get("relationships", [])

    # ── QA pairs ─────────────────────────────────────────────────────────────
    qa_data = _safe_load_json(qa_file) or []
    id_to_qa: Dict[int, list] = {}
    for item in qa_data:
        iid = int(item["id"])
        id_to_qa[iid] = item.get("qas", [])

    # ── Filter to available images ────────────────────────────────────────────
    available_ids = []
    for iid, meta in id_to_meta.items():
        img_path = os.path.join(images_dir, meta["file_name"])
        if os.path.exists(img_path):
            available_ids.append(iid)

    logger.info("Images found on disk: %d / %d", len(available_ids), len(id_to_meta))

    if subset_size and len(available_ids) > subset_size:
        rng = random.Random(seed)
        available_ids = rng.sample(available_ids, subset_size)

    manifest = {}
    for iid in available_ids:
        meta = id_to_meta[iid]
        # Build object list enriched with attributes
        objects = []
        for obj in id_to_objects.get(iid, []):
            oid = obj.get("object_id")
            names = obj.get("names", [obj.get("name", "object")])
            obj_attrs = id_to_attrs.get(oid, [])
            objects.append({
                "object_id": oid,
                "names": names,
                "x": obj.get("x", 0),
                "y": obj.get("y", 0),
                "w": obj.get("w", 0),
                "h": obj.get("h", 0),
                "attributes": obj_attrs,
            })

        captions = id_to_captions.get(iid, [])
        # Select the longest phrase as the best image-level caption
        image_caption = max(captions, key=len) if captions else ""

        manifest[iid] = {
            "image_id": iid,
            "file_name": meta["file_name"],
            "image_path": os.path.join(images_dir, meta["file_name"]),
            "image_caption": image_caption,
            "region_descriptions": captions,
            "objects": objects,
            "relationships": id_to_rels.get(iid, []),
            "qa_pairs": id_to_qa.get(iid, []),
        }

    logger.info("Manifest built: %d entries", len(manifest))
    return manifest


def save_manifest(manifest: Dict, path: str) -> None:
    """Save manifest to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    logger.info("Manifest saved to %s", path)


def load_manifest(path: str) -> Dict:
    """Load a previously saved manifest."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # JSON keys are strings; convert back to int
    return {int(k): v for k, v in raw.items()}


def split_manifest(
    manifest: Dict,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split image IDs into train / val / test by ID (no data leakage)."""
    ids = sorted(manifest.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train_ids = ids[:n_train]
    val_ids   = ids[n_train : n_train + n_val]
    test_ids  = ids[n_train + n_val:]
    logger.info("Split → Train:%d  Val:%d  Test:%d", len(train_ids), len(val_ids), len(test_ids))
    return train_ids, val_ids, test_ids


def dataset_statistics(manifest: Dict) -> Dict:
    """Return summary statistics of the loaded manifest."""
    total = len(manifest)
    with_caption = sum(1 for v in manifest.values() if v.get("image_caption"))
    total_objects = sum(len(v.get("objects", [])) for v in manifest.values())
    total_rels = sum(len(v.get("relationships", [])) for v in manifest.values())
    total_qa = sum(len(v.get("qa_pairs", [])) for v in manifest.values())
    avg_obj = total_objects / total if total else 0
    avg_rel = total_rels / total if total else 0
    return {
        "total_images": total,
        "with_caption": with_caption,
        "total_objects": total_objects,
        "total_relationships": total_rels,
        "total_qa_pairs": total_qa,
        "avg_objects_per_image": round(avg_obj, 2),
        "avg_relationships_per_image": round(avg_rel, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Fallback synthetic catalog (runs without VG download)
# ─────────────────────────────────────────────────────────────────────────────
def build_synthetic_catalog(images_dir: str, user_dog_path: Optional[str] = None) -> Dict:
    """
    Build a small catalog of synthetic + real scenes so the app works
    immediately without the full Visual Genome download.
    """
    from PIL import Image, ImageDraw
    import shutil

    os.makedirs(images_dir, exist_ok=True)
    catalog = {}

    def _add(iid, title, img_path, captions, objects, rels):
        catalog[iid] = {
            "image_id": iid,
            "title": title,
            "file_name": os.path.basename(img_path),
            "image_path": img_path,
            "image_caption": captions[0] if captions else "",
            "region_descriptions": captions,
            "objects": objects,
            "relationships": rels,
            "qa_pairs": [],
        }

    # Scene 0: User's real dog image (if present)
    if user_dog_path and os.path.exists(user_dog_path):
        dst = os.path.join(images_dir, "white_dog.png")
        shutil.copyfile(user_dog_path, dst)
        _add(1000, "White Spitz Dog in Green Lawn", dst,
             ["A fluffy white dog sitting in the grass",
              "A white Spitz with pointed ears on mowed lawn"],
             [{"object_id": 1, "names": ["dog"], "x": 50, "y": 42, "w": 392, "h": 336,
               "attributes": ["white", "fluffy", "sitting"]}],
             [])

    # Scene 1: Street
    p1 = os.path.join(images_dir, "street_scene.jpg")
    if not os.path.exists(p1):
        img = Image.new("RGB", (640, 480), (135, 206, 235))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 280, 640, 480], fill=(70, 75, 80))
        d.rectangle([200, 280, 460, 390], fill=(220, 38, 38))
        d.rectangle([250, 220, 410, 280], fill=(180, 20, 20))
        d.ellipse([220, 375, 265, 420], fill=(20, 20, 20))
        d.ellipse([395, 375, 440, 420], fill=(20, 20, 20))
        d.rectangle([60, 170, 100, 310], fill=(115, 60, 30))
        d.ellipse([25, 80, 135, 200], fill=(34, 139, 34))
        img.save(p1)
    _add(1001, "Urban Road with Red Car & Tree", p1,
         ["A red car driving on an asphalt road", "A green tree beside the road"],
         [{"object_id": 101, "names": ["car"], "x": 200, "y": 220, "w": 260, "h": 200, "attributes": ["red", "metallic"]},
          {"object_id": 102, "names": ["tree"], "x": 25, "y": 80, "w": 110, "h": 230, "attributes": ["green", "tall"]}],
         [{"relationship_id": 1001, "predicate": "on",
           "subject": {"object_id": 101, "name": "car"},
           "object": {"object_id": 999, "name": "road"}}])

    # Scene 2: Living Room
    p2 = os.path.join(images_dir, "living_room.jpg")
    if not os.path.exists(p2):
        img = Image.new("RGB", (640, 480), (245, 235, 220))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 320, 640, 480], fill=(160, 110, 75))
        d.rectangle([80, 200, 560, 360], fill=(30, 58, 138))
        d.rectangle([60, 220, 115, 360], fill=(23, 37, 84))
        d.rectangle([525, 220, 580, 360], fill=(23, 37, 84))
        d.rectangle([190, 360, 450, 440], fill=(107, 68, 35))
        d.rectangle([280, 340, 370, 365], fill=(220, 38, 38))
        img.save(p2)
    _add(1002, "Living Room — Sofa, Table & Book", p2,
         ["A blue sofa in a living room", "A wooden coffee table in front of the couch",
          "A red book resting on the table"],
         [{"object_id": 201, "names": ["sofa"], "x": 60, "y": 200, "w": 520, "h": 160, "attributes": ["blue", "large"]},
          {"object_id": 202, "names": ["table"], "x": 190, "y": 360, "w": 260, "h": 80, "attributes": ["wooden", "brown"]},
          {"object_id": 203, "names": ["book"], "x": 280, "y": 340, "w": 90, "h": 25, "attributes": ["red", "small"]}],
         [{"relationship_id": 1002, "predicate": "in front of",
           "subject": {"object_id": 202, "name": "table"},
           "object": {"object_id": 201, "name": "sofa"}},
          {"relationship_id": 1003, "predicate": "on",
           "subject": {"object_id": 203, "name": "book"},
           "object": {"object_id": 202, "name": "table"}}])

    # Scene 3: Dining table with food
    p3 = os.path.join(images_dir, "dining_scene.jpg")
    if not os.path.exists(p3):
        img = Image.new("RGB", (640, 480), (230, 240, 245))
        d = ImageDraw.Draw(img)
        d.rectangle([50, 200, 590, 460], fill=(205, 170, 125))
        d.ellipse([150, 240, 350, 410], fill=(255, 255, 255), outline=(200, 200, 200), width=4)
        d.ellipse([200, 290, 300, 365], fill=(220, 38, 38))
        d.rectangle([400, 140, 450, 350], fill=(16, 120, 70))
        d.rectangle([400, 100, 450, 140], fill=(16, 100, 60))
        d.rectangle([350, 270, 400, 345], fill=(245, 245, 245))
        img.save(p3)
    _add(1003, "Dining Table — Apple, Plate & Bottle", p3,
         ["A red apple on a round white plate", "A tall green bottle beside a cup",
          "A dining table set with refreshments"],
         [{"object_id": 301, "names": ["apple"], "x": 200, "y": 290, "w": 100, "h": 75, "attributes": ["red", "round"]},
          {"object_id": 302, "names": ["plate"], "x": 150, "y": 240, "w": 200, "h": 170, "attributes": ["white", "round"]},
          {"object_id": 303, "names": ["bottle"], "x": 400, "y": 100, "w": 50, "h": 250, "attributes": ["green", "tall", "glass"]}],
         [{"relationship_id": 2001, "predicate": "on",
           "subject": {"object_id": 301, "name": "apple"},
           "object": {"object_id": 302, "name": "plate"}}])

    return catalog


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────
class VisualGenomeCaptioningDataset(Dataset):
    """
    Returns pixel_values + tokenized label for image captioning fine-tuning.
    """
    def __init__(self, manifest_subset: List[dict], blip_processor, max_target_length: int = 64):
        self.items = manifest_subset
        self.processor = blip_processor
        self.max_len = max_target_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        caption = item.get("image_caption", "A scene.")

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["input_ids"].squeeze(0).clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }

