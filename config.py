"""
Central configuration for Vision-Language Framework.
Adjust paths and hyperparameters here — no other file needs to change.
"""

import os
from dataclasses import dataclass, field
from typing import List

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
CKPT_DIR   = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Visual Genome dataset root (set VG_ROOT env-var or edit here)
VG_ROOT = os.environ.get("VG_ROOT", os.path.join(DATA_DIR, "visual_genome"))

# ── Dataset ────────────────────────────────────────────────────────────────────
VG_IMAGES_DIR         = os.path.join(VG_ROOT, "images")
VG_REGION_DESC_FILE   = os.path.join(VG_ROOT, "region_descriptions.json")
VG_OBJECTS_FILE       = os.path.join(VG_ROOT, "objects.json")
VG_ATTRIBUTES_FILE    = os.path.join(VG_ROOT, "attributes.json")
VG_RELATIONSHIPS_FILE = os.path.join(VG_ROOT, "relationships.json")
VG_QA_FILE            = os.path.join(VG_ROOT, "question_answers.json")
VG_IMAGE_DATA_FILE    = os.path.join(VG_ROOT, "image_data.json")

MANIFEST_FILE = os.path.join(DATA_DIR, "manifest.json")

DATASET_SUBSET_SIZE = 2000     # max images to use; set None for all
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# ── Model names ────────────────────────────────────────────────────────────────
CAPTION_MODEL_NAME   = "Salesforce/blip-image-captioning-base"
DETECTOR_MODEL_NAME  = "hustvl/yolos-tiny"          # pure ViT detector
VQA_MODEL_NAME       = "Salesforce/blip-vqa-base"
CLIP_MODEL_NAME      = "openai/clip-vit-base-patch32"

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 8
NUM_EPOCHS      = 3
LEARNING_RATE   = 5e-5
WARMUP_STEPS    = 100
GRAD_CLIP       = 1.0
EARLY_STOP_PAT  = 3
LOG_STEPS       = 50

# ── Attribute vocabulary (Visual Genome top-100 attributes) ───────────────────
ATTRIBUTE_VOCAB: List[str] = [
    "white", "black", "brown", "red", "green", "blue", "yellow",
    "orange", "gray", "pink", "purple", "silver", "gold",
    "large", "small", "big", "little", "long", "short", "tall", "wide",
    "wooden", "metal", "metallic", "glass", "plastic", "leather",
    "fluffy", "furry", "smooth", "rough", "shiny", "dark", "bright",
    "sitting", "standing", "lying", "running", "jumping",
    "open", "closed", "empty", "full", "old", "young", "dry", "wet",
    "round", "rectangular", "square", "striped", "spotted", "plain"
]

# ── Relationship vocabulary (Visual Genome top-50 predicates) ─────────────────
RELATION_VOCAB: List[str] = [
    "on", "in", "has", "near", "with", "behind", "under", "above",
    "in front of", "next to", "inside", "on top of", "beside",
    "holding", "wearing", "carrying", "eating", "drinking", "looking at",
    "sitting on", "standing on", "lying on", "walking on", "riding",
    "attached to", "hanging from", "made of", "covered in", "part of",
    "belonging to", "between", "along", "at", "around", "across",
    "over", "through", "along with", "inside of", "below"
]

# ── Object Detection ───────────────────────────────────────────────────────────
DETECT_THRESHOLD = 0.25
MAX_DETECTIONS   = 10

# ── Dashboard ──────────────────────────────────────────────────────────────────
DASHBOARD_TITLE    = "ADVANCED MULTIMODAL SCENE ANALYTICS DASHBOARD"
DASHBOARD_SUBTITLE = "Vision-Language Joint Framework for Multimodal Scene Understanding"
DASHBOARD_PORT     = 8502
