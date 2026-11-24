# Vision-Language Framework for Multimodal Scene Understanding
---

## Project Overview
An end-to-end Vision-Language multimodal framework that processes any uploaded image through:
- **BLIP** (ViT-B/16 + Transformer Decoder) — SOTA captioning, fine-tunable on Visual Genome
- **YOLOS** (pure Vision Transformer detector) — object detection + bounding boxes
- **CLIP ViT** — zero-shot semantic attribute prediction & image-text retrieval
- **BLIP-VQA** — visual question answering
- **NetworkX** — semantic scene graph construction & visualization
- **Streamlit** — professional dark-themed interactive dashboard


DOWNLOAD THE ZIP FILE ------


## Installation
```bash
cd vision-language-framework
pip install -r requirements.txt
```

## Run Dashboard
```bash
cd vision-language-framework
python -m streamlit run frontend/app.py --server.port 8502
```

## Run Tests
```bash
cd vision-language-framework
python tests/run_tests.py
```

## Visual Genome Dataset
Set the `VG_ROOT` environment variable to your VG download directory:
```bash
set VG_ROOT=C:\path\to\visual_genome
python -c "from src.data_loader import build_manifest; ..."
```
Without VG, the app auto-generates a synthetic benchmark catalog for immediate use.

## Project Structure
```
vision-language-framework/
├── configs/config.py           ← All hyperparameters & paths
├── src/
│   ├── data_loader.py          ← VG manifest builder + PyTorch Dataset
│   ├── caption_model.py        ← BLIP + ViT-GPT2 captioners
│   ├── object_detector.py      ← YOLOS pure-ViT detector
│   ├── attribute_classifier.py ← CLIP ROI attribute head
│   ├── relationship_model.py   ← CLIP pairwise relation predictor
│   ├── scene_graph.py          ← NetworkX graph builder + renderer
│   ├── vqa.py                  ← BLIP-VQA module
│   ├── retrieval.py            ← CLIP embedding index + search
│   ├── evaluation.py           ← BLEU, ROUGE-L, Precision, Recall
│   └── inference.py            ← Unified pipeline engine
├── frontend/app.py             ← Streamlit dashboard
├── tests/run_tests.py          ← Automated test suite
├── requirements.txt
└── README.md
```

## Architecture
```
Image Input → BLIP ViT Encoder → Transformer Decoder → Caption
           → YOLOS ViT         → Bounding Boxes
           → CLIP ViT ROI      → Attributes per Object
           → CLIP Pairwise     → Relationship Triplets
           → NetworkX          → Scene Graph
           → BLIP-VQA          → Free-form Answers
           → CLIP Index        → Image Retrieval
```

