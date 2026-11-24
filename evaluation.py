"""
Evaluation Module
Computes BLEU, ROUGE-L, object detection Precision/Recall/mAP,
attribute F1, relationship accuracy, and retrieval Recall@K.
"""

import json
import logging
import os
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ── Captioning ────────────────────────────────────────────────────────────────
def compute_bleu(predictions: List[str], references: List[List[str]]) -> Dict:
    """BLEU-1 through BLEU-4 using sacrebleu if available, else corpus_bleu."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU()
        refs_transposed = list(zip(*references)) if references and len(references[0]) > 1 else [
            [r[0] for r in references]
        ]
        result = bleu.corpus_score(predictions, refs_transposed)
        return {"bleu": result.score}
    except ImportError:
        pass
    try:
        from nltk.translate.bleu_score import corpus_bleu
        refs = [[ref.split() for ref in row] for row in references]
        hyps = [p.split() for p in predictions]
        b1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0))
        b4 = corpus_bleu(refs, hyps, weights=(0.25,) * 4)
        return {"bleu1": round(b1, 4), "bleu4": round(b4, 4)}
    except ImportError:
        return {"bleu": None, "note": "Install sacrebleu or nltk for BLEU scores."}


def compute_rouge_l(predictions: List[str], references: List[str]) -> Dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(references, predictions)]
        return {"rouge_l": round(np.mean(scores), 4)}
    except ImportError:
        return {"rouge_l": None, "note": "Install rouge_score for ROUGE-L."}


# ── Detection ─────────────────────────────────────────────────────────────────
def _iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def compute_detection_metrics(
    predictions: List[List[Dict]],
    ground_truths: List[List[Dict]],
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Computes per-image precision / recall and macro-averages.
    predictions: list of detection lists per image, each detection has 'label' and 'box'.
    ground_truths: list of GT lists per image, each GT has 'names' and 'box' = [x,y,w,h].
    """
    precisions, recalls = [], []
    for preds, gts in zip(predictions, ground_truths):
        if not gts:
            continue
        matched = set()
        tp = 0
        for pred in preds:
            for gi, gt in enumerate(gts):
                if gi in matched:
                    continue
                # GT box format may be [x,y,w,h] → convert to [x1,y1,x2,y2]
                if len(gt["box"]) == 4:
                    gb = gt["box"]
                    gt_box = [gb[0], gb[1], gb[0] + gb[2], gb[1] + gb[3]]
                else:
                    gt_box = gt["box"]
                gt_labels = [n.lower() for n in gt.get("names", gt.get("name", [""]))]
                if pred["label"].lower() in gt_labels and _iou(pred["box"], gt_box) >= iou_threshold:
                    tp += 1
                    matched.add(gi)
                    break
        precision = tp / len(preds) if preds else 0.0
        recall = tp / len(gts)
        precisions.append(precision)
        recalls.append(recall)

    avg_p = float(np.mean(precisions)) if precisions else 0.0
    avg_r = float(np.mean(recalls)) if recalls else 0.0
    f1 = (2 * avg_p * avg_r / (avg_p + avg_r)) if (avg_p + avg_r) > 0 else 0.0
    return {"precision": round(avg_p, 4), "recall": round(avg_r, 4), "f1": round(f1, 4)}


# ── Attribute ─────────────────────────────────────────────────────────────────
def compute_attribute_metrics(
    predicted_attrs: List[List[str]],
    gt_attrs: List[List[str]],
) -> Dict:
    """Micro-averaged F1 for multi-label attribute classification."""
    tp = fp = fn = 0
    for pred_set, gt_set in zip(predicted_attrs, gt_attrs):
        p = set(a.lower() for a in pred_set)
        g = set(a.lower() for a in gt_set)
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ── Retrieval ─────────────────────────────────────────────────────────────────
def compute_recall_at_k(
    ranked_indices: List[List[int]],
    relevant: List[int],
    k_list: List[int] = [1, 5, 10],
) -> Dict:
    """Recall@K for text-to-image retrieval."""
    result = {}
    for k in k_list:
        hits = sum(1 for top_k, rel in zip(ranked_indices, relevant) if rel in top_k[:k])
        result[f"recall@{k}"] = round(hits / len(relevant), 4) if relevant else 0.0
    return result


# ── Report writer ─────────────────────────────────────────────────────────────
def save_report(metrics: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation report saved → %s", path)

