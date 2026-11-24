"""
Image Captioning Module
Primary: BLIP (ViT-B/16 + Transformer Decoder, Visual Genome fine-tuned)
Baseline: ViT-GPT2
"""

import os
import time
import logging
import torch
from PIL import Image
from typing import Optional

logger = logging.getLogger(__name__)


class BLIPCaptioner:
    """
    SOTA captioner using Salesforce BLIP.
    Pretrained on Visual Genome + COCO.
    """
    MODEL_NAME = "Salesforce/blip-image-captioning-base"

    def __init__(self, device: Optional[str] = None, fine_tuned_ckpt: Optional[str] = None):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading BLIP captioner on %s …", self.device)
        self.processor = BlipProcessor.from_pretrained(self.MODEL_NAME)
        self.model = BlipForConditionalGeneration.from_pretrained(self.MODEL_NAME).to(self.device)
        if fine_tuned_ckpt and os.path.exists(fine_tuned_ckpt):
            state = torch.load(fine_tuned_ckpt, map_location=self.device)
            self.model.load_state_dict(state)
            logger.info("Loaded fine-tuned weights from %s", fine_tuned_ckpt)
        self.model.eval()

    def caption(self, image: Image.Image, num_beams: int = 1, max_length: int = 48) -> dict:
        t0 = time.time()
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=1.2,
            )
        text = self.processor.decode(out[0], skip_special_tokens=True).strip().capitalize()
        return {"caption": text, "latency_s": round(time.time() - t0, 3), "model": "BLIP-ViT-B"}

    def fine_tune(self, train_loader, val_loader, num_epochs: int = 3, lr: float = 5e-5,
                  ckpt_dir: str = "checkpoints", patience: int = 3) -> list:
        """Full training loop with AdamW, LR warmup, gradient clipping, early stopping."""
        from transformers import get_linear_schedule_with_warmup
        from torch.optim import AdamW

        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = num_epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, 100, total_steps)

        best_val_loss = float("inf")
        no_improve = 0
        history = []

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                pixel_values  = batch["pixel_values"].to(self.device)
                input_ids     = batch["input_ids"].to(self.device)
                attention_mask= batch["attention_mask"].to(self.device)
                labels        = batch["labels"].to(self.device)
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg_train = epoch_loss / len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = self.model(
                        pixel_values=batch["pixel_values"].to(self.device),
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                        labels=batch["labels"].to(self.device),
                    )
                    val_loss += outputs.loss.item()
            avg_val = val_loss / len(val_loader)
            history.append({"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val})
            logger.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f", epoch + 1, num_epochs, avg_train, avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                no_improve = 0
                ckpt_path = os.path.join(ckpt_dir, "blip_caption_best.pt")
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info("Checkpoint saved → %s", ckpt_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        return history


class ViTGPT2Captioner:
    """
    Baseline captioner: nlpconnect/vit-gpt2-image-captioning.
    Kept for comparison purposes only.
    """
    MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

    def __init__(self, device: Optional[str] = None):
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained(self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()

    def caption(self, image: Image.Image, num_beams: int = 1, max_length: int = 24) -> dict:
        t0 = time.time()
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=2,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip().capitalize()
        return {"caption": text, "latency_s": round(time.time() - t0, 3), "model": "ViT-GPT2"}

