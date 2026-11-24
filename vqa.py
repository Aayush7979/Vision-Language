"""
Visual Question Answering Module
Uses Salesforce BLIP-VQA-base.
"""

import time
import logging
import torch
from PIL import Image
from typing import Optional

logger = logging.getLogger(__name__)


class BLIPVQAModel:
    MODEL_NAME = "Salesforce/blip-vqa-base"

    def __init__(self, device: Optional[str] = None):
        from transformers import BlipProcessor, BlipForQuestionAnswering
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading BLIP-VQA on %s …", self.device)
        self.processor = BlipProcessor.from_pretrained(self.MODEL_NAME)
        # Avoid .to(self.device) if already on CPU or meta-device
        try:
            self.model = BlipForQuestionAnswering.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False
            )
            if self.device != "cpu":
                self.model = self.model.to(self.device)
        except Exception:
            self.model = BlipForQuestionAnswering.from_pretrained(self.MODEL_NAME)
        self.model.eval()

    def answer(self, image: Image.Image, question: str) -> dict:
        t0 = time.time()
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=32, num_beams=4)
        answer = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return {
            "question": question,
            "answer": answer.capitalize() if answer else "No answer found.",
            "latency_s": round(time.time() - t0, 3),
        }

    SAMPLE_QUESTIONS = [
        "What animal is in the image?",
        "What color is the dog?",
        "Is the dog sitting or standing?",
        "What is the person holding?",
        "How many people are in the image?",
        "What is behind the main object?",
        "What is the weather like?",
        "What type of vehicle is shown?",
    ]

