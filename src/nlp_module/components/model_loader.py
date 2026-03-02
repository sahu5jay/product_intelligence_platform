import os
import sys
from pathlib import Path
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModel

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


BASE_DIR = Path(__file__).resolve().parents[3]


@dataclass
class ModelLoaderConfig:
    # HuggingFace model name OR local path
    model_name: str = "distilbert-base-uncased"

    # Optional: local directory to save/load model
    model_dir: str = str(BASE_DIR / "artifacts" / "nlp_model")


class ModelLoader:
    def __init__(self):
        self.config = ModelLoaderConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        logging.info("Model loading started")

        try:
            os.makedirs(self.config.model_dir, exist_ok=True)

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name
            )

            # Load model
            model = AutoModel.from_pretrained(
                self.config.model_name
            )

            model.to(self.device)
            model.eval()

            logging.info(f"Model loaded successfully on device: {self.device}")

            return tokenizer, model

        except Exception as e:
            logging.error("Error occurred while loading NLP model")
            raise CustomException(e, sys)