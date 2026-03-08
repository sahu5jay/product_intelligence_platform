# src/nlp_module/components/model_loader.py

import os
import sys
from pathlib import Path
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.config_loader import load_config

# Base directories
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src" / "nlp_module" / "config.yaml"
config = load_config(CONFIG_PATH)


@dataclass
class ModelLoaderConfig:
    # HuggingFace model name or local path
    model_name: str = config.get("model", {}).get("model_type", "distilbert-base-uncased")
    model_dir: str = str(BASE_DIR / "artifacts" / "nlp_model")


class ModelLoader:
    def __init__(self, model_name: str = None, model_dir: str = None):
        try:
            self.config = ModelLoaderConfig()
            self.model_name = model_name or self.config.model_name
            self.model_dir = model_dir or self.config.model_dir
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            os.makedirs(self.model_dir, exist_ok=True)
        except Exception as e:
            logging.error("Error during ModelLoader initialization")
            raise CustomException(e, sys)

    def load_model(self):
        logging.info(f"Loading HuggingFace model: {self.model_name}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_dir)

            # Load pre-trained model
            model = AutoModel.from_pretrained(self.model_name, cache_dir=self.model_dir)
            model.to(self.device)
            model.eval()

            logging.info(f"Model loaded successfully on device: {self.device}")
            logging.info(f"Tokenizer and model cached at: {self.model_dir}")

            return tokenizer, model

        except Exception as e:
            logging.error(f"Error occurred while loading NLP model: {e}")
            raise CustomException(e, sys)