# src/nlp_module/pipeline/training_pipeline.py

import sys
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging
from src.nlp_module.components.text_ingestion import TextIngestion
from src.nlp_module.components.text_cleaning import TextCleaning
from src.nlp_module.components.tokenizer_pipeline import TokenizerPipeline
from src.nlp_module.components.dataset_builder import DatasetBuilder
from src.shared_utils.config_loader import load_config
from src.shared_utils.constants import NLP_ARTIFACTS
from pathlib import Path

# -------------------------
# Load YAML Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = BASE_DIR / "src" / "nlp_module" / "config.yaml"
config = load_config(CONFIG_PATH)

TEXT_COLUMN = "review"
TARGET_COLUMN = "sentiment"

# Define raw and processed paths from config/constants
RAW_DATA_PATH = NLP_ARTIFACTS / "data" / "raw.csv"
PROCESSED_DATA_PATH = NLP_ARTIFACTS / "data" / "processed.csv"

if __name__ == "__main__":

    try:
        logging.info("Starting NLP Training Pipeline")

        # -------------------------
        # Step 1: Text Ingestion
        # -------------------------
        text_ingestion = TextIngestion()
        raw_file_path = text_ingestion.initiate_text_ingestion()
        logging.info(f"Raw dataset ingested at: {raw_file_path}")

        # -------------------------
        # Step 2: Text Cleaning
        # -------------------------
        logging.info("Step 2: Text Cleaning Started")
        text_cleaner = TextCleaning(raw_data_path=RAW_DATA_PATH, processed_data_path=PROCESSED_DATA_PATH)
        processed_csv_path = text_cleaner.initiate_text_cleaning()
        # logging.info(f"Processed dataset saved at {processed_csv_path}")

        # -------------------------
        # Step 3+: Continue with DatasetBuilder, Tokenizer, Model Training, Evaluation
        # -------------------------
        # dataset_builder = DatasetBuilder(...)
        # tokenizer_obj = TokenizerPipeline(...)
        # model training and evaluation...

    except Exception as e:
        logging.error("Exception in NLP training pipeline")
        raise CustomException(e, sys)