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
from src.nlp_module.components.trainer import Trainer
from src.nlp_module.components.evaluator import Evaluator

from src.shared_utils.config_loader import load_config
from src.shared_utils.constants import NLP_ARTIFACTS, NLP_RAW_DATA_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH
from src.shared_utils.constants import CLEANED_DATA_PATH, RAW_DATA_PATH
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
# RAW_DATA_PATH = NLP_ARTIFACTS / "data" / "raw.csv"
# PROCESSED_DATA_PATH = NLP_ARTIFACTS / "data" / "processed.csv"

if __name__ == "__main__":

    try:
        logging.info("Starting NLP Training Pipeline")

        # -------------------------
        # Step 1: Text Ingestion
        # -------------------------
        text_ingestion = TextIngestion(dataset_path = NLP_RAW_DATA_DIR, raw_path = RAW_DATA_PATH)
        raw_file_path = text_ingestion.initiate_text_ingestion()
        logging.info(f"Raw dataset ingested at: {raw_file_path}")

        # -------------------------
        # Step 2: Text Cleaning
        # -------------------------
        logging.info("Step 2: Text Cleaning Started")
        text_cleaner = TextCleaning(raw_data_path=RAW_DATA_PATH, processed_data_path=CLEANED_DATA_PATH)
        processed_csv_path = text_cleaner.initiate_text_cleaning()
        # logging.info(f"Processed dataset saved at {processed_csv_path}")

        # -------------------------
        # Step 3: Build Dataset
        # -------------------------
        
        dataset_builder = DatasetBuilder(processed_csv_path= CLEANED_DATA_PATH)
        train_path, test_path = dataset_builder.build_dataset()
        logging.info(f"Train dataset: {train_path}, Test dataset: {test_path}")

        # -------------------------
        # Step 4: Load Train/Test
        # -------------------------
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)

        X_train_text = train_df[TEXT_COLUMN]
        y_train = train_df[TARGET_COLUMN]
        X_test_text = test_df[TEXT_COLUMN]
        y_test = test_df[TARGET_COLUMN]

        # -------------------------
        # Step 5: Tokenization / TF-IDF
        # -------------------------
        tokenizer_obj = TokenizerPipeline()

        X_train_arr, X_test_arr, tokenizer_path = tokenizer_obj.initiate_tokenizer_transformation(
            train_text=X_train_text,
            test_text=X_test_text
        )

        logging.info(f"Tokenizer saved at: {tokenizer_path}")
        logging.info(f"Train TF-IDF shape: {X_train_arr.shape}, Test TF-IDF shape: {X_test_arr.shape}")
        # -------------------------
        # Step 6: Train Model
        # -------------------------

        trainer = Trainer(config)

        model = trainer.initiate_model_training(
            X_train=X_train_arr,
            y_train=y_train
        )

        logging.info("Model training completed")

        evaluator = Evaluator(config)

        metrics = evaluator.evaluate(
            model=model,
            X_test=X_test_arr,
            y_test=y_test
        )

        logging.info(f"Model evaluation completed: {metrics}")

    except Exception as e:
        logging.error("Exception in NLP training pipeline")
        raise CustomException(e, sys)