# src/nlp_module/pipeline/training_pipeline.py

import sys
import pandas as pd

from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging
from src.nlp_module.components.tokenizer_pipeline import TokenizerPipeline
from src.nlp_module.components.dataset_builder import DatasetBuilder


if __name__ == "__main__":

    try:
        logging.info("Starting NLP Training Pipeline")

        # ------------------------------------------------
        # Paths (already saved during ingestion)
        # ------------------------------------------------
        train_text_path = "artifacts/nlp/train.csv"
        test_text_path = "artifacts/nlp/test.csv"

        # ------------------------------------------------
        # Load CSV files
        # ------------------------------------------------
        train_df = pd.read_csv(train_text_path)
        test_df = pd.read_csv(test_text_path)

        logging.info(f"Train Shape: {train_df.shape}")
        logging.info(f"Test Shape: {test_df.shape}")

        # ------------------------------------------------
        # CHANGE THIS to your actual text column name
        # ------------------------------------------------
        TEXT_COLUMN = "review"
        TARGET_COLUMN = "sentiment"

        if TEXT_COLUMN not in train_df.columns:
            raise Exception(f"{TEXT_COLUMN} not found in train dataset")

        if TARGET_COLUMN not in train_df.columns:
            raise Exception(f"{TARGET_COLUMN} not found in train dataset")

        # Extract text column
        train_text = train_df[TEXT_COLUMN]
        test_text = test_df[TEXT_COLUMN]

        logging.info("Text column extracted successfully")

        # ------------------------------------------------
        # Initialize Tokenizer
        # ------------------------------------------------
        tokenizer_obj = TokenizerPipeline()

        X_train_arr, X_test_arr, tokenizer_path = (
            tokenizer_obj.initiate_tokenizer_transformation(
                train_text=train_text,
                test_text=test_text
            )
        )

        logging.info("Tokenizer transformation completed successfully")
        logging.info(f"Tokenizer saved at: {tokenizer_path}")

        print("Train TF-IDF shape:", X_train_arr.shape)
        print("Test TF-IDF shape:", X_test_arr.shape)

        data_builder_obj = DatasetBuilder()
        train_path, test_path = data_builder_obj.build_dataset(
            input_file_path="artifacts/nlp/raw.csv"
        )

    except Exception as e:
        logging.error("Exception in NLP training pipeline")
        raise CustomException(e, sys)