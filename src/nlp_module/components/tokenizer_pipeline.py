# src/nlp_module/components/tokenizer_pipeline.py

import os
import sys
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging
from src.shared_utils.save_object import save_object


# ==========================================
# Config
# ==========================================

@dataclass
class TokenizerPipelineConfig:
    tokenizer_obj_file_path: str = os.path.join(
        "artifacts", "nlp", "tokenizer.pkl"
    )


# ==========================================
# Tokenizer Pipeline Class
# ==========================================

class TokenizerPipeline:

    def __init__(self):
        self.config = TokenizerPipelineConfig()

    # --------------------------------------
    # Create TF-IDF Pipeline
    # --------------------------------------
    def get_tokenizer_pipeline(self):

        try:
            logging.info("Creating TF-IDF tokenizer pipeline")

            tokenizer_pipeline = Pipeline(
                steps=[
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=5000,
                            ngram_range=(1, 2),
                            stop_words="english"
                        )
                    )
                ]
            )

            return tokenizer_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    # --------------------------------------
    # Fit & Transform Text
    # --------------------------------------
    def initiate_tokenizer_transformation(
        self,
        train_text,
        test_text=None
    ):

        try:
            logging.info("Starting tokenizer transformation")

            tokenizer_pipeline = self.get_tokenizer_pipeline()

            X_train_arr = tokenizer_pipeline.fit_transform(train_text)

            if test_text is not None:
                X_test_arr = tokenizer_pipeline.transform(test_text)
            else:
                X_test_arr = None

            os.makedirs(
                os.path.dirname(self.config.tokenizer_obj_file_path),
                exist_ok=True
            )

            save_object(
                file_path=self.config.tokenizer_obj_file_path,
                obj=tokenizer_pipeline
            )

            logging.info(f"Tokenizer saved at {self.config.tokenizer_obj_file_path}")

            return X_train_arr, X_test_arr, self.config.tokenizer_obj_file_path

        except Exception as e:
            logging.error("Error in tokenizer transformation")
            raise CustomException(e, sys)