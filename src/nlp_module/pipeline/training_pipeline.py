# src/structured_ml/pipeline/training_pipeline.py

from src.nlp_module.components.text_ingestion import TextIngestion
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
import sys


if __name__ == "__main__":
    obj=TextIngestion()
    train_text_path, test_text_path=obj.initiate_text_ingestion()
    print(train_text_path, test_text_path)