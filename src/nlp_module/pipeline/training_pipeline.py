# src/nlp_module/pipeline/training_pipeline.py

import sys
import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging
from src.nlp_module.components.tokenizer_pipeline import TokenizerPipeline
from src.nlp_module.components.evaluator import ModelEvaluator
from src.nlp_module.components.inference_engine import SentimentInferenceEngine


if __name__ == "__main__":

    try:
        logging.info("Starting NLP Training Pipeline")

        # ------------------------------------------------
        # Load Data
        # ------------------------------------------------
        train_path = "artifacts/nlp/train.csv"
        test_path = "artifacts/nlp/test.csv"

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logging.info(f"Train Shape: {train_df.shape}")
        logging.info(f"Test Shape: {test_df.shape}")

        TEXT_COLUMN = "review"
        TARGET_COLUMN = "sentiment"

        if TEXT_COLUMN not in train_df.columns:
            raise Exception(f"{TEXT_COLUMN} not found in train dataset")

        if TARGET_COLUMN not in train_df.columns:
            raise Exception(f"{TARGET_COLUMN} not found in train dataset")

        X_train_text = train_df[TEXT_COLUMN]
        y_train = train_df[TARGET_COLUMN]

        X_test_text = test_df[TEXT_COLUMN]
        y_test = test_df[TARGET_COLUMN]

        logging.info("Text and Target columns extracted successfully")

        # ------------------------------------------------
        # Tokenization + TF-IDF
        # ------------------------------------------------
        tokenizer_obj = TokenizerPipeline()

        X_train_arr, X_test_arr, tokenizer_path = (
            tokenizer_obj.initiate_tokenizer_transformation(
                train_text=X_train_text,
                test_text=X_test_text
            )
        )

        logging.info("TF-IDF transformation completed")
        logging.info(f"Tokenizer saved at: {tokenizer_path}")

        print("Train TF-IDF shape:", X_train_arr.shape)
        print("Test TF-IDF shape:", X_test_arr.shape)

        # ------------------------------------------------
        # Train Model
        # ------------------------------------------------
        model = LogisticRegression(max_iter=500)
        model.fit(X_train_arr, y_train)

        logging.info("Model training completed")

        # ------------------------------------------------
        # Evaluate Model
        # ------------------------------------------------
        y_pred = model.predict(X_test_arr)
        accuracy = accuracy_score(y_test, y_pred)

        logging.info(f"Model Accuracy: {accuracy}")
        print("Model Accuracy:", accuracy)

        # ------------------------------------------------
        # Save Model
        # ------------------------------------------------
        os.makedirs("artifacts/nlp/model", exist_ok=True)

        model_path = "artifacts/nlp/sentiment_model.pkl"
        joblib.dump(model, model_path)

        logging.info(f"Model saved at: {model_path}")

        print("Training Completed Successfully")

        model_eval_obj = ModelEvaluator()

        model_evaluation_dict = model_eval_obj.evaluate_model(
            model=model,
            X_test=X_test_arr,
            y_test=y_test
        )

        print("Evaluation Metrics:", model_evaluation_dict)

        text = "This movie was absolutely bad"

        sentiment_infrence_obj = SentimentInferenceEngine()
        result = sentiment_infrence_obj.predict_sentiment(text)
        logging.info(f"----->>>{result}")

    except Exception as e:
        logging.error("Exception in NLP training pipeline")
        raise CustomException(e, sys)