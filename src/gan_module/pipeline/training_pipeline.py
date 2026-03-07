import os
from src.gan_module.components.image_ingestion import ImageIngestion
from src.gan_module.components.gan_trainer import GANTrainer
from src.gan_module.components.evaluation import GANEvaluator

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
import sys


if __name__ == "__main__":

    try:

        processed_data_path = "artifacts/gan/processed_images.npy"

        # Step 1: Ingestion (only if not already processed)
        if not os.path.exists(processed_data_path):

            logging.info("Processed data not found. Running ingestion...")

            ingestion = ImageIngestion()
            processed_data_path = ingestion.initiate_image_ingestion()

        else:

            logging.info("Processed data already exists. Skipping ingestion.")

        # Step 2: Training
        trainer = GANTrainer()
        trainer.train()

        logging.info("GAN Model Training Completed")

        # Step 3: Evaluation
        evaluator = GANEvaluator("artifacts/gan/models/generator.pth")
        evaluator.generate_images(10)

        logging.info("GAN Evaluation Completed")

    except Exception as e:
        raise CustomException(e, sys)