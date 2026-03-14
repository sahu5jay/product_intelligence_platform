import sys
from pathlib import Path

from src.shared_utils.exception import CustomException
from src.shared_utils.logger import logging
from src.shared_utils.config_loader import load_config

from src.gan_module.components.image_ingestion import ImageIngestion
from src.gan_module.components.image_transformation import ImageTransformation
from src.gan_module.components.gan_trainer import GANTrainer
from src.gan_module.components.evaluation import GANEvaluation
from src.shared_utils.constants import GAN_RAW_DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH

# -------------------------

# Load Config

# -------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = BASE_DIR / "gan_module" / "config.yaml"

config = load_config(CONFIG_PATH)

# Dataset path from config

# RAW_DATA_PATH = config["image_ingestion"]["raw_data_path"]

if __name__ == "__main__":

    try:

        logging.info("Starting GAN Training Pipeline")

        # -------------------------
        # Step 1: Image Ingestion
        # -------------------------

        logging.info("Step 1: Image Ingestion Started")

        image_ingestion = ImageIngestion(
            gan_data_path=GAN_RAW_DATA_PATH,
            raw_data_path = RAW_DATA_PATH,
            processed_data_path = PROCESSED_DATA_PATH
        )

        processed_data_path = image_ingestion.initiate_image_ingestion()

        logging.info(f"Processed dataset saved at: {processed_data_path}")
        logging.info("Image Ingestion Completed Successfully")


        # -------------------------
        # Step 2: Image Transformation
        # -------------------------

        logging.info("Step 2: Image Transformation Started")

        image_transformation = ImageTransformation(config=config)

        dataloader = image_transformation.initiate_image_transformation()

        logging.info("Image Transformation Completed Successfully")


        # -------------------------
        # Step 3: GAN Training
        # -------------------------

        logging.info("Step 3: GAN Training Started")

        # trainer = GANTrainer(config=config)

        # trainer.train(dataloader=dataloader)

        logging.info("GAN Training Completed Successfully")


        # -------------------------
        # Step 4: GAN Evaluation
        # -------------------------

        logging.info("Step 4: GAN Evaluation Started")

        evaluator = GANEvaluation()

        evaluator.generate_images()

        logging.info("GAN Evaluation Completed Successfully")


    except Exception as e:

        logging.error("Exception occurred in GAN training pipeline")

        raise CustomException(e, sys)

