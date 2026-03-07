# src/gan_module/pipeline/inference_pipeline.py

import sys
import os

from src.gan_module.components.evaluation import GANEvaluator
from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException


class InferencePipeline:

    def __init__(self):

        self.model_path = "artifacts/gan/models/generator.pth"
        self.num_images = 10

    def run_pipeline(self):

        try:

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Trained model not found at {self.model_path}"
                )

            logging.info("Starting GAN inference pipeline")

            evaluator = GANEvaluator(self.model_path)

            evaluator.generate_images(self.num_images)

            logging.info("Image generation completed successfully")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    pipeline = InferencePipeline()
    pipeline.run_pipeline()