from src.gan_module.components.image_ingestion import ImageIngestion
from src.gan_module.components.image_transformation import ImageTransformation

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
import sys


if __name__ == "__main__":

    try:

        # Step 1: Ingestion
        ingestion = ImageIngestion()
        processed_data_path = ingestion.initiate_image_ingestion()

        logging.info(f"Processed data saved at {processed_data_path}")

        # Step 2: Transformation
        transformation = ImageTransformation()
        transformation.initiate_image_transformation(processed_data_path)

        logging.info("Training pipeline completed successfully")

    except Exception as e:
        raise CustomException(e, sys)