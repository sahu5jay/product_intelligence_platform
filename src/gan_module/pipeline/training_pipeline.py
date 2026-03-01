# src/structured_ml/pipeline/training_pipeline.py

from src.gan_module.components.image_ingestion import ImageIngestion
from src.gan_module.components.image_transformation import ImageTransformation

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
import sys




if __name__ == "__main__":
    # obj=ImageIngestion()
    # processed_data_path=obj.initiate_image_ingestion()
    # print(processed_data_path)

    transformation = ImageTransformation()
    transformation.initiate_image_transformation()