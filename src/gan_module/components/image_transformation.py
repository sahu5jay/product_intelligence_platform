import sys
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

from src.shared_utils.logger import logging
from src.shared_utils.exception import CustomException
from src.shared_utils.utils import save_numpy
from src.shared_utils.constants import PROCESSED_DATA_PATH, TRANSFORMED_IMAGES_DIR
from src.shared_utils.constants import IMAGES_PNG_DIR  # <- new PNG dir constant
from src.shared_utils.utils import save_object


class ImageTransformation:

    def __init__(self, config):

        self.image_size = config["image_transformation"]["image_size"]
        self.channels = config["image_transformation"]["channels"]
        self.batch_size = config["image_transformation"]["batch_size"]
        self.shuffle = config["image_transformation"]["shuffle"]
        self.num_workers = config["image_transformation"]["num_workers"]
        self.pin_memory = config["image_transformation"]["pin_memory"]

        self.processed_data_path = PROCESSED_DATA_PATH

        # Ensure PNG directory exists
        IMAGES_PNG_DIR.mkdir(parents=True, exist_ok=True)

    def initiate_image_transformation(self):

        try:

            logging.info("Starting Image Transformation")

            if not self.processed_data_path.exists():
                raise FileNotFoundError(
                    f"Processed dataset not found: {self.processed_data_path}"
                )

            # -------------------------
            # Load numpy dataset
            # -------------------------
            logging.info("Loading processed numpy dataset")
            images = np.load(self.processed_data_path)
            logging.info(f"Dataset loaded with shape: {images.shape}")

            # -------------------------
            # Reshape pixels → images
            # -------------------------
            images = images.reshape(-1, 1, 28, 28)
            logging.info("Images reshaped to 28x28 format")

            # -------------------------
            # Convert to tensor
            # -------------------------
            images = torch.tensor(images, dtype=torch.float32)

            # -------------------------
            # Image Transformations
            # -------------------------
            transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.Normalize((0.5,), (0.5,))
                ]
            )

            transformed_images = [transform(img) for img in images]
            transformed_images = torch.stack(transformed_images)

            logging.info(f"Images resized to {self.image_size}x{self.image_size}")

            # -------------------------
            # Save transformed images as .npy
            # -------------------------
            TRANSFORMED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            transformed_file_path = TRANSFORMED_IMAGES_DIR / "transformed_images.npy"
            save_numpy(transformed_file_path, transformed_images.numpy())
            logging.info(f"Transformed images saved at {transformed_file_path}")

            # -------------------------
            # Save transformed images as PNGs
            # -------------------------
            for idx, img_tensor in enumerate(transformed_images):
                # Ensure tensor shape is (H, W)
                img = img_tensor.squeeze().detach().cpu()  # remove channel if 1, ensure CPU
                # Denormalize from [-1, 1] → [0, 255]
                img = (img - img.min()) / (img.max() - img.min())  # scale to [0,1]
                img = (img * 255).numpy().astype(np.uint8)
                # Convert to PIL Image
                pil_img = Image.fromarray(img, mode='L')  # 'L' for grayscale
                pil_img.save(IMAGES_PNG_DIR / f"transformed_{idx+1}.png")

            # -------------------------
            # Create Dataset & DataLoader
            # -------------------------
            dataset = TensorDataset(transformed_images)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )

            logging.info("DataLoader created successfully")
            return dataloader

        except Exception as e:
            logging.error("Error during image transformation")
            raise CustomException(e, sys)