import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def get_transform(image_size):
    """
    Returns image preprocessing pipeline
    """

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])


def load_image(image_path, image_size):
    """
    Load and preprocess a single image
    """

    transform = get_transform(image_size)

    image = Image.open(image_path).convert("L")

    return transform(image)


def load_images_from_folder(folder_path, image_size):
    """
    Load all images from a folder
    """

    images = []
    transform = get_transform(image_size)

    for file in os.listdir(folder_path):

        if file.lower().endswith((".png", ".jpg", ".jpeg")):

            img_path = os.path.join(folder_path, file)

            image = Image.open(img_path).convert("L")
            image = transform(image)

            images.append(image)

    if len(images) == 0:
        raise ValueError(f"No images found in {folder_path}")

    return torch.stack(images)


def save_generated_images(images, save_dir, label):
    """
    Save generated images individually
    """

    os.makedirs(save_dir, exist_ok=True)

    image_paths = []

    for i, img in enumerate(images):

        filename = f"{label}_{i}.png"
        path = os.path.join(save_dir, filename)

        save_image(img, path, normalize=True)

        image_paths.append(path)

    return image_paths


def denormalize_image(tensor):
    """
    Convert normalized tensor back to image format
    """

    tensor = tensor * 0.5 + 0.5

    return tensor.clamp(0, 1)