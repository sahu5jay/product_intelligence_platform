import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def get_transform(image_size):
    """
    Returns image preprocessing pipeline
    """

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return transform


def load_image(image_path, image_size):
    """
    Load and preprocess a single image
    """

    transform = get_transform(image_size)

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    return image


def load_images_from_folder(folder_path, image_size):
    """
    Load all images from a folder
    """

    images = []
    transform = get_transform(image_size)

    for file in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file)

        if file.lower().endswith((".png", ".jpg", ".jpeg")):

            image = Image.open(img_path).convert("RGB")
            image = transform(image)

            images.append(image)

    return torch.stack(images)


def save_generated_images(images, save_path):
    """
    Save generated images from GAN
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    save_image(
        images,
        save_path,
        normalize=True
    )


def denormalize_image(tensor):
    """
    Convert normalized tensor back to image format
    """

    tensor = tensor * 0.5 + 0.5
    return tensor.clamp(0, 1)