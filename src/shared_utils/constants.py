import torch
from pathlib import Path

# ===============================
# Base Directory
# ===============================
BASE_DIR = Path(__file__).resolve().parents[2]

# ===============================
# Artifacts Root
# ===============================
ARTIFACTS_ROOT = BASE_DIR / "artifacts"
GAN_ARTIFACTS = ARTIFACTS_ROOT / "gan"

# ===============================
# Device
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# GAN Data Paths
# ===============================
GAN_DATA_DIR = GAN_ARTIFACTS / "data"
DATA_PATH = BASE_DIR / "notebook/data/images/fashion_image.csv"  # matches YAML
PROCESSED_DATA_PATH = GAN_DATA_DIR / "processed_data.npy"

# ===============================
# Image Transformation / PNGs
# ===============================
TRANSFORMED_IMAGES_DIR = GAN_ARTIFACTS / "transformed_images"
IMAGES_PNG_DIR = GAN_ARTIFACTS / "images_png"
IMAGES_PNG_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# GAN Generated Images (frontend)
# ===============================
GENERATED_IMAGES_DIR = BASE_DIR / "frontend/static/generated_images"
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# Models & Checkpoints
# ===============================
GAN_MODELS_DIR = GAN_ARTIFACTS / "models"
GENERATOR_MODEL_PATH = GAN_MODELS_DIR / "generator.pth"
DISCRIMINATOR_MODEL_PATH = GAN_MODELS_DIR / "discriminator.pth"
CHECKPOINTS_DIR = GAN_ARTIFACTS / "checkpoints"

# ===============================
# Evaluation
# ===============================
GAN_EVAL_DIR = GAN_ARTIFACTS / "evaluation"
GAN_SAMPLE_GRID = GAN_EVAL_DIR / "sample_grid.png"

# ===============================
# Logs
# ===============================
TRAINING_LOGS_JSON = GAN_ARTIFACTS / "training_logs.json"
LOG_FILE = GAN_ARTIFACTS / "training.log"

# ===============================
# GAN Hyperparameters
# ===============================
DEFAULT_IMAGE_SIZE = 64
DEFAULT_CHANNELS = 1
DEFAULT_LATENT_DIM = 100
DEFAULT_FEATURE_MAPS_GEN = 64
DEFAULT_FEATURE_MAPS_DISC = 64
DEFAULT_BATCH_SIZE = 20
DEFAULT_EPOCHS = 5
DEFAULT_LR = 0.0002
DEFAULT_BETA1 = 0.5
DEFAULT_BETA2 = 0.999
MAX_GENERATED_IMAGES = 20

# ===============================
# Create all necessary directories
# ===============================
for path in [
    ARTIFACTS_ROOT,
    GAN_ARTIFACTS,
    GAN_DATA_DIR,
    TRANSFORMED_IMAGES_DIR,
    GENERATED_IMAGES_DIR,
    GAN_MODELS_DIR,
    CHECKPOINTS_DIR,
    GAN_EVAL_DIR,
    IMAGES_PNG_DIR
]:
    path.mkdir(parents=True, exist_ok=True)