import torch
from pathlib import Path

# ===============================
# Base Directory
# ===============================
BASE_DIR = Path(__file__).resolve().parents[2]  # points to project root

# ===============================
# Artifacts Paths
# ===============================
ARTIFACTS_ROOT = BASE_DIR / "artifacts"

# GAN paths
GAN_ARTIFACTS = ARTIFACTS_ROOT / "gan"
PROCESSED_IMAGES_PATH = GAN_ARTIFACTS / "processed_images.npy"
GENERATOR_MODEL_PATH = GAN_ARTIFACTS / "generator.pt"
DISCRIMINATOR_MODEL_PATH = GAN_ARTIFACTS / "discriminator.pt"
CHECKPOINTS_DIR = GAN_ARTIFACTS / "checkpoints"
GAN_SAMPLES_DIR = GAN_ARTIFACTS / "samples"
TRAINING_LOGS_JSON = GAN_ARTIFACTS / "training_logs.json"

# Structured ML paths
STRUCTURED_ARTIFACTS = ARTIFACTS_ROOT / "structured_ml"
STRUCTURED_MODEL_PATH = STRUCTURED_ARTIFACTS / "model" / "model.pkl"
PREPROCESSOR_PATH = STRUCTURED_ARTIFACTS / "model" / "preprocessor.pkl"
STRUCTURED_METRICS_JSON = STRUCTURED_ARTIFACTS / "metrics.json"

# NLP paths
NLP_ARTIFACTS = ARTIFACTS_ROOT / "nlp"
FINE_TUNED_MODEL_DIR = NLP_ARTIFACTS / "fine_tuned_model"
TOKENIZER_DIR = NLP_ARTIFACTS / "tokenizer"
NLP_METRICS_JSON = NLP_ARTIFACTS / "metrics.json"

# New paths for raw and processed text data
NLP_RAW_DATA_PATH = NLP_ARTIFACTS / "data" / "raw.csv"
NLP_PROCESSED_DATA_PATH = NLP_ARTIFACTS / "data" / "processed.csv"

# Frontend generated images
GENERATED_IMAGES_DIR = BASE_DIR / "frontend" / "static" / "generated_images"

# ===============================
# Default GAN Hyperparameters
# ===============================
DEFAULT_IMAGE_SIZE = 64       # height/width of input images
DEFAULT_CHANNELS = 1          # 1 for grayscale, 3 for RGB
DEFAULT_LATENT_DIM = 100      # input noise dimension
DEFAULT_FEATURE_MAPS_GEN = 64
DEFAULT_FEATURE_MAPS_DISC = 64
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_LR = 0.0002
DEFAULT_BETA1 = 0.5

# ===============================
# Logging Constants
# ===============================
LOG_FORMAT = "%(asctime)s [%(levelname)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = GAN_ARTIFACTS / "training.log"

# ===============================
# Device
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"