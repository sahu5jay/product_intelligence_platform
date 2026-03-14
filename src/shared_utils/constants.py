import torch
from pathlib import Path

# ===============================
# Base Directory (Project Root)
# ===============================
BASE_DIR = Path(__file__).resolve().parents[2]

# ===============================
# Artifacts Root
# ===============================
ARTIFACTS_ROOT = BASE_DIR / "artifacts"

STRUCTURED_ARTIFACTS = ARTIFACTS_ROOT / "structured_ml"
GAN_ARTIFACTS = ARTIFACTS_ROOT / "gan"
NLP_ARTIFACTS = ARTIFACTS_ROOT / "nlp"

# ===============================
# Device Configuration
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# STRUCTURED ML PATHS
# ==========================================================

STRUCTURED_DATA_DIR = STRUCTURED_ARTIFACTS / "data"
STRUCTURED_MODEL_DIR = STRUCTURED_ARTIFACTS / "model"
STRUCTURED_METRICS_DIR = STRUCTURED_ARTIFACTS / "metrics"
ARRAY_DIR = STRUCTURED_ARTIFACTS / "data_array"

# Raw data (notebook data folder)
RAW_DATA_PATH = BASE_DIR / "notebook/data/structured/train.csv"

# Processed data    
PROCESSED_STRUCTURED_DATA = STRUCTURED_DATA_DIR / "processed_data.csv"
TRIMED_DATA_PATH = STRUCTURED_DATA_DIR / "trimed_data.csv"
TRAIN_DATA_PATH = STRUCTURED_DATA_DIR / "train_data.csv"
TEST_DATA_PATH = STRUCTURED_DATA_DIR / "test_data.csv"
TRAIN_ARRAY_PATH = ARRAY_DIR / "train_data.csv"
TEST_ARRAY_PATH = ARRAY_DIR / "test_data.csv"


# Model artifacts
STRUCTURED_MODEL_PATH = STRUCTURED_MODEL_DIR / "model.pkl"
FINAL_MODEL_PATH = STRUCTURED_MODEL_DIR / "final_model.pkl"
PREPROCESSOR_PATH = STRUCTURED_MODEL_DIR / "preprocessor.pkl"
DATA_VALIDATION_REPORT_PATH = STRUCTURED_METRICS_DIR / "metrics.json"
TRIMED_VALIDATION_REPORT_PATH = STRUCTURED_METRICS_DIR / "trimed_validation.json"

# ==========================================================
# GAN DATA PATHS
# ==========================================================

GAN_DATA_DIR = GAN_ARTIFACTS / "data"
GAN_MODELS_DIR = GAN_ARTIFACTS / "models"
CHECKPOINTS_DIR = GAN_ARTIFACTS / "checkpoints"

# CSV dataset for images

GAN_RAW_DATA_PATH = BASE_DIR / "notebook/data/images/fashion_image.csv"

# Processed numpy data
RAW_DATA_PATH = GAN_DATA_DIR / "raw_data.csv"
PROCESSED_DATA_PATH = GAN_DATA_DIR / "processed_data.npy"

# Image folders
TRANSFORMED_IMAGES_DIR = GAN_ARTIFACTS / "transformed_images"
IMAGES_PNG_DIR = GAN_ARTIFACTS / "images_png"

# Generated images for frontend
GENERATED_IMAGES_DIR = BASE_DIR / "frontend/static/generated_images"

# GAN model files
GENERATOR_MODEL_PATH = GAN_MODELS_DIR / "generator.pth"
DISCRIMINATOR_MODEL_PATH = GAN_MODELS_DIR / "discriminator.pth"

# ==========================================================
# GAN Evaluation
# ==========================================================

GAN_EVAL_DIR = GAN_ARTIFACTS / "evaluation"
GAN_SAMPLE_GRID = GAN_EVAL_DIR / "sample_grid.png"

# Logs
TRAINING_LOGS_JSON = GAN_ARTIFACTS / "training_logs.json"
LOG_FILE = GAN_ARTIFACTS / "training.log"

# ==========================================================
# NLP PATHS
# ==========================================================



# NLP raw data
NLP_RAW_DATA_DIR = BASE_DIR / "notebook/data/text/IMDB.csv"

# NLP processed data
NLP_PROCESSED_DATA_DIR = NLP_ARTIFACTS / "data"
NLP_MODEL_DIR = NLP_ARTIFACTS / "model"
NLP_REPORT_DIR = NLP_ARTIFACTS / "report"

RAW_DATA_PATH = NLP_PROCESSED_DATA_DIR / "raw.csv"
CLEANED_DATA_PATH = NLP_PROCESSED_DATA_DIR / "clean_data.csv"
TRAIN_DATA_PATH = NLP_PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = NLP_PROCESSED_DATA_DIR / "test.csv"

NLP_SENTIMENT_DIR = NLP_MODEL_DIR / "model.pkl"
NLP_TOKENIZER_DIR = NLP_MODEL_DIR / "tokenizer.pkl"
NLP_METRICS_PATH = NLP_REPORT_DIR / "metrics.json"

# NLP_METRICS_JSON = NLP_ARTIFACTS / "report/metrics.json"

# ==========================================================
# GAN Hyperparameters
# ==========================================================

DEFAULT_IMAGE_SIZE = 64
DEFAULT_CHANNELS = 1

DEFAULT_LATENT_DIM = 100
DEFAULT_FEATURE_MAPS_GEN = 64
DEFAULT_FEATURE_MAPS_DISC = 64

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20

DEFAULT_LR = 0.0002
DEFAULT_BETA1 = 0.5
DEFAULT_BETA2 = 0.999

MAX_GENERATED_IMAGES = 20

# ==========================================================
# Create Required Directories Automatically
# ==========================================================

ALL_DIRS = [

    # Artifact root
    ARTIFACTS_ROOT,

    # Structured
    STRUCTURED_ARTIFACTS,
    STRUCTURED_DATA_DIR,
    STRUCTURED_MODEL_DIR,
    STRUCTURED_METRICS_DIR,

    # GAN
    GAN_ARTIFACTS,
    GAN_DATA_DIR,
    GAN_MODELS_DIR,
    CHECKPOINTS_DIR,
    TRANSFORMED_IMAGES_DIR,
    IMAGES_PNG_DIR,
    GAN_EVAL_DIR,

    # NLP
    NLP_ARTIFACTS,
    NLP_MODEL_DIR,
    NLP_TOKENIZER_DIR,

    # Frontend images
    GENERATED_IMAGES_DIR
]

# for directory in ALL_DIRS:
#     directory.mkdir(parents=True, exist_ok=True)