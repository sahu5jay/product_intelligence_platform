##  Project Progress

###  Project Architecture Designed
- Created modular multi-modal architecture:
  - structured_ml/
  - gan_module/
  - nlp_module/
  - shared_utils/
  - orchestration/
  - deployment/
- Defined artifacts/ and data/ storage structure.
- Designed production-ready folder hierarchy.


Project Structure and File Summary
1. artifacts/
Contains all the outputs, trained models, and evaluation results of different modules.
structured/ – Artifacts for structured ML models:
model.pkl – trained predictive model.
preprocessor.pkl – preprocessing pipeline for structured data.
metrics.json – performance metrics of structured models.

gan/ – Artifacts for GAN-based image generation:
generator.pt – trained generator model.
discriminator.pt – trained discriminator model.
samples/ – generated image samples.
training_logs.json – GAN training history and logs.
nlp/ – Artifacts for NLP module:
fine_tuned_model/ – fine-tuned NLP model directory.
tokenizer/ – tokenizer for text preprocessing.
metrics.json – NLP model evaluation metrics.

2. data/
Holds raw and processed data for training and inference.
structured/ – Tabular or structured datasets:
raw/ – original datasets.
processed/ – cleaned and feature-engineered data.
images/ – Image datasets for GAN module:
raw/ – original images.
processed/ – resized/normalized images ready for training.
text/ – Text datasets for NLP module:
raw/ – unprocessed text files.
processed/ – tokenized, cleaned, and vectorized text.


3. src/
Core Python code for all modules and utilities.
structured_ml/ – Structured data ML module:
components/ – individual scripts for each ML step (ingestion, validation, transformation, training, evaluation, saving models).
pipeline/ – orchestrates full structured data ML training and prediction pipelines.
config.yaml – configuration for structured ML pipeline (paths, hyperparameters, etc.).

gan_module/ – GAN-based image generation:
components/ – scripts for GAN training, generator, discriminator, preprocessing, evaluation, and checkpointing.
pipeline/ – orchestrates training and inference for GANs.
config.yaml – GAN module configuration.

nlp_module/ – NLP module for text processing:
components/ – scripts for ingestion, cleaning, tokenization, dataset building, model loading, training, evaluation, and inference.
pipeline/ – full training and prediction orchestration.
config.yaml – NLP module configuration.

shared_utils/ – Utility scripts shared across modules:
logger.py – centralized logging.
exception.py – custom exception handling.
config_loader.py – load configuration files.
constants.py – common constants used in multiple modules.
utils.py – helper functions.

orchestration/ – High-level orchestration scripts:
main_training_flow.py – main script to trigger all training pipelines.
batch_scheduler.py – schedule batch runs or retraining.
retraining_pipeline.py – automated retraining workflow.

deployment/
Contains deployment scripts for exposing APIs and running services.
api/ – FastAPI application scripts:
main.py – API entry point.
structured_routes.py – endpoints for structured ML predictions.
gan_routes.py – endpoints for GAN image generation.
nlp_routes.py – endpoints for NLP predictions.
Dockerfile – Docker image setup for deployment.
docker-compose.yml – orchestration for local multi-service deployment.
requirements.txt – Python dependencies for deployment.

5. frontend/
Web frontend for interacting with the platform.
templates/ – HTML pages:
index.html – home page.
predict.html – structured ML prediction UI.
generate.html – GAN image generation UI.
analyze.html – NLP analysis UI.

static/ – Static assets:
css/ – styling files.

js/ – frontend scripts.

assets/ – additional images, icons, or media files for the frontend.

6. setup.py
Setup script for packaging and installing the platform as a Python package. Handles dependencies via requirements.txt.

7. .env
Environment variables for local development or deployment (e.g., API keys, database URLs).

8. .gitignore
Specifies files and folders to ignore in Git (artifacts, .env, logs, etc.).

9. README.md
Project documentation (this file), including project overview, structure, setup instructions, and usage examples.

Structured Data Ingestion Process'
src/structured_ml/components/data_ingestion.py

Objective
The purpose of the Data Ingestion process is to:
Load structured dataset (CSV format)
Store a copy of the raw dataset in the artifacts directory
Split the dataset into training and testing sets
Save train and test datasets for further pipeline steps
Ensure reproducibility of experiments

Workflow
The ingestion process follows these steps:
1️Dataset Loading
Reads structured data using Pandas
Validates dataset availability
Logs dataset shape and successful loading

NLP Data Ingestion Summary
Objective

The NLP Data Ingestion module prepares raw text data for model training by:
Loading the IMDB dataset
Cleaning and normalizing text
Splitting into train and test sets
Saving processed data into artifacts

Dataset structure:
review	sentiment
"Movie was great"	positive
"Worst movie ever"	negative

Architecture Alignment
✔ Uses shared_utils.logger
✔ Uses shared_utils.exception.CustomException
✔ Dynamic BASE_DIR path
✔ Compatible with NLP training pipeline
✔ UI-ready (artifact driven)

GAN Data Ingestion Summary
Objective
The GAN Data Ingestion module prepares image data (stored in CSV format) for GAN training by:
Loading pixel-based image dataset
Normalizing pixel values
Reshaping images
Saving processed numpy array

Architecture Alignment
✔ Uses shared logger
✔ Uses CustomException
✔ Dynamic root path handling
✔ Compatible with GAN training module
✔ Artifact-based storage
✔ UI ready for model output visualization
