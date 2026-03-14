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

Final Multi-Modal Architecture Status
Module	          Dataset Type	          Ingestion Output
Structured ML	     Tabular CSV	          train.csv, test.csv
NLP	Text            CSV	train.csv,          test.csv
GAN	Image             CSV	              processed_images.npy

Data Validation Report for Strutured data:  
  artifacts/structured/validation_report.txt

NLP Module – Model Builder & Model Loader

Feature Engineering
TF-IDF Vectorizer
Maximum features: 5000
Output shape:
Train: (40000, 5000)
Test: (10000, 5000)

Model Performance
  Algorithm: Logistic Regression
  Accuracy: 88.79%


Model Loader

Load saved TF-IDF vectorizer
Load trained classification model
Prepare pipeline for prediction
Handle device configuration (if using transformers)





GAN Module
The GAN module is responsible for generating synthetic product images.

Generator

Creates synthetic images from random noise.

Discriminator

Evaluates whether an image is real or fake.

GAN Trainer
Handles:
Data loading
Training loop
Loss computation
Model optimization
Logging

Training Logs (Example)

INFO - Generator model initialized
INFO - Discriminator model initialized
INFO - Loading processed image data

Epoch [1/50]  D Loss: 1.8720 | G Loss: 0.5797
Epoch [2/50]  D Loss: 1.8010 | G Loss: 1.0615
Epoch [3/50]  D Loss: 0.7647 | G Loss: 2.4324
Epoch [10/50] D Loss: 0.7213 | G Loss: 1.4630
Epoch [20/50] D Loss: 1.0591 | G Loss: 1.3283
Epoch [30/50] D Loss: 0.6911 | G Loss: 1.4786
Epoch [40/50] D Loss: 0.9694 | G Loss: 1.7146
Epoch [50/50] D Loss: 0.5271 | G Loss: 1.6215


🧠 Product Intelligence Platform

An end-to-end AI platform that integrates Structured Machine Learning, Generative Adversarial Networks (GAN), and NLP to analyze product data, generate synthetic images, and perform sentiment analysis on customer reviews.

The system is built using modular ML pipelines, Flask APIs, and a web interface for easy interaction.

🚀 Key Features

📊 Structured ML – Predict product insights from tabular data

🎨 GAN Image Generation – Generate synthetic product images

🧾 NLP Sentiment Analysis – Analyze customer reviews

🌐 Flask API for model interaction

🖥 Web UI for predictions, image generation, and analysis

🐳 Docker support for deployment

📝 Centralized logging and error handling

📂 Project Structure
product_intelligence_platform/
│
├── artifacts/              # Trained models & outputs
│   ├── structured/
│   ├── gan/
│   └── nlp/
│
├── data/                   # Datasets
│   ├── structured/
│   ├── images/
│   └── text/
│
├── src/
│   ├── structured_ml/      # Structured ML pipelines
│   ├── gan_module/         # GAN training & inference
│   ├── nlp_module/         # NLP model training & prediction
│   ├── shared_utils/       # Logging, constants, utilities
│   └── orchestration/      # Training workflows
│
├── deployment/             # API & Docker setup
│   └── api/
│
├── frontend/               # Web interface
│   ├── templates/
│   └── static/
│
├── logs/
├── requirements.txt
├── setup.py
└── README.md
⚙️ Modules
📊 Structured ML

Handles tabular data prediction using traditional machine learning.

Pipeline steps:

Data ingestion

Data validation

Data transformation

Model training

Model evaluation

Prediction pipeline

Artifacts saved in:

artifacts/structured/
🎨 GAN Module

Generates synthetic product images using a Generator–Discriminator architecture.

Pipeline steps:

Image ingestion

Data loading

GAN training

Generator inference

Image generation

Artifacts saved in:

artifacts/gan/
🧾 NLP Module

Performs sentiment analysis on product reviews.

Pipeline steps:

Text ingestion

Text cleaning

Tokenization

Dataset preparation

Model training

Inference

Artifacts saved in:

artifacts/nlp/
📊 Model Evaluation

Each module includes evaluation metrics.

Structured ML

Accuracy

Precision

Recall

F1 Score

GAN

Generator loss

Discriminator loss

Visual inspection of generated samples

NLP

Accuracy

Precision

Recall

F1 Score

Evaluation results are stored in:

artifacts/*/metrics.json
🌐 API Endpoints

Flask routes expose model functionality.

Endpoint	Function
/predict	Structured ML predictions
/generate-image	Generate synthetic images
/analyze-review	NLP sentiment analysis
🖥 Web Interface

Frontend allows users to:

Predict structured data results

Generate synthetic images

Analyze product reviews

Pages include:

index.html
predict.html
generate.html
analyze.html
⚡ Installation

Clone repository:

git clone <repo_url>
cd product_intelligence_platform

Install dependencies:

pip install -r requirements.txt

Run the API server:

python deployment/api/main.py

Open browser:

http://localhost:5000
🛠 Tech Stack

Python

PyTorch

Scikit-learn

Flask

Docker

HTML / CSS / JS

👨‍💻 Author

Jay Sahu
MBA | Data Analytics & Machine Learning