# Transaction Fraud Detection with MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.9-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-v0.95-brightgreen) ![MLflow](https://img.shields.io/badge/MLflow-v2.8-orange) ![Docker](https://img.shields.io/badge/Docker-20.10-blue) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-purple)

## 🚀 Overview
This project implements an **end-to-end machine learning pipeline** for detecting fraudulent credit card transactions, demonstrating practical MLOps practices including model tracking, versioning, REST API serving, Docker deployment, and CI/CD automation.

The pipeline covers all stages: data generation and preprocessing, model training and evaluation, model serving via API, and deployment with Docker/Docker Compose, along with GitHub Actions CI/CD simulation.

---

## 🎯 Project Objectives
- Detect fraudulent transactions in a simulated or real dataset.
- Build a reproducible ML pipeline with proper MLOps workflows.
- Deploy the trained model using a REST API.
- Dockerize the application and simulate a mini production environment.
- Automate testing, building, and deployment using GitHub Actions.

---

## 📂 Project Structure
```bash
MLOpsTraining/
│
├── api/                      # FastAPI application for serving predictions
│   └── app.py
│
├── data/                     # Data generation scripts
│   └── load_data.py
│
├── utils/                    # Helper functions & logging
│   ├── logger.py
│   └── preprocessing.py
│
├── training/                 # Model training scripts
│   └── train_model.py
│
├── evaluation/               # Evaluation scripts & visualizations
│   └── evaluate.py
│
├── checks/                   # Model validation & helper scripts
│   ├── model_checks.py
│   ├── Input_runs.py
│   └── model_downloader.py
│
├── models/                   # MLflow-logged models & artifacts
│   └── artifacts/
│       ├── MLmodel
│       ├── python_model.pkl
│       └── artifacts/EnsembleLearning.pkl
│
├── Automation.github/        # GitHub Actions workflows
│   └── workflows/docker-ci.yml
│
├── docker-compose.yml        # Docker Compose configuration
├── myapp.dockerfile          # Dockerfile for API container
├── main.py                   # Orchestrator script: generate, preprocess, train, evaluate
├── requirements.txt          # Python dependencies
└── README.md
```



| Area          | Tools & Libraries           |
| ------------- | --------------------------- |
| ML            | scikit-learn, pandas, numpy |
| MLOps         | MLflow, Git, GitHub Actions |
| Serving       | FastAPI                     |
| Packaging     | Docker, Docker Compose      |
| CI/CD         | GitHub Actions              |
| UI (optional) | Streamlit / Gradio          |


## ⚡ Setup & Installation
### Clone the repository

git clone git@github.com:SalahXIX/Transaction-Fraud-Detector.git
cd Transaction-Fraud-Detector

### Install dependencies

pip install -r requirements.txt

### Build Docker image

docker build -f myapp.dockerfile -t fraud-api .

### Run the API locally

docker run -p 8000:8000 fraud-api

Access the API at http://127.0.0.1:8000/predict

### Using Docker Compose

docker-compose up --build

This starts the API service in a simulated production environment.

## 🖥 REST API Usage
Endpoint: POST /predict
Input: JSON with transaction features

Example:
{
  "Hour": 14,
  "Day": 3,
  "Boundary": 2,
  "Suspicious_car_rental": 1,
  "Suspicious_fuel": 0,
  "Cumulative_type_percent": 0.2,
  "Cumulative_Unique_Locations": 1,
  "Days_since_last": 5
}

{
  "prediction": 1
}

Output: Fraud prediction (0 = non-fraud, 1 = fraud)

### Test the API
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @sample_input.json

## 🧩 ML Pipeline

### Data Generation
  - Synthetic transaction data using Faker with realistic locations, amounts, and transaction types.

  - Boundary and feature engineering applied for anomaly detection.

### Preprocessing
 - Compute derived features: Cumulative_type_percent, Cumulative_Unique_Locations, Days_since_last, Suspicious_car_rental, Suspicious_fuel.

 - Drop irrelevant columns for model input.

### Model Training
 - Ensemble of Isolation Forest + Local Outlier Factor.

 - Contamination thresholds computed and logged via MLflow.

 - Model artifacts saved and registered with MLflow.

### Evaluation
 - SHAP-based explanations (summary and waterfall plots)

 - Correlation heatmaps and sample transaction visualizations

 - Metrics & plots saved to evaluation_outputs/.

## 🔧 MLOps Features

 - MLflow Integration: Track experiments, log metrics, save models, register ensemble model.

 - Dockerization: Isolated container with all dependencies.

 - Docker Compose: Simulate multi-service production environment.

 - CI/CD: GitHub Actions workflow for automated testing and Docker build.

## 📦 Deliverables

 - Structured GitHub repository with full codebase and README

 - MLflow-logged models and experiments

 - Dockerized API and Docker Compose setup

 - CI/CD pipeline via GitHub Actions

 - Evaluation visualizations and metrics


## 🎓 Learning Outcomes
 - Practical understanding of end-to-end ML pipelines

 - Experience with MLOps tools: MLflow, Docker, GitHub Actions

 - Deployment and serving of ML models via REST APIs

 - Feature engineering and model interpretability using SHAP

