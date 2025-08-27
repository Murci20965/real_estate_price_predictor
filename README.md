# AI-Powered Real Estate Price Predictor

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-green?logo=xgboost)](https://xgboost.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-brightgreen?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-Deployed-yellow?logo=amazon-aws)](https://aws.amazon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Overview

This project is a complete, end-to-end MLOps system designed to predict housing prices in Ames, Iowa. It follows a modern software engineering and machine learning workflow, encompassing everything from data exploration and model training to API development, containerization with Docker, and automated deployment to the AWS cloud.

The core of the project is a **Gradient Boosting (XGBoost)** model, a high-performance algorithm renowned for its accuracy on tabular data.

---

## 📊 Model Performance

The model was evaluated on an unseen test set (20% of the original data). The performance metrics indicate a highly accurate and reliable model.

-   **R-squared (R²)**: **0.9037**
    -   This means the model can explain approximately **90.4%** of the variance in the house sale prices.
-   **Root Mean Squared Error (RMSE)**: **0.1341**
    -   This is the average prediction error on the log-transformed sale prices.

| Metric                | Score    | Description                         |
| :-------------------- | :------- | :---------------------------------- |
| **R-squared (R²)** | `0.9037` | Explains 90.4% of price variance.   |
| **RMSE (on log scale)** | `0.1341` | Low average prediction error.       |

---

## 🛠️ Tech Stack

| Component          | Technology/Library                                      |
| :----------------- | :------------------------------------------------------ |
| **Data Science** | `Pandas`, `NumPy`, `Scikit-learn`                         |
| **ML Model** | `XGBoost`                                               |
| **Backend API** | `FastAPI`                                               |
| **Web UI** | `Streamlit`                                             |
| **Containerization** | `Docker`                                                |
| **Testing** | `Pytest`                                                |
| **Automation** | `GitHub Actions`                                        |
| **Cloud Deployment** | `AWS` (Elastic Beanstalk, S3, ECR)                      |
| **Version Control** | `Git` & `GitHub`                                        |

---

## 🚀 CI/CD Automation

This project is configured with a complete CI/CD pipeline using GitHub Actions. On every push to the `main` branch, the following automated workflow is triggered:
1.  **Run Tests**: The `pytest` suite is executed to ensure code quality and prevent regressions.
2.  **Build & Push**: A new Docker image is built and pushed to a private Amazon ECR repository.
3.  **Deploy**: The new image is automatically deployed to the AWS Elastic Beanstalk environment, updating the live application.

---

## 🚀 Getting Started

To run this project locally, follow these steps:

### Prerequisites
- Python 3.11+
- Docker Desktop

### 1. Clone the Repository
```bash
git clone [https://github.com/Murci20965/real_estate_price_predictor.git](https://github.com/Murci20965/real_estate_price_predictor.git)
cd real_estate_price_predictor
```

### 2. Set Up the Environment
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies and the project in editable mode
pip install -r requirements.txt
pip install -e .
```

### 3. Train the Model
```bash
python -m src.train
```

### 4. Run the Application Locally
**Terminal 1: Start the FastAPI Backend**
```bash
uvicorn app.main:app --reload
```
**Terminal 2: Start the Streamlit UI**
```bash
streamlit run ui/interface.py
```

---

## ✅ Testing

To run the automated unit tests, first ensure you have installed the testing framework:
```bash
pip install pytest
```
Then, run the following command from the project root:
```bash
pytest
```

---

## 🎯 Future Goals
- **Monitoring**: Integrate Prometheus and Grafana for live monitoring of the deployed application's performance and health.
- **Advanced Feature Engineering**: Experiment with more complex features to further improve model accuracy.
- **Hyperparameter Tuning**: Implement an automated hyperparameter tuning pipeline to find the optimal model settings.
```
