# AI-Powered Real Estate Price Predictor

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Ready-brightgreen?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
## 📖 Overview

This project is a complete, end-to-end MLOps system designed to predict housing prices in Ames, Iowa. It follows a modern software engineering and machine learning workflow, encompassing everything from data exploration and model training to API development, containerization with Docker, and deployment to the AWS cloud. The final product is a production-grade application with a user-friendly interface.

The core of the project is a **Gradient Boosting (XGBoost)** model, a high-performance algorithm renowned for its accuracy on tabular data.

---

## 📊 Model Performance

The model was evaluated on an unseen test set (20% of the original data). The performance metrics indicate a highly accurate and reliable model.

-   **R-squared (R²)**: **0.9037**
    -   This means the model can explain approximately **90.4%** of the variance in the house sale prices, indicating a very strong fit to the data.
-   **Root Mean Squared Error (RMSE)**: **0.1341**
    -   This is the average prediction error on the log-transformed sale prices. A lower value is better, and this score is highly competitive for the Ames dataset.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **R-squared (R²)** | `0.9037` | Explains 90.4% of price variance. |
| **RMSE (on log scale)** | `0.1341` | Low average prediction error. |

---

## 🛠️ Tech Stack

This project utilizes a modern, robust tech stack to ensure scalability, maintainability, and reproducibility.

| Component | Technology/Library |
| :--- | :--- |
| **Data Science** | `Pandas`, `NumPy`, `Scikit-learn` |
| **ML Model** | `XGBoost` |
| **Backend API** | `FastAPI` |
| **Web UI** | `Streamlit` |
| **Containerization** | `Docker` |
| **Testing** | `Pytest` (Future Goal) |
| **Logging** | Python `logging` module |
| **Monitoring** | `Prometheus`, `Grafana` (Future Goal) |
| **CI/CD** | `GitHub Actions` (Future Goal) |
| **Cloud Deployment** | `AWS` (Elastic Beanstalk, S3, ECR) |
| **Version Control** | `Git` & `GitHub` |

---

## 🏗️ System Architecture

The application is designed with a decoupled architecture, where the frontend UI, backend API, and machine learning model are separate, containerized components. This allows for independent scaling and maintenance.

```
+----------------------+     +------------------------+     +---------------------+
|                      |     |                        |     |                     |
|    Streamlit UI      |----->|    FastAPI Backend     |----->|    XGBoost Model    |
|  (User Interface)    |     | (REST API w/ Metrics)  |     |  (Prediction Logic) |
|                      |     |                        |     |                     |
+----------------------+     +------------------------+     +---------------------+
        |
        | (Deployed as a Docker Container on AWS Elastic Beanstalk)
        v
+-----------------------------------------------------------------------------------+
|                               AWS Cloud (Free Tier)                               |
|  +-----------------+     +-----------------------+     +-----------------+        |
|  |   Amazon S3     |----->| AWS Elastic Beanstalk |<-----|  User / Client  |        |
|  | (Model Storage) |     | (Hosts Docker Container)|     |  (Web Browser)    |        |
|  +-----------------+     +-----------------------+     +-----------------+        |
+-----------------------------------------------------------------------------------+
```

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
Create and activate a virtual environment, then install dependencies.
```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Train the Model
Run the training script to preprocess data and train the model.
```bash
python -m src.train
```

### 4. Run the Application Locally
This project uses a decoupled frontend and backend. You will need two separate terminals.

**Terminal 1: Start the FastAPI Backend**
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

**Terminal 2: Start the Streamlit UI**
```bash
streamlit run ui/interface.py
```
The user interface will be available at `http://localhost:8501`.

### 5. Run with Docker
Build and run the containerized application.
```bash
# Build the Docker image
docker build -t real-estate-predictor .

# Run the container
docker run -d -p 8000:80 --name predictor-app real-estate-predictor
```
The application API will be available at `http://localhost:8000`.

---

## 📈 Project Workflow

The project follows a structured MLOps workflow:
1.  **Data Exploration & Feature Engineering**: Analyze the data in a Jupyter Notebook to understand its characteristics and engineer new features.
2.  **Model Training & Evaluation**: Develop a robust pipeline to train the XGBoost model and evaluate its performance.
3.  **API Development**: Create a FastAPI backend to serve model predictions via a REST API.
4.  **UI Development**: Build an interactive web interface with Streamlit.
5.  **Containerization**: Package the entire application into a Docker image for portability.
6.  **Deployment**: Deploy the container to AWS Elastic Beanstalk.

---

## 🎯 Future Goals
- **CI/CD Automation**: Implement a full CI/CD pipeline with GitHub Actions to automate testing and deployment.
- **Unit Testing**: Add comprehensive unit tests with `Pytest` to ensure code quality and reliability.
- **Monitoring**: Integrate Prometheus and Grafana for live monitoring of the deployed application's performance and health.
```
</markdo
