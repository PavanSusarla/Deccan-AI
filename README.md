House Price Prediction - Machine Learning Engineer Assessment Task

Objective

This project aims to build and deploy a machine learning model to predict house prices. It covers data preprocessing, model training, evaluation, hyperparameter tuning, and deployment as a REST API using FastAPI.

Part 1: Data Preprocessing

Dataset

You can use datasets like Kaggle's House Price Prediction Dataset or California Housing Dataset from Scikit-learn.

Steps

Load Dataset:

Read the data using Pandas.

Exploratory Data Analysis (EDA):

Check data statistics, distributions, and visualize feature correlations.

Handle Missing Values:

Impute numerical features with median or mean.

Fill categorical features with mode.

Feature Engineering:

Encode categorical variables using One-Hot Encoding.

Scale numerical features using MinMaxScaler or StandardScaler.

Create new features if needed (e.g., price per square foot).

Part 2: Model Training & Evaluation

Steps

Train-Test Split:

Split data into training (80%) and testing (20%) sets.

Model Selection:

Train multiple regression models: Linear Regression, Decision Tree, Random Forest, XGBoost.

Evaluation Metrics:

Calculate RMSE, MAE, and R² scores.

Hyperparameter Tuning:

Use GridSearchCV or RandomizedSearchCV to find the best model parameters.

Model Saving:

Save the trained model using Pickle (model.pkl).

Part 3: Model Deployment

Steps

Build API:

Create a FastAPI app (app.py) with an endpoint /predict that accepts JSON inputs and returns a price prediction.

Test API:

Test the API locally using Postman or CURL.

Docker Container (Optional):

Write a Dockerfile to containerize the API.

Part 4: Report & Documentation

Approach Overview

Data preprocessing: Detailed handling of missing values, encoding, scaling.

Model training: Compared multiple regression models and tuned hyperparameters.

Deployment: Created a FastAPI service with a /predict endpoint.

API Usage Guide

Run the API:

uvicorn app:app --reload

API Endpoint:

POST /predict
{
  "feature1": value,
  "feature2": value,
  ...
}

Response:

{ "predicted_price": 250000 }

Bonus Features

Implemented logging and error handling.

Dockerfile provided for containerization.

Can extend for deployment on AWS/GCP/Azure.

MLflow/DVC integration for model versioning (optional).

Frontend UI integration (optional).

Repository Structure

📁 House-Price-Prediction
│
├── 📄 app.py                 # FastAPI app
├── 📄 model.pkl              # Trained model
├── 📄 Dockerfile             # Docker container setup
├── 📄 requirements.txt       # Dependencies
└── 📄 README.md              # Project documentation

Setup Instructions

Clone the Repository:

git clone <repository_url>
cd House-Price-Prediction

Install Dependencies:

pip install -r requirements.txt

Run FastAPI:

uvicorn app:app --reload

Test the API:

Use Postman or CURL to test /predict.

(Optional) Run Docker Container:

docker build -t house-price-api .
docker run -p 8000:8000 house-price-api

Author

Pavan Susarla

**Happy Coding! **

