from fastapi import FastAPI
import joblib
import numpy as np

# Load trained model
model = joblib.load("models/house_price_model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

@app.post("/predict")
def predict(features: dict):
    input_data = np.array(list(features.values())).reshape(1, -1)
    prediction = model.predict(input_data)
    return {"predicted_price": float(prediction[0])}