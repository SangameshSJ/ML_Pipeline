import pickle
import numpy as np
from fastapi import FastAPI

# Load model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ML Model API is running!"}

@app.post("/predict/")
def predict(features: list):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}