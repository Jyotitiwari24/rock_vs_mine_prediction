from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load trained model
model = joblib.load("src/sonar_model.pkl")

app = FastAPI(title="Sonar Object Classifier API")

# Input schema (60 features)
class SonarInput(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "Sonar Rock vs Mine Classifier API is running 🚀"}


@app.post("/predict")
def predict(data: SonarInput):
    if len(data.features) != 60:
        return {"error": "Input must contain exactly 60 values"}

    input_array = np.asarray(data.features).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    result = "Rock" if prediction == "R" else "Mine"

    return {
        "prediction": result
    }
