
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Diabetes Prediction API")


class DiabetesInput(BaseModel):
    Pregnancies: int = Field(ge=0, le=30)
    Glucose: float = Field(ge=0, le=300)
    BloodPressure: float = Field(ge=0, le=200)
    SkinThickness: float = Field(ge=0, le=100)
    Insulin: float = Field(ge=0, le=1000)
    BMI: float = Field(ge=0, le=100)
    DiabetesPedigreeFunction: float = Field(ge=0, le=5)
    Age: int = Field(ge=1, le=120)


@app.get("/")
def ready_to_serve():
    return {"status": "Ready to serve"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Endpoint for making predictions with the Diabetes model
@app.post("/predict")
def predict(data: DiabetesInput):
    features = pd.DataFrame([{
        "Pregnancies": data.Pregnancies,
        "Glucose": data.Glucose,
        "BloodPressure": data.BloodPressure,
        "SkinThickness": data.SkinThickness,
        "Insulin": data.Insulin,
        "BMI": data.BMI,
        "DiabetesPedigreeFunction": data.DiabetesPedigreeFunction,
        "Age": data.Age,
    }])

    # Apply the same zero-value handling used during training
    zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    features[zero_as_missing] = features[zero_as_missing].replace(0, np.nan)

    prediction = int(model.predict(features)[0])
    print("Prediction is", prediction)

    probabilities = model.predict_proba(features)[0]
    positive_index = list(model.classes_).index(1)

    return {
        "prediction": prediction,
        "class_name": (
            "Diabetes positive" if prediction == 1
            else "Diabetes negative"
        ),
        "probability_of_positive_class": round(
            float(probabilities[positive_index]), 4
        ),
    }
