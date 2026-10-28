
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def sample_input():
    return {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 33,
    }


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "Ready to serve"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    response = client.post("/predict", json=sample_input())

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["class_name"] in [
        "Diabetes positive",
        "Diabetes negative",
    ]
    assert 0 <= result["probability_of_positive_class"] <= 1


def test_prediction_with_zero_measurements():
    data = sample_input()
    data["Insulin"] = 0
    data["SkinThickness"] = 0

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]


def test_prediction_rejects_invalid_age():
    data = sample_input()
    data["Age"] = 0

    response = client.post("/predict", json=data)

    assert response.status_code == 422


def test_prediction_rejects_missing_fields():
    data = {"Glucose": 120}

    response = client.post("/predict", json=data)

    assert response.status_code == 422
