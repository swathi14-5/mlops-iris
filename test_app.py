from fastapi.testclient import TestClient
from main import app
from datetime import datetime

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong","timestamp":datetime.datetime.now()}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 5,
        "petal_length": 3.2,
        "petal_width": 4.4,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Virginica","timestamp":datetime.datetime.now()}
#Task2 Writing test cases
def test_hi():
    with TestClient(app) as client:
        response = client.get("/hi")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"hi": "Hi welcome to my git ML project","timestamp":datetime.datetime.now()}
def test_hello():
    with TestClient(app) as client:
        response = client.get("/hello")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"hello": "Executing my code successfully","timestamp":datetime.datetime.now()}
