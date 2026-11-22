from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

PATIENT = {"age": 58, "bmi": 31.2, "HbA1c_level": 7.1, "glucose": 160, "cholesterol": 240, "sleep_hours": 5.5,
           "triglycerides": 220, "physical_activity": 0, "family_history": 1, "stress_level": 2,
           "high_blood_pressure": 1, "sugar_consumption": 2, "crp_level": 6.0, "gender_Male": 1,
           "smoking_Former": 0, "smoking_Never": 0, "education_level_Secondary": 1, "education_level_Tertiary": 0,
           "employment_status_Retired": 0, "employment_status_Unemployed": 0}


def test_health_and_index():
    assert client.get("/health").json()["status"] == "ok"
    assert "MédiPredict" in client.get("/").text


def test_predict_returns_probability_and_three_factors():
    r = client.post("/predict", json=PATIENT).json()
    assert 0 <= r["probability_abnormal"] <= 1
    assert r["label"] in ("Normal", "Abnormal")
    assert len(r["factors"]) == 3 and {"label", "impact", "direction"} <= set(r["factors"][0])
    assert r["saved"] is False


def test_validation_rejects_impossible_values():
    assert client.post("/predict", json={**PATIENT, "age": 400}).status_code == 422


def test_register_login_history():
    creds = {"email": "doc@example.com", "password": "secret123"}
    assert client.post("/register", json={**creds, "fullname": "Dr Test"}).status_code == 201
    assert client.post("/register", json={**creds, "fullname": "Dr Test"}).status_code == 400
    assert client.post("/login", json={**creds, "password": "wrong-pass"}).status_code == 401
    token = client.post("/login", json=creds).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post("/predict", json=PATIENT, headers=headers).json()["saved"] is True
    history = client.get("/history", headers=headers).json()
    assert len(history) == 1 and history[0]["factors"]
    assert client.get("/stats", headers=headers).json()["total"] == 1
    assert client.get("/history").status_code in (401, 403)


def test_model_card_endpoint():
    m = client.get("/model").json()
    assert m["comparison"] and m["cross_source"] and m["fairness"]
