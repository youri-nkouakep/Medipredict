import json

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import joblib

from config import DECISION_THRESHOLD, MODEL_PATH
from database import engine, get_db
from models import Base, User, Prediction
from schemas import UserRegister, UserLogin, TokenOut, PatientData, PredictionOut
from auth import (
    hash_password, verify_password,
    create_access_token,
    get_current_user, get_current_user_optional
)
from preprocess import preprocess_input

# ── CREATE TABLES ─────────────────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MediPredict API",
    description="API de détection d'anomalies médicales",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LOAD ML MODEL ─────────────────────────────────────────────────────────────
# XGBoost : meilleur modèle de l'étude (AUC-ROC 0,940, Brier 0,088), plus léger
# que le Random Forest (7 Mo contre 33 Mo).
model = joblib.load(MODEL_PATH)


# ── ROOT ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "MediPredict API v2.1",
        "status":  "online",
        "model":   "XGBoost",
        "decision_threshold": DECISION_THRESHOLD,
        "endpoints": {
            "POST /register":  "Créer un compte",
            "POST /login":     "Se connecter",
            "POST /predict":   "Faire une prédiction",
            "GET  /history":   "Historique (authentifié)",
            "GET  /me":        "Profil utilisateur (authentifié)",
            "GET  /stats":     "Statistiques (authentifié)",
        }
    }


# ── REGISTER ──────────────────────────────────────────────────────────────────
@app.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email déjà utilisé")

    new_user = User(
        fullname=user.fullname,
        email=user.email,
        password=hash_password(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token({"sub": new_user.email})
    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":         new_user.id,
            "fullname":   new_user.fullname,
            "email":      new_user.email,
            "created_at": str(new_user.created_at),
        }
    }


# ── LOGIN ─────────────────────────────────────────────────────────────────────
@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()

    if not db_user:
        raise HTTPException(status_code=401, detail="Email incorrect")

    if not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Mot de passe incorrect")

    token = create_access_token({"sub": db_user.email})
    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":         db_user.id,
            "fullname":   db_user.fullname,
            "email":      db_user.email,
            "created_at": str(db_user.created_at),
        }
    }


# ── ME ────────────────────────────────────────────────────────────────────────
@app.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "id":         current_user.id,
        "fullname":   current_user.fullname,
        "email":      current_user.email,
        "created_at": str(current_user.created_at),
    }


# ── PREDICT ───────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(
    patient: PatientData,
    db:      Session = Depends(get_db),
    current_user: User = Depends(get_current_user_optional),
):
    try:
        data = patient.model_dump()
        X = preprocess_input(data)

        # Décision au seuil de Youden (et non 0,5), comme dans l'évaluation
        probabilities = model.predict_proba(X)[0]
        prediction    = int(probabilities[1] >= DECISION_THRESHOLD)
        label         = "Abnormal" if prediction == 1 else "Normal"
        confidence    = float(max(probabilities) * 100)

        if current_user:
            record = Prediction(
                user_id              = current_user.id,
                label                = label,
                confidence           = confidence,
                probability_normal   = float(probabilities[0]),
                probability_abnormal = float(probabilities[1]),
                patient_data         = json.dumps(data),
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            prediction_id = record.id
        else:
            prediction_id = None

        return {
            "prediction_id":        prediction_id,
            "prediction":           int(prediction),
            "label":                label,
            "probability_normal":   float(probabilities[0] * 100),
            "probability_abnormal": float(probabilities[1] * 100),
            "confidence":           confidence,
            "saved":                current_user is not None,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── HISTORY ───────────────────────────────────────────────────────────────────
@app.get("/history")
def history(
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    records = (
        db.query(Prediction)
          .filter(Prediction.user_id == current_user.id)
          .order_by(Prediction.created_at.desc())
          .all()
    )
    return [
        {
            "id":                   r.id,
            "label":                r.label,
            "confidence":           r.confidence,
            "probability_normal":   r.probability_normal,
            "probability_abnormal": r.probability_abnormal,
            "patient_data":         r.patient_data,
            "created_at":           str(r.created_at),
        }
        for r in records
    ]


# ── STATS ─────────────────────────────────────────────────────────────────────
@app.get("/stats")
def stats(
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user),
):
    records = db.query(Prediction).filter(
        Prediction.user_id == current_user.id
    ).all()

    total    = len(records)
    abnormal = sum(1 for r in records if r.label == "Abnormal")
    normal   = total - abnormal

    return {
        "total":    total,
        "normal":   normal,
        "abnormal": abnormal,
        "avg_confidence": round(
            sum(r.confidence for r in records) / total, 2
        ) if total else 0,
    }


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
