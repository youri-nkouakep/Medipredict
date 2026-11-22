"""MédiPredict API. Run from backend/: uvicorn app:app --reload (docs at /docs)."""
import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from auth import create_access_token, get_current_user, get_current_user_optional, hash_password, verify_password
from config import RESULTS_DIR
from database import engine, get_db
from models import Base, Prediction, User
from predictor import Predictor
from schemas import PatientData, UserLogin, UserRegister

Base.metadata.create_all(bind=engine)

app = FastAPI(title="MédiPredict API", version="3.0.0",
              description="Prédiction du risque de maladies chroniques, avec explication SHAP de chaque prédiction.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = Predictor()
FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _user_out(u: User) -> dict:
    return {"id": u.id, "fullname": u.fullname, "email": u.email, "created_at": str(u.created_at)}


def _token(u: User) -> dict:
    return {"access_token": create_access_token({"sub": u.email}), "token_type": "bearer", "user": _user_out(u)}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "threshold": predictor.threshold}


@app.get("/model")
def model_card():
    """Evaluation summary shown in the interface."""
    import pandas as pd

    def csv(name):
        path = RESULTS_DIR / name
        return pd.read_csv(path).to_dict("records") if path.exists() else []

    analysis = RESULTS_DIR / "analysis.json"
    return {"threshold": predictor.threshold, "comparison": csv("model_comparison.csv"),
            "cross_source": csv("cross_source.csv"), "fairness": csv("fairness.csv"),
            "analysis": json.loads(analysis.read_text()) if analysis.exists() else {}}


@app.post("/register", status_code=201)
def register(user: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(400, "Email déjà utilisé")
    u = User(fullname=user.fullname, email=user.email, password=hash_password(user.password))
    db.add(u)
    db.commit()
    db.refresh(u)
    return _token(u)


@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == user.email).first()
    if not u or not verify_password(user.password, u.password):
        raise HTTPException(401, "Email ou mot de passe incorrect")
    return _token(u)


@app.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return _user_out(current_user)


@app.post("/predict")
def predict(patient: PatientData, db: Session = Depends(get_db),
            current_user: User | None = Depends(get_current_user_optional)):
    data = patient.model_dump()
    out = predictor.predict(data)
    p = out["probability_abnormal"]
    label = "Abnormal" if out["prediction"] else "Normal"
    record = Prediction(user_id=current_user.id if current_user else None, label=label,
                        confidence=max(p, 1 - p) * 100, probability_normal=1 - p, probability_abnormal=p,
                        patient_data=json.dumps(data), factors=json.dumps(out["factors"]))
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"prediction_id": record.id, "label": label, **out, "saved": current_user is not None}


@app.get("/history")
def history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    records = (db.query(Prediction).filter(Prediction.user_id == current_user.id)
               .order_by(Prediction.created_at.desc()).all())
    return [{"id": r.id, "label": r.label, "probability_abnormal": r.probability_abnormal,
             "patient_data": json.loads(r.patient_data or "{}"), "factors": json.loads(r.factors or "[]"),
             "created_at": str(r.created_at)} for r in records]


@app.get("/stats")
def stats(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    records = db.query(Prediction).filter(Prediction.user_id == current_user.id).all()
    abnormal = sum(r.label == "Abnormal" for r in records)
    return {"total": len(records), "abnormal": abnormal, "normal": len(records) - abnormal}
