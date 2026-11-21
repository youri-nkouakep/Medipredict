from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


# ── AUTH ──────────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    fullname: str = Field(..., min_length=2, max_length=100)
    email:    EmailStr
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    email:    EmailStr
    password: str


class UserOut(BaseModel):
    id:         int
    fullname:   str
    email:      str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenOut(BaseModel):
    access_token: str
    token_type:   str
    user:         UserOut


# ── PREDICTION INPUT ──────────────────────────────────────────────────────────
# 20 variables utilisées par le modèle XGBoost (model/xgboost.pkl)
# Encodages :
#   physical_activity  : Low=0 | Moderate=1 | High=2
#   stress_level       : Low=0 | Medium=1   | High=2
#   sugar_consumption  : Low=0 | Medium=1   | High=2
#   family_history     : Non=0 | Oui=1
#   high_blood_pressure: Non=0 | Oui=1
#   gender_Male        : Femme=0 | Homme=1
#   smoking_Former/Never : one-hot  (Current → les deux à 0)
#   education_level_Secondary/Tertiary : one-hot (Primary → les deux à 0)
#   employment_status_Retired/Unemployed : one-hot (Employed → les deux à 0)

class PatientData(BaseModel):
    age:                          float = Field(..., ge=1.0,   le=120.0)
    bmi:                          float = Field(..., ge=10.0,  le=70.0)
    HbA1c_level:                  float = Field(..., ge=3.0,   le=20.0)
    glucose:                      float = Field(..., ge=20.0,  le=600.0)
    cholesterol:                  float = Field(..., ge=50.0,  le=600.0)
    sleep_hours:                  float = Field(..., ge=0.0,   le=24.0)
    triglycerides:                float = Field(..., ge=20.0,  le=1000.0)
    physical_activity:            int   = Field(..., ge=0,     le=2)   
    family_history:               int   = Field(..., ge=0,     le=1)
    stress_level:                 int   = Field(..., ge=0,     le=2)  
    high_blood_pressure:          int   = Field(..., ge=0,     le=1)
    sugar_consumption:            int   = Field(..., ge=0,     le=2)  
    crp_level:                    float = Field(..., ge=0.0,   le=100.0)
    gender_Male:                  int   = Field(..., ge=0,     le=1)
    smoking_Former:               int   = Field(..., ge=0,     le=1)
    smoking_Never:                int   = Field(..., ge=0,     le=1)
    education_level_Secondary:    int   = Field(..., ge=0,     le=1)
    education_level_Tertiary:     int   = Field(..., ge=0,     le=1)
    employment_status_Retired:    int   = Field(..., ge=0,     le=1)
    employment_status_Unemployed: int   = Field(..., ge=0,     le=1)


# ── PREDICTION OUTPUT ─────────────────────────────────────────────────────────

class PredictionOut(BaseModel):
    id:                   int
    label:                str
    confidence:           float
    probability_normal:   float
    probability_abnormal: float
    created_at:           datetime

    class Config:
        from_attributes = True
