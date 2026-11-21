"""Configuration lue dans les variables d'environnement (fichier .env à la racine)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")


def _obligatoire(nom: str) -> str:
    valeur = os.getenv(nom)
    if not valeur:
        raise RuntimeError(f"Variable d'environnement manquante : {nom} (voir .env.example)")
    return valeur


SECRET_KEY = _obligatoire("SECRET_KEY")
DATABASE_URL = _obligatoire("DATABASE_URL")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

MODEL_PATH = BASE_DIR / "model" / "xgboost.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"
# Seuil de décision de XGBoost (indice de Youden, calculé sur le jeu d'entraînement,
# notebook 02). C'est avec ce seuil que sont obtenues les métriques du README.
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.689"))
