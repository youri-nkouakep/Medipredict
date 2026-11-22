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

# Pipeline exported by ml/train.py: preprocessing + XGBoost + decision threshold
# (Youden index on out-of-fold training predictions).
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "medipredict.joblib"))
RESULTS_DIR = BASE_DIR.parent / "results"
