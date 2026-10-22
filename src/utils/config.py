"""
Configuration centrale du projet.
Charge les variables d'environnement depuis le fichier .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Racine du projet (2 niveaux au-dessus de ce fichier : src/utils/ -> racine)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Charge le fichier .env situé à la racine du projet
load_dotenv(BASE_DIR / ".env")

# --- MongoDB ---
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "worldcup_db")

# --- API-Football ---
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
API_FOOTBALL_HOST = os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io")

# --- Kaggle ---
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# --- Chemins du projet ---
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_INTERIM_DIR = BASE_DIR / "data" / "interim"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def check_config():
    """Vérifie que les variables essentielles sont bien chargées."""
    missing = []
    if not MONGODB_URI:
        missing.append("MONGODB_URI")
    if missing:
        print(f"⚠️  Variables manquantes dans .env : {', '.join(missing)}")
        return False
    print("✅ Configuration chargée avec succès.")
    return True


if __name__ == "__main__":
    check_config()
    print(f"BASE_DIR = {BASE_DIR}")
