"""
Module de connexion à MongoDB Atlas.
Fournit une fonction unique get_database() utilisée dans tout le projet.
"""

from pymongo import MongoClient
from pymongo.server_api import ServerApi

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import MONGODB_URI, MONGODB_DB_NAME

_client = None


def get_client():
    """Retourne un client MongoDB unique (singleton) pour éviter les connexions multiples."""
    global _client
    if _client is None:
        if not MONGODB_URI:
            raise ValueError(
                "MONGODB_URI introuvable. Vérifie ton fichier .env à la racine du projet."
            )
        _client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
    return _client


def get_database(db_name: str = None):
    """Retourne l'objet base de données MongoDB."""
    client = get_client()
    return client[db_name or MONGODB_DB_NAME]


def test_connection():
    """Teste la connexion à MongoDB Atlas en envoyant un ping."""
    try:
        client = get_client()
        client.admin.command("ping")
        print("✅ Connexion à MongoDB Atlas réussie !")
        db = get_database()
        print(f"📂 Base de données active : {db.name}")
        print(f"📋 Collections existantes : {db.list_collection_names()}")
        return True
    except Exception as e:
        print(f"❌ Échec de connexion à MongoDB Atlas : {e}")
        return False


if __name__ == "__main__":
    test_connection()
