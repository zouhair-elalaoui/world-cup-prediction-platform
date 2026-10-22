"""
Point d'entrée principal du projet.
Pour l'instant : vérifie que l'environnement est correctement configuré
(variables .env + connexion MongoDB Atlas).
"""

from src.utils.config import check_config
from src.utils.db import test_connection


def main():
    print("=" * 50)
    print("🏆 World Cup Intelligence & Prediction Platform")
    print("=" * 50)

    print("\n[1/2] Vérification de la configuration...")
    config_ok = check_config()

    print("\n[2/2] Test de connexion MongoDB Atlas...")
    db_ok = test_connection()

    print("\n" + "=" * 50)
    if config_ok and db_ok:
        print("✅ Environnement prêt. Tu peux lancer la collecte des données.")
    else:
        print("⚠️  Corrige les points ci-dessus avant de continuer.")
    print("=" * 50)


if __name__ == "__main__":
    main()
