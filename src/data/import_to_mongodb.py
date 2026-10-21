import pandas as pd
from pymongo import MongoClient, UpdateOne
from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils import config
from src.utils.db import get_database

DATA_PATH = Path("data/raw/historical_1930_2018")

def load_csv(filename):
    path = DATA_PATH / filename
    df = pd.read_csv(path)
    # Convertir les colonnes de dates
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    return df

def import_collection(db, collection_name, df, id_field=None):
    col = db[collection_name]
    col.drop()
    records = df.to_dict(orient='records')
    # Nettoyer les NaN
    import math
    def clean(record):
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in record.items()}
    records = [clean(r) for r in records]
    if records:
        col.insert_many(records)
    print(f"  ✅ {collection_name}: {len(records)} documents importés")

def main():
    print("🚀 Import des données historiques vers MongoDB Atlas...\n")
    db = get_database()

    fichiers = {
        "tournaments":          "tournaments.csv",
        "teams":                "teams.csv",
        "matches":              "matches.csv",
        "team_appearances":     "team_appearances.csv",
        "group_standings":      "group_standings.csv",
        "tournament_standings": "tournament_standings.csv",
        "qualified_teams":      "qualified_teams.csv",
        "goals":                "goals.csv",
        "bookings":             "bookings.csv",
        "substitutions":        "substitutions.csv",
        "players":              "players.csv",
        "squads":               "squads.csv",
        "stadiums":             "stadiums.csv",
        "confederations":       "confederations.csv",
        "award_winners":        "award_winners.csv",
    }

    for collection_name, filename in fichiers.items():
        try:
            df = load_csv(filename)
            import_collection(db, collection_name, df)
        except Exception as e:
            print(f"  ❌ {collection_name}: {e}")

    print("\n📊 Résumé des collections dans MongoDB:")
    for name in db.list_collection_names():
        print(f"  - {name}: {db[name].count_documents({})} documents")

    print("\n✅ Import terminé !")

if __name__ == "__main__":
    main()
