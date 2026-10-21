import requests
import math
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from src.utils.db import get_database

load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "x-apisports-key": API_KEY
}
WC_2022_ID = 1  # World Cup
SEASON = 2022

def api_get(endpoint, params={}, delay=6):
    time.sleep(delay)  # respect free plan limit (10 req/min)
    r = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()

def clean(record):
    if isinstance(record, dict):
        return {k: clean(v) for k, v in record.items()}
    if isinstance(record, list):
        return [clean(v) for v in record]
    if isinstance(record, float) and math.isnan(record):
        return None
    return record

def main():
    print("📡 Collecte des données Coupe du Monde 2022 via API-Football...\n")
    db = get_database()

    # ── 1. Infos du tournoi ────────────────────────────────────────────────
    print("1️⃣  Récupération des infos du tournoi...")
    data = api_get("leagues", {"id": WC_2022_ID, "season": SEASON})
    league_info = data.get("response", [])
    print(f"   ✅ {len(league_info)} tournoi(s) trouvé(s)")

    # ── 2. Groupes & classements ───────────────────────────────────────────
    print("2️⃣  Récupération des groupes et classements...")
    data = api_get("standings", {"league": WC_2022_ID, "season": SEASON})
    standings = data.get("response", [])
    
    groups_2022 = []
    if standings:
        league_standings = standings[0].get("league", {}).get("standings", [])
        for group in league_standings:
            for team_entry in group:
                groups_2022.append(clean(team_entry))
    
    if groups_2022:
        db["wc2022_standings"].drop()
        db["wc2022_standings"].insert_many(groups_2022)
        print(f"   ✅ {len(groups_2022)} entrées de classement importées")
    else:
        print("   ⚠️  Pas de classements trouvés")

    # ── 3. Matchs ─────────────────────────────────────────────────────────
    print("3️⃣  Récupération des matchs...")
    data = api_get("fixtures", {"league": WC_2022_ID, "season": SEASON})
    fixtures = data.get("response", [])
    print(f"   ✅ {len(fixtures)} matchs trouvés")

    if fixtures:
        db["wc2022_matches"].drop()
        db["wc2022_matches"].insert_many([clean(f) for f in fixtures])
        print(f"   ✅ {len(fixtures)} matchs importés dans MongoDB")

    # ── 4. Statistiques par match ──────────────────────────────────────────
    print("4️⃣  Récupération des stats par match...")
    print(f"   ⏳ {len(fixtures)} matchs × 6s = ~{len(fixtures)*6//60} min d'attente (limite API gratuite)")
    stats_all = []
    for i, fixture in enumerate(fixtures):
        fid = fixture["fixture"]["id"]
        try:
            data = api_get("fixtures/statistics", {"fixture": fid})
            stats = data.get("response", [])
            for s in stats:
                s["fixture_id"] = fid
                stats_all.append(clean(s))
        except Exception as e:
            print(f"   ⚠️  Match {fid} : {e}")
        
        if (i+1) % 10 == 0:
            print(f"   ... {i+1}/{len(fixtures)} matchs traités")

    if stats_all:
        db["wc2022_stats"].drop()
        db["wc2022_stats"].insert_many(stats_all)
        print(f"   ✅ {len(stats_all)} entrées de stats importées")

    # ── 5. Buteurs ────────────────────────────────────────────────────────
    print("5️⃣  Récupération des buteurs...")
    data = api_get("players/topscorers", {"league": WC_2022_ID, "season": SEASON})
    scorers = data.get("response", [])
    if scorers:
        db["wc2022_topscorers"].drop()
        db["wc2022_topscorers"].insert_many([clean(s) for s in scorers])
        print(f"   ✅ {len(scorers)} buteurs importés")

    # ── Résumé ─────────────────────────────────────────────────────────────
    print("\n📊 Résumé dans MongoDB :")
    for col in ["wc2022_standings", "wc2022_matches", "wc2022_stats", "wc2022_topscorers"]:
        count = db[col].count_documents({})
        print(f"   - {col}: {count} documents")

    print("\n✅ Collecte 2022 terminée !")

if __name__ == "__main__":
    main()
