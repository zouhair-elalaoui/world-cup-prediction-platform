import pandas as pd
import os
import pickle
import numpy as np
from src.utils.db import get_database

FIFA_RANKING = {
    "Argentina": 1, "Spain": 2, "France": 3, "England": 4,
    "Portugal": 5, "Brazil": 6, "Morocco": 7, "Netherlands": 8,
    "Belgium": 9, "Germany": 10, "Croatia": 11,
    "Colombia": 13, "Senegal": 15, "Mexico": 14, "United States": 17,
    "Uruguay": 16, "Japan": 18, "Switzerland": 19, "Denmark": 21,
    "Ecuador": 23, "South Korea": 25, "Austria": 24, "Turkey": 22,
    "Australia": 27, "Norway": 31, "Canada": 30, "Ukraine": 32,
    "Panama": 34, "Algeria": 28, "Scotland": 42, "Egypt": 29,
    "Paraguay": 41, "Tunisia": 45, "Ivory Coast": 33, "Sweden": 38,
    "Czechia": 40, "Bosnia and Herzegovina": 64, "Ghana": 73,
    "Iraq": 57, "South Africa": 60, "Qatar": 56, "Saudi Arabia": 61,
    "Jordan": 63, "Congo DR": 46, "Cape Verde": 67, "Uzbekistan": 50,
    "New Zealand": 85, "Curacao": 82, "Haiti": 83, "Iran": 20,
}

NAME_MAP = {
    "United States": "United States",
    "Turkey": "Turkey",
    "Ivory Coast": "Ivory Coast",
    "South Korea": "South Korea",
    "Congo DR": "Congo DR",
}

def main():
    print("📊 Export des données pour Power BI...\n")
    db = get_database()
    os.makedirs("exports", exist_ok=True)

    # ── 1. Probabilités par équipe ────────────────────────────────
    print("1️⃣  Export probabilités par équipe...")
    docs = list(db["simulation_probabilities"].find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    df = df.sort_values("proba_champion", ascending=False)
    df.to_csv("exports/probabilites_equipes.csv", index=False)
    print(f"   ✅ probabilites_equipes.csv ({len(df)} équipes)")

    # ── 2. Classements des groupes ────────────────────────────────
    print("2️⃣  Export classements des groupes...")
    docs = list(db["simulation_groups"].find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    df.to_csv("exports/groupes.csv", index=False)
    print(f"   ✅ groupes.csv ({len(df)} entrées)")

    # ── 3. Probabilités par groupe ────────────────────────────────
    print("3️⃣  Export probabilités par groupe...")
    docs = list(db["simulation_group_probas"].find({}, {"_id": 0}))
    if docs:
        df = pd.DataFrame(docs)
        df = df.sort_values(["group", "proba_1st"], ascending=[True, False])
        df.to_csv("exports/probabilites_groupes.csv", index=False)
        print(f"   ✅ probabilites_groupes.csv ({len(df)} entrées)")
    else:
        print("   ⚠️  Pas de données simulation_group_probas — relance la simulation")

    # ── 4. Prédiction finale ──────────────────────────────────────
    print("4️⃣  Export prédiction finale...")
    doc = db["simulation_prediction"].find_one({}, {"_id": 0})
    rows = [
        {"position": 1, "medal": "🥇", "team": doc["champion"]["team"], "proba": doc["champion"]["proba"]},
        {"position": 2, "medal": "🥈", "team": doc["finalist"]["team"], "proba": doc["finalist"]["proba"]},
        {"position": 3, "medal": "🥉", "team": doc["third"]["team"],    "proba": doc["third"]["proba"]},
        {"position": 4, "medal": "4e",  "team": doc["fourth"]["team"],  "proba": doc["fourth"]["proba"]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv("exports/prediction_finale.csv", index=False)
    print(f"   ✅ prediction_finale.csv")

    # ── 5. Stats historiques par équipe ───────────────────────────
    print("5️⃣  Export stats historiques...")
    docs = list(db["team_features"].find({}, {"_id": 0}))
    df = pd.DataFrame(docs)
    df = df.sort_values("win_rate", ascending=False)
    df.to_csv("exports/stats_historiques.csv", index=False)
    print(f"   ✅ stats_historiques.csv ({len(df)} équipes)")

    # ── 6. Simulation matchs (toutes combinaisons possibles) ──────
    print("6️⃣  Export simulation matchs...")

    with open("models/lr_model_v4.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open("models/lr_feature_cols_v4.pkl", "rb") as f:
        lr_cols = pickle.load(f)

    t_feats = pd.DataFrame(list(db["tournament_features"].find({}, {"_id": 0})))
    t_feats["year"] = t_feats["tournament_id"].str.extract(r"(\d{4})").astype(int)

    def get_team_stats(team_name):
        mapped = NAME_MAP.get(team_name, team_name)
        hist = t_feats[t_feats["team_name"] == mapped]
        if hist.empty:
            return {"cum_matches": 5, "cum_wins": 1, "cum_win_rate": 0.20,
                    "cum_goals_for": 5, "cum_goals_against": 8, "cum_goal_diff": -0.3,
                    "cum_appearances": 1, "recent_win_rate": 0.20, "recent_goal_diff": -0.3}
        total_m = hist["matches"].sum(); total_w = hist["wins"].sum()
        total_gf = hist["goals_for"].sum(); total_ga = hist["goals_against"].sum()
        recent = hist.tail(3)
        return {"cum_matches": int(total_m), "cum_wins": int(total_w),
                "cum_win_rate": total_w/total_m if total_m > 0 else 0.20,
                "cum_goals_for": int(total_gf), "cum_goals_against": int(total_ga),
                "cum_goal_diff": (total_gf-total_ga)/total_m if total_m > 0 else 0.0,
                "cum_appearances": len(hist),
                "recent_win_rate": recent["win_rate"].mean() if len(recent) > 0 else 0.20,
                "recent_goal_diff": recent["goal_diff"].mean() if len(recent) > 0 else 0.0}

    def fifa_to_elo(rank):
        return 2000 - (rank - 1) * (600 / 84)

    def predict(team1, team2):
        rank1 = FIFA_RANKING.get(team1, 85)
        rank2 = FIFA_RANKING.get(team2, 85)
        elo1 = fifa_to_elo(rank1); elo2 = fifa_to_elo(rank2)
        diff = elo1 - elo2
        exp1 = 1 / (1 + 10 ** (-diff / 400))
        elo_p = np.array([1-exp1, 0.0, exp1])

        hs = get_team_stats(team1); as_ = get_team_stats(team2)
        row1 = {}; row2 = {}
        for k in hs:
            row1[f"home_{k}"] = hs[k]; row1[f"away_{k}"] = as_[k]
            row1[f"diff_{k}"] = hs[k] - as_[k]
            row2[f"home_{k}"] = as_[k]; row2[f"away_{k}"] = hs[k]
            row2[f"diff_{k}"] = as_[k] - hs[k]

        X1 = pd.DataFrame([row1]); X2 = pd.DataFrame([row2])
        for c in lr_cols:
            if c not in X1.columns: X1[c] = 0.0
            if c not in X2.columns: X2[c] = 0.0

        p1 = lr_model.predict_proba(X1[lr_cols])[0]
        p2 = lr_model.predict_proba(X2[lr_cols])[0]
        p2_inv = np.array([p2[2], p2[1], p2[0]])
        lr_p = (p1 + p2_inv) / 2

        ph = lr_p[2] + lr_p[1] * 0.5
        pa = lr_p[0] + lr_p[1] * 0.5
        lr_ko = np.array([pa, 0.0, ph]) / (ph + pa)

        combined = 0.60 * elo_p + 0.40 * lr_ko
        combined /= combined.sum()
        return combined[2], combined[0]

    teams = list(FIFA_RANKING.keys())
    match_rows = []
    for t1 in teams:
        for t2 in teams:
            if t1 != t2:
                p1, p2 = predict(t1, t2)
                match_rows.append({
                    "equipe_1":          t1,
                    "equipe_2":          t2,
                    "fifa_rank_1":       FIFA_RANKING.get(t1, 99),
                    "fifa_rank_2":       FIFA_RANKING.get(t2, 99),
                    "proba_victoire_1":  round(p1 * 100, 1),
                    "proba_victoire_2":  round(p2 * 100, 1),
                })

    df_matches = pd.DataFrame(match_rows)
    df_matches.to_csv("exports/simulation_matchs.csv", index=False)
    print(f"   ✅ simulation_matchs.csv ({len(df_matches)} combinaisons)")

    # ── Résumé ─────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("✅ Export terminé ! Fichiers dans exports/")
    print("="*50)
    for f in ["probabilites_equipes.csv", "probabilites_groupes.csv",
              "groupes.csv", "prediction_finale.csv",
              "stats_historiques.csv", "simulation_matchs.csv"]:
        path = f"exports/{f}"
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"   📄 {f} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
