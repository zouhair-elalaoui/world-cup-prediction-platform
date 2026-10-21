import math
import pandas as pd
from src.utils.db import get_database

def clean(v):
    return 0.0 if (isinstance(v, float) and math.isnan(v)) else v

def main():
    print("⚙️  Intégration des données 2022 dans le dataset d'entraînement...\n")
    db = get_database()

    matches_2022 = pd.DataFrame(list(db["wc2022_matches"].find({}, {"_id": 0})))

    rows = []
    for _, m in matches_2022.iterrows():
        home = m["teams"]["home"]["name"]
        away = m["teams"]["away"]["name"]
        home_score = m["goals"]["home"]
        away_score = m["goals"]["away"]

        if home_score is None or away_score is None:
            continue

        home_score = int(home_score)
        away_score = int(away_score)

        if home_score > away_score:
            result = "home team win"
            home_win, away_win, draw = 1, 0, 0
        elif home_score < away_score:
            result = "away team win"
            home_win, away_win, draw = 0, 1, 0
        else:
            result = "draw"
            home_win, away_win, draw = 0, 0, 1

        rows.append({
            "tournament_id":    "WC-2022",
            "tournament_name":  "2022 FIFA World Cup",
            "match_id":         str(m["fixture"]["id"]),
            "stage_name":       m["league"].get("round", ""),
            "match_date":       str(m["fixture"]["date"])[:10],
            "home_team_name":   home,
            "away_team_name":   away,
            "home_team_score":  home_score,
            "away_team_score":  away_score,
            "home_team_score_margin": home_score - away_score,
            "away_team_score_margin": away_score - home_score,
            "extra_time":       1 if m["fixture"]["status"]["short"] in ["AET", "PEN"] else 0,
            "penalty_shootout": 1 if m["fixture"]["status"]["short"] == "PEN" else 0,
            "result":           result,
            "home_team_win":    home_win,
            "away_team_win":    away_win,
            "draw":             draw,
            "group_stage":      1 if "Group" in str(m["league"].get("round", "")) else 0,
            "knockout_stage":   0 if "Group" in str(m["league"].get("round", "")) else 1,
        })

    df_2022 = pd.DataFrame(rows)
    db["wc2022_matches_clean"].drop()
    db["wc2022_matches_clean"].insert_many(df_2022.to_dict(orient="records"))
    print(f"✅ {len(df_2022)} matchs 2022 nettoyés et importés")

    # Ajouter aussi dans tournament_features
    print("\n⚙️  Calcul des features 2022 par équipe...")
    t_feats_new = []
    teams = list(set(df_2022['home_team_name'].tolist() + df_2022['away_team_name'].tolist()))

    for team in teams:
        home_m = df_2022[df_2022['home_team_name'] == team]
        away_m = df_2022[df_2022['away_team_name'] == team]

        matches_played = len(home_m) + len(away_m)
        wins  = home_m['home_team_win'].sum() + away_m['away_team_win'].sum()
        draws = home_m['draw'].sum() + away_m['draw'].sum()
        losses = matches_played - wins - draws
        gf = home_m['home_team_score'].sum() + away_m['away_team_score'].sum()
        ga = home_m['away_team_score'].sum() + away_m['home_team_score'].sum()
        ko = home_m['knockout_stage'].sum() + away_m['knockout_stage'].sum()

        t_feats_new.append({
            "team_id":      "",
            "tournament_id": "WC-2022",
            "team_name":    team,
            "year":         2022,
            "matches":      int(matches_played),
            "wins":         int(wins),
            "draws":        int(draws),
            "losses":       int(losses),
            "goals_for":    int(gf),
            "goals_against": int(ga),
            "goal_diff":    int(gf - ga),
            "win_rate":     wins / matches_played if matches_played > 0 else 0,
            "avg_goals_for": gf / matches_played if matches_played > 0 else 0,
            "knockout_games": int(ko),
        })

    db["tournament_features"].insert_many(t_feats_new)
    print(f"✅ {len(t_feats_new)} équipes 2022 ajoutées dans tournament_features")
    print(f"\n✅ Total tournament_features : {db['tournament_features'].count_documents({})} documents")

if __name__ == "__main__":
    main()
