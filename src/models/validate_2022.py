import pickle, math
import pandas as pd
import numpy as np
from src.utils.db import get_database
from sklearn.metrics import accuracy_score, classification_report

def clean(v):
    return 0.0 if (isinstance(v, float) and math.isnan(v)) else v

def get_team_stats(t_feats, team_name):
    hist = t_feats[t_feats['team_name'] == team_name]
    if hist.empty:
        return {
            'cum_matches': 0, 'cum_wins': 0, 'cum_win_rate': 0.25,
            'cum_goals_for': 0, 'cum_goals_against': 0,
            'cum_goal_diff': 0.0, 'cum_appearances': 0,
            'recent_win_rate': 0.25, 'recent_goal_diff': 0.0,
        }
    total_m = hist['matches'].sum()
    total_w = hist['wins'].sum()
    total_gf = hist['goals_for'].sum()
    total_ga = hist['goals_against'].sum()
    recent = hist.tail(3)
    return {
        'cum_matches':       int(total_m),
        'cum_wins':          int(total_w),
        'cum_win_rate':      total_w / total_m if total_m > 0 else 0.25,
        'cum_goals_for':     int(total_gf),
        'cum_goals_against': int(total_ga),
        'cum_goal_diff':     (total_gf - total_ga) / total_m if total_m > 0 else 0.0,
        'cum_appearances':   len(hist),
        'recent_win_rate':   recent['win_rate'].mean() if len(recent) > 0 else 0.25,
        'recent_goal_diff':  recent['goal_diff'].mean() if len(recent) > 0 else 0.0,
    }

def main():
    print("🔍 Validation du modèle sur la Coupe du Monde 2022...\n")
    db = get_database()

    with open("models/xgb_model_v2.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_cols_v2.pkl", "rb") as f:
        cols = pickle.load(f)

    # Charger les matchs 2022 et les features historiques (jusqu'à 2018)
    matches_2022 = pd.DataFrame(list(db["wc2022_matches"].find({}, {"_id": 0})))
    t_feats = pd.DataFrame(list(db["tournament_features"].find({}, {"_id": 0})))

    rows = []
    y_true = []
    match_info = []

    for _, m in matches_2022.iterrows():
        home = m["teams"]["home"]["name"]
        away = m["teams"]["away"]["name"]
        home_score = m["goals"]["home"]
        away_score = m["goals"]["away"]

        if home_score is None or away_score is None:
            continue

        hs = get_team_stats(t_feats, home)
        as_ = get_team_stats(t_feats, away)

        row = {}
        for k in hs:
            row[f'home_{k}'] = clean(hs[k])
            row[f'away_{k}'] = clean(as_[k])
            row[f'diff_{k}'] = row[f'home_{k}'] - row[f'away_{k}']
        rows.append(row)

        # Résultat réel
        if home_score > away_score:
            y_true.append(2)  # home win
        elif home_score == away_score:
            y_true.append(1)  # draw
        else:
            y_true.append(0)  # away win

        match_info.append({
            "match": f"{home} vs {away}",
            "score": f"{int(home_score)}-{int(away_score)}",
            "home": home, "away": away
        })

    X = pd.DataFrame(rows)[cols]
    y_pred = model.predict(X)
    probs = model.predict_proba(X)

    acc = accuracy_score(y_true, y_pred)
    print(f"🎯 Accuracy sur CdM 2022 : {acc:.2%} ({sum(np.array(y_true)==np.array(y_pred))}/{len(y_true)} matchs corrects)\n")
    print(classification_report(y_true, y_pred,
          target_names=['Away win', 'Draw', 'Home win']))

    # Détail des prédictions
    label_map = {0: 'Away win', 1: 'Draw', 2: 'Home win'}
    print("\n📋 Détail des prédictions (phase de groupes) :")
    print(f"{'Match':<35} {'Réel':<12} {'Prédit':<12} {'✓'}")
    print("-" * 65)
    for i, info in enumerate(match_info[:48]):  # groupes seulement
        reel   = label_map[y_true[i]]
        predit = label_map[y_pred[i]]
        ok = "✅" if y_true[i] == y_pred[i] else "❌"
        print(f"{info['match']:<35} {reel:<12} {predit:<12} {ok}")

    print("\n🏆 Détail phase éliminatoire :")
    print(f"{'Match':<35} {'Score':<8} {'Réel':<12} {'Prédit':<12} {'✓'}")
    print("-" * 72)
    for i, info in enumerate(match_info[48:]):
        idx = i + 48
        reel   = label_map[y_true[idx]]
        predit = label_map[y_pred[idx]]
        ok = "✅" if y_true[idx] == y_pred[idx] else "❌"
        print(f"{info['match']:<35} {info['score']:<8} {reel:<12} {predit:<12} {ok}")

if __name__ == "__main__":
    main()
