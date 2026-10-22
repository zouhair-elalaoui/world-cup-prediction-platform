import pandas as pd
import numpy as np
import math, pickle
from pathlib import Path
from src.utils.db import get_database
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def clean(v):
    return 0.0 if (isinstance(v, float) and math.isnan(v)) else v

def get_cumulative_stats(t_feats, team_name, before_year):
    hist = t_feats[(t_feats['team_name'] == team_name) & (t_feats['year'] < before_year)]
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

def build_dataset(matches, t_feats):
    rows = []
    for _, m in matches.iterrows():
        home = m['home_team_name']
        away = m['away_team_name']
        year = int(str(m.get('tournament_id', 'WC-1930')).split('-')[1])

        hs = get_cumulative_stats(t_feats, home, year)
        as_ = get_cumulative_stats(t_feats, away, year)

        row = {}
        for k in hs:
            row[f'home_{k}'] = clean(hs[k])
            row[f'away_{k}'] = clean(as_[k])
            row[f'diff_{k}'] = row[f'home_{k}'] - row[f'away_{k}']

        if m['home_team_win'] == 1:
            row['result'] = 2
        elif m['draw'] == 1:
            row['result'] = 1
        else:
            row['result'] = 0

        rows.append(row)
    return pd.DataFrame(rows)

def main():
    print("🤖 Entraînement modèle v3 (1930-2022)...\n")
    db = get_database()

    # Charger matchs historiques 1930-2018
    matches_hist = pd.DataFrame(list(db['matches'].find({}, {'_id': 0})))

    # Charger matchs 2022 nettoyés
    matches_2022 = pd.DataFrame(list(db['wc2022_matches_clean'].find({}, {'_id': 0})))

    # Combiner
    all_matches = pd.concat([matches_hist, matches_2022], ignore_index=True)
    print(f"✅ Total matchs : {len(all_matches)} (900 hist + {len(matches_2022)} 2022)")

    # Features
    t_feats = pd.DataFrame(list(db['tournament_features'].find({}, {'_id': 0})))
    t_feats['year'] = t_feats['tournament_id'].str.extract(r'(\d{4})').astype(int)
    print(f"✅ {len(t_feats)} entrées de features\n")

    # Construire dataset
    df = build_dataset(all_matches, t_feats)
    print(f"✅ Dataset : {len(df)} matchs\n")

    X = df.drop('result', axis=1)
    y = df['result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy v3 (1930-2022) : {acc:.2%}\n")
    print(classification_report(y_test, y_pred,
          target_names=['Away win', 'Draw', 'Home win']))

    fi = pd.Series(model.feature_importances_, index=X.columns)
    print("🔑 Top 10 features :")
    print(fi.sort_values(ascending=False).head(10).to_string())

    Path("models").mkdir(exist_ok=True)
    with open("models/xgb_model_v3.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/feature_cols_v3.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)
    print("\n✅ Modèle v3 sauvegardé dans models/xgb_model_v3.pkl")

if __name__ == "__main__":
    main()
