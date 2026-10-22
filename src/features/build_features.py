import pandas as pd
import math
from pathlib import Path
from src.utils.db import get_database

def main():
    print("⚙️  Feature Engineering — calcul des indicateurs par équipe...\n")
    db = get_database()

    # Charger team_appearances depuis MongoDB
    records = list(db['team_appearances'].find({}, {'_id': 0}))
    df = pd.DataFrame(records)

    # ── 1. Stats globales par équipe (toutes éditions confondues) ──────────
    team_stats = df.groupby('team_id').agg(
        team_name        = ('team_name', 'first'),
        team_code        = ('team_code', 'first'),
        total_matches    = ('match_id', 'count'),
        total_wins       = ('win', 'sum'),
        total_draws      = ('draw', 'sum'),
        total_losses     = ('lose', 'sum'),
        total_goals_for  = ('goals_for', 'sum'),
        total_goals_against = ('goals_against', 'sum'),
        appearances      = ('tournament_id', 'nunique'),
    ).reset_index()

    team_stats['win_rate']       = team_stats['total_wins'] / team_stats['total_matches']
    team_stats['draw_rate']      = team_stats['total_draws'] / team_stats['total_matches']
    team_stats['loss_rate']      = team_stats['total_losses'] / team_stats['total_matches']
    team_stats['avg_goals_for']  = team_stats['total_goals_for'] / team_stats['total_matches']
    team_stats['avg_goals_against'] = team_stats['total_goals_against'] / team_stats['total_matches']
    team_stats['goal_diff_avg']  = team_stats['avg_goals_for'] - team_stats['avg_goals_against']

    print(f"✅ Stats globales calculées pour {len(team_stats)} équipes")

    # ── 2. Stats par tournoi (performance à chaque édition) ────────────────
    tournament_stats = df.groupby(['team_id', 'tournament_id']).agg(
        team_name     = ('team_name', 'first'),
        year          = ('match_date', 'first'),
        matches       = ('match_id', 'count'),
        wins          = ('win', 'sum'),
        draws         = ('draw', 'sum'),
        losses        = ('lose', 'sum'),
        goals_for     = ('goals_for', 'sum'),
        goals_against = ('goals_against', 'sum'),
        knockout_games= ('knockout_stage', 'sum'),
    ).reset_index()

    tournament_stats['win_rate']      = tournament_stats['wins'] / tournament_stats['matches']
    tournament_stats['goal_diff']     = tournament_stats['goals_for'] - tournament_stats['goals_against']
    tournament_stats['avg_goals_for'] = tournament_stats['goals_for'] / tournament_stats['matches']

    print(f"✅ Stats par tournoi calculées : {len(tournament_stats)} entrées")

    # ── 3. Forme récente (3 dernières éditions par équipe) ─────────────────
    tournament_stats_sorted = tournament_stats.sort_values(['team_id', 'tournament_id'])
    recent = tournament_stats_sorted.groupby('team_id').tail(3)
    recent_form = recent.groupby('team_id').agg(
        recent_win_rate   = ('win_rate', 'mean'),
        recent_goal_diff  = ('goal_diff', 'mean'),
        recent_avg_goals  = ('avg_goals_for', 'mean'),
    ).reset_index()

    print(f"✅ Forme récente calculée pour {len(recent_form)} équipes")

    # ── 4. Merge toutes les features ───────────────────────────────────────
    features = team_stats.merge(recent_form, on='team_id', how='left')

    # ── 5. Ajouter le titre (vainqueur de tournoi) ─────────────────────────
    tournaments = pd.DataFrame(list(db['tournaments'].find({}, {'_id': 0})))
    winners = tournaments[['winner']].value_counts().reset_index()
    winners.columns = ['team_name', 'titles']
    features = features.merge(winners, on='team_name', how='left')
    features['titles'] = features['titles'].fillna(0).astype(int)

    print(f"✅ Titres ajoutés")

    # ── 6. Sauvegarder dans MongoDB ────────────────────────────────────────
    def clean(record):
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in record.items()}

    records_global = [clean(r) for r in features.to_dict(orient='records')]
    db['team_features'].drop()
    db['team_features'].insert_many(records_global)
    print(f"✅ Collection 'team_features' : {len(records_global)} documents dans MongoDB")

    records_tournament = [clean(r) for r in tournament_stats.to_dict(orient='records')]
    db['tournament_features'].drop()
    db['tournament_features'].insert_many(records_tournament)
    print(f"✅ Collection 'tournament_features' : {len(records_tournament)} documents dans MongoDB")

    # ── 7. Aperçu ──────────────────────────────────────────────────────────
    print("\n📊 Aperçu des features (top 10 équipes par win_rate) :")
    top = features.sort_values('win_rate', ascending=False)[
        ['team_name', 'appearances', 'total_matches', 'win_rate', 'avg_goals_for', 'titles']
    ].head(10)
    print(top.to_string(index=False))

if __name__ == "__main__":
    main()
