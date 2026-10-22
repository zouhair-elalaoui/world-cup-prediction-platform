#!/bin/bash
echo "🚀 World Cup 2026 Prediction Platform"
echo "=========================================================="

# ── 1. Vérifier Python ────────────────────────────────────────────
echo ""
echo "1️⃣  Vérification de Python..."
python3 --version || { echo "❌ Python3 non installé"; exit 1; }
echo "✅ Python OK"

# ── 2. Créer le venv si nécessaire ───────────────────────────────
echo ""
echo "2️⃣  Configuration du venv..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ venv créé"
else
    echo "✅ venv déjà existant"
fi
source venv/bin/activate

# ── 3. Installer les dépendances ──────────────────────────────────
echo ""
echo "3️⃣  Installation des dépendances..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Dépendances installées"

# ── 4. Configurer .env ────────────────────────────────────────────
echo ""
echo "4️⃣  Configuration des variables d'environnement..."

# MongoDB URI
if grep -q "mongodb+srv" .env 2>/dev/null; then
    echo "✅ MongoDB URI déjà configurée"
else
    read -p "   Colle ta MongoDB URI : " MONGO_URI
    if [ -f ".env" ]; then
        sed -i "s|MONGODB_URI=.*|MONGODB_URI=$MONGO_URI|" .env
    else
        echo "MONGODB_URI=$MONGO_URI" > .env
        echo "MONGODB_DB_NAME=world_cup_db" >> .env
        echo "API_FOOTBALL_KEY=" >> .env
    fi
    echo "✅ MongoDB URI configurée"
fi

# API Football Key
if grep -q "API_FOOTBALL_KEY=." .env 2>/dev/null; then
    echo "✅ API Football Key déjà configurée"
else
    read -p "   Colle ta clé API Football : " API_KEY
    if grep -q "API_FOOTBALL_KEY" .env 2>/dev/null; then
        sed -i "s|API_FOOTBALL_KEY=.*|API_FOOTBALL_KEY=$API_KEY|" .env
    else
        echo "API_FOOTBALL_KEY=$API_KEY" >> .env
    fi
    echo "✅ API Football Key configurée"
fi

# ── 5. Tester MongoDB ─────────────────────────────────────────────
echo ""
echo "5️⃣  Test de connexion MongoDB..."
python main.py || { echo "❌ Connexion MongoDB échouée — vérifie ton URI"; exit 1; }

# ── 6. Vérifier les données en base ──────────────────────────────
echo ""
echo "6️⃣  Vérification des données historiques..."
python3 -c "
from src.utils.db import get_database
db = get_database()
count = db['matches'].count_documents({})
print(f'   matches: {count} documents')
exit(0 if count > 0 else 1)
"

if [ $? -ne 0 ]; then
    echo "   ⚠️  Base vide — import des données historiques (1930-2018)..."
    python -m src.data.import_to_mongodb || { echo "❌ Import échoué"; exit 1; }
    python -m src.data.fix_players
    echo "   ✅ Données historiques importées"
else
    echo "   ✅ Données historiques déjà en base"
fi

# ── 7. Vérifier les données 2022 ─────────────────────────────────
echo ""
echo "7️⃣  Vérification des données 2022..."
python3 -c "
from src.utils.db import get_database
db = get_database()
count = db['wc2022_matches'].count_documents({})
print(f'   wc2022_matches: {count} documents')
exit(0 if count > 0 else 1)
"

if [ $? -ne 0 ]; then
    echo "   ⚠️  Données 2022 manquantes — collecte via API Football..."
    python -m src.data.fetch_2022 || { echo "❌ Fetch 2022 échoué — vérifie ta clé API"; exit 1; }
    echo "   ✅ Données 2022 collectées"
else
    echo "   ✅ Données 2022 déjà en base"
fi

# ── 8. Vérifier préparation 2022 ─────────────────────────────────
echo ""
echo "8️⃣  Vérification préparation données 2022..."
python3 -c "
from src.utils.db import get_database
db = get_database()
count = db['wc2022_matches_clean'].count_documents({})
print(f'   wc2022_matches_clean: {count} documents')
exit(0 if count > 0 else 1)
"

if [ $? -ne 0 ]; then
    echo "   ⚠️  Préparation 2022 manquante..."
    python -m src.data.prepare_2022_for_training
    echo "   ✅ Données 2022 préparées"
else
    echo "   ✅ Données 2022 déjà préparées"
fi

# ── 9. Vérifier features ──────────────────────────────────────────
echo ""
echo "9️⃣  Vérification des features..."
python3 -c "
from src.utils.db import get_database
db = get_database()
count = db['team_features'].count_documents({})
print(f'   team_features: {count} documents')
exit(0 if count > 0 else 1)
"

if [ $? -ne 0 ]; then
    echo "   ⚠️  Features manquantes — feature engineering..."
    python -m src.features.build_features
    echo "   ✅ Features calculées"
else
    echo "   ✅ Features déjà en base"
fi

# ── 10. Vérifier les modèles ──────────────────────────────────────
echo ""
echo "🔟  Vérification des modèles ML..."

if [ ! -f "models/xgb_model_v3.pkl" ]; then
    echo "   ⚠️  Modèle XGBoost manquant — entraînement..."
    python -m src.models.train_model_v3
    echo "   ✅ XGBoost v3 entraîné"
else
    echo "   ✅ XGBoost v3 présent"
fi

if [ ! -f "models/lr_model_v4.pkl" ]; then
    echo "   ⚠️  Modèle LR manquant — entraînement..."
    python -m src.models.train_model_v4
    echo "   ✅ LR v4 entraîné"
else
    echo "   ✅ LR v4 présent"
fi

if [ ! -f "models/feature_cols_v3.pkl" ]; then
    echo "   ⚠️  feature_cols_v3.pkl manquant — recréation..."
    python3 -c "
import pickle
from src.utils.db import get_database
import pandas as pd, math

def clean(v):
    return 0.0 if (isinstance(v, float) and math.isnan(v)) else v

def get_cumulative_stats(t_feats, team_name, before_year):
    hist = t_feats[(t_feats['team_name'] == team_name) & (t_feats['year'] < before_year)]
    if hist.empty:
        return {'cum_matches': 0, 'cum_wins': 0, 'cum_win_rate': 0.25,
                'cum_goals_for': 0, 'cum_goals_against': 0, 'cum_goal_diff': 0.0,
                'cum_appearances': 0, 'recent_win_rate': 0.25, 'recent_goal_diff': 0.0}
    total_m = hist['matches'].sum(); total_w = hist['wins'].sum()
    total_gf = hist['goals_for'].sum(); total_ga = hist['goals_against'].sum()
    recent = hist.tail(3)
    return {'cum_matches': int(total_m), 'cum_wins': int(total_w),
            'cum_win_rate': total_w/total_m if total_m > 0 else 0.25,
            'cum_goals_for': int(total_gf), 'cum_goals_against': int(total_ga),
            'cum_goal_diff': (total_gf-total_ga)/total_m if total_m > 0 else 0.0,
            'cum_appearances': len(hist),
            'recent_win_rate': recent['win_rate'].mean() if len(recent) > 0 else 0.25,
            'recent_goal_diff': recent['goal_diff'].mean() if len(recent) > 0 else 0.0}

db = get_database()
matches = pd.concat([
    pd.DataFrame(list(db['matches'].find({}, {'_id': 0}))),
    pd.DataFrame(list(db['wc2022_matches_clean'].find({}, {'_id': 0})))
], ignore_index=True)
t_feats = pd.DataFrame(list(db['tournament_features'].find({}, {'_id': 0})))
t_feats['year'] = t_feats['tournament_id'].str.extract(r'(\d{4})').astype(int)
rows = []
for _, m in matches.iterrows():
    home = m['home_team_name']; away = m['away_team_name']
    year = int(str(m.get('tournament_id', 'WC-1930')).split('-')[1])
    hs = get_cumulative_stats(t_feats, home, year)
    as_ = get_cumulative_stats(t_feats, away, year)
    row = {}
    for k in hs:
        row[f'home_{k}'] = clean(hs[k]); row[f'away_{k}'] = clean(as_[k])
        row[f'diff_{k}'] = row[f'home_{k}'] - row[f'away_{k}']
    rows.append(row)
cols = list(pd.DataFrame(rows).columns)
with open('models/feature_cols_v3.pkl', 'wb') as f:
    pickle.dump(cols, f)
print(f'   ✅ feature_cols_v3.pkl recréé ({len(cols)} colonnes)')
"
else
    echo "   ✅ feature_cols_v3.pkl présent"
fi

# ── 11. Lancer la simulation ──────────────────────────────────────
echo ""
echo "1️⃣1️⃣  Lancement de la simulation CdM 2026..."
echo ""
python -m src.simulation.simulate_2026_v3

# ── 12. Dashboard HTML && Export Power BI ───────────────────────────────────────────
echo ""
echo "1️⃣2️⃣  Génération du dashboard HTML..."
mkdir -p exports
python -m src.simulation.generate_dashboard || { echo "❌ Dashboard échoué"; exit 1; }
echo "   ✅ Dashboard disponible dans reports/dashboard_2026.html"
echo ""
echo "1️⃣3️⃣  Export des données pour Power BI..."
mkdir -p exports
python -m src.data.export_for_powerbi || echo "⚠️  Export Power BI échoué"
echo "   ✅ Fichiers disponibles dans exports/"

echo ""
echo "=========================================================="
echo "✅ Pipeline terminé !"
echo ""
echo "Pour relancer uniquement la simulation :"
echo "   source venv/bin/activate"
echo "   python -m src.simulation.simulate_2026_v3"
echo ""
echo "Pour relancer uniquement le dashboard :"
echo "   source venv/bin/activate"
echo "   python -m src.simulation.generate_dashboard"
echo "=========================================================="
