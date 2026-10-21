"""
Script d'exploration du dataset Kaggle (World Cup Database 1930-2018).

Objectif : afficher pour chaque fichier CSV clé :
- le nombre de lignes / colonnes
- les noms et types de colonnes
- un aperçu des premières lignes
- le nombre de valeurs manquantes par colonne

Usage :
    python -m src.data.explore_kaggle_data
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import DATA_RAW_DIR

# Dossier contenant les CSV Kaggle
KAGGLE_DIR = DATA_RAW_DIR / "historical_1930_2018"

# Fichiers jugés essentiels pour le projet (prédiction + simulation)
KEY_FILES = [
    "tournaments.csv",
    "host_countries.csv",
    "confederations.csv",
    "teams.csv",
    "team_appearances.csv",
    "qualified_teams.csv",
    "groups.csv",
    "group_standings.csv",
    "tournament_standings.csv",
    "matches.csv",
    "goals.csv",
]

# Fichiers secondaires (utiles plus tard, juste un aperçu rapide)
SECONDARY_FILES = [
    "squads.csv",
    "players.csv",
    "player_appearances.csv",
    "penalty_kicks.csv",
    "bookings.csv",
    "substitutions.csv",
    "managers.csv",
    "manager_appearances.csv",
    "manager_appointments.csv",
    "referees.csv",
    "referee_appearances.csv",
    "referee_appointments.csv",
    "awards.csv",
    "award_winners.csv",
    "stadiums.csv",
    "tournament_stages.csv",
]


def explore_file(filepath: Path, n_preview: int = 5, detailed: bool = True):
    """Affiche un résumé détaillé d'un fichier CSV."""
    if not filepath.exists():
        print(f"❌ Fichier introuvable : {filepath.name}")
        return None

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ Erreur de lecture pour {filepath.name} : {e}")
        return None

    print(f"\n{'=' * 70}")
    print(f"📄 {filepath.name}")
    print(f"{'=' * 70}")
    print(f"Dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    if detailed:
        print(f"\nColonnes et types :")
        for col in df.columns:
            n_missing = df[col].isna().sum()
            pct_missing = (n_missing / len(df) * 100) if len(df) > 0 else 0
            print(f"  - {col:<30} {str(df[col].dtype):<10} "
                  f"({n_missing} manquants, {pct_missing:.1f}%)")

        print(f"\nAperçu ({n_preview} premières lignes) :")
        print(df.head(n_preview).to_string())
    else:
        print(f"Colonnes : {list(df.columns)}")

    return df


def main():
    print("🔍 EXPLORATION DU DATASET KAGGLE — WORLD CUP DATABASE")
    print(f"📂 Dossier source : {KAGGLE_DIR}")

    if not KAGGLE_DIR.exists():
        print(f"❌ Le dossier {KAGGLE_DIR} n'existe pas.")
        return

    all_csvs = sorted(KAGGLE_DIR.glob("*.csv"))
    print(f"📊 {len(all_csvs)} fichiers CSV trouvés dans le dossier.\n")

    print("\n" + "#" * 70)
    print("# FICHIERS CLÉS (exploration détaillée)")
    print("#" * 70)
    for fname in KEY_FILES:
        explore_file(KAGGLE_DIR / fname, n_preview=5, detailed=True)

    print("\n\n" + "#" * 70)
    print("# FICHIERS SECONDAIRES (aperçu rapide des colonnes uniquement)")
    print("#" * 70)
    for fname in SECONDARY_FILES:
        explore_file(KAGGLE_DIR / fname, detailed=False)

    # Vérifie s'il y a des fichiers présents dans le dossier mais non listés ci-dessus
    known_files = set(KEY_FILES) | set(SECONDARY_FILES)
    found_files = {f.name for f in all_csvs}
    unknown = found_files - known_files
    if unknown:
        print(f"\n⚠️  Fichiers présents mais non répertoriés dans ce script : {unknown}")

    print("\n✅ Exploration terminée.")


if __name__ == "__main__":
    main()
