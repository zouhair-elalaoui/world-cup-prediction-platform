import pickle, math, random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from src.utils.db import get_database

GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Australia", "Paraguay", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "Congo DR"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

FIFA_RANKING = {
    "Argentina": 1, "Spain": 2, "France": 3, "England": 4,
    "Portugal": 5, "Brazil": 6, "Morocco": 7, "Netherlands": 8,
    "Belgium": 9, "Germany": 10, "Croatia": 11,
    "Colombia": 13, "Mexico": 14, "Senegal": 15, "Uruguay": 16,
    "United States": 17, "Japan": 18, "Switzerland": 19, "Iran": 20,
    "Turkey": 22, "Ecuador": 23, "Austria": 24, "South Korea": 25,
    "Australia": 27, "Algeria": 28, "Egypt": 29, "Canada": 30,
    "Norway": 31, "Ivory Coast": 33, "Panama": 34, "Sweden": 38,
    "Czechia": 40, "Paraguay": 41, "Scotland": 42, "Tunisia": 45,
    "Congo DR": 46, "Uzbekistan": 50, "Qatar": 56, "Iraq": 57,
    "South Africa": 60, "Saudi Arabia": 61, "Jordan": 63,
    "Bosnia and Herzegovina": 64, "Cape Verde": 67, "Ghana": 73,
    "Curacao": 82, "Haiti": 83, "New Zealand": 85,
}

NAME_MAP = {
    "United States": "United States", "Turkey": "Turkey",
    "Ivory Coast": "Ivory Coast", "South Korea": "South Korea",
    "Congo DR": "Congo DR",
}

ROUND_NAMES = {
    32: "Round of 32", 16: "Round of 16",
    8: "Quarts de finale", 4: "Demi-finales", 2: "Finale",
}

def load_models():
    with open("models/xgb_model_v3.pkl",      "rb") as f: xgb_model = pickle.load(f)
    with open("models/feature_cols_v3.pkl",   "rb") as f: xgb_cols  = pickle.load(f)
    with open("models/lr_model_v4.pkl",       "rb") as f: lr_model  = pickle.load(f)
    with open("models/lr_feature_cols_v4.pkl","rb") as f: lr_cols   = pickle.load(f)
    return xgb_model, xgb_cols, lr_model, lr_cols

def fifa_to_elo(rank):
    return 2000 - (rank - 1) * (600 / 49)

def elo_probs(rank1, rank2, knockout=False):
    elo1 = fifa_to_elo(rank1)
    elo2 = fifa_to_elo(rank2)
    diff = elo1 - elo2
    exp1 = 1 / (1 + 10 ** (-diff / 400))
    exp2 = 1 - exp1
    if knockout:
        return np.array([exp2, 0.0, exp1])
    draw = max(0.10, 0.28 - abs(diff) / 2000)
    p1 = exp1 * (1 - draw)
    p2 = exp2 * (1 - draw)
    total = p1 + draw + p2
    return np.array([p2/total, draw/total, p1/total])

def get_team_stats(t_feats, team_name):
    mapped = NAME_MAP.get(team_name, team_name)
    hist   = t_feats[t_feats['team_name'] == mapped]
    if hist.empty:
        return {
            'cum_matches': 5, 'cum_wins': 1, 'cum_win_rate': 0.20,
            'cum_goals_for': 5, 'cum_goals_against': 8,
            'cum_goal_diff': -0.3, 'cum_appearances': 1,
            'recent_win_rate': 0.20, 'recent_goal_diff': -0.3,
        }
    total_m  = hist['matches'].sum()
    total_w  = hist['wins'].sum()
    total_gf = hist['goals_for'].sum()
    total_ga = hist['goals_against'].sum()
    recent   = hist.tail(3)
    return {
        'cum_matches':       int(total_m),
        'cum_wins':          int(total_w),
        'cum_win_rate':      total_w / total_m if total_m > 0 else 0.20,
        'cum_goals_for':     int(total_gf),
        'cum_goals_against': int(total_ga),
        'cum_goal_diff':     (total_gf - total_ga) / total_m if total_m > 0 else 0.0,
        'cum_appearances':   len(hist),
        'recent_win_rate':   recent['win_rate'].mean() if len(recent) > 0 else 0.20,
        'recent_goal_diff':  recent['goal_diff'].mean() if len(recent) > 0 else 0.0,
    }

def precompute_probs(xgb_model, xgb_cols, lr_model, lr_cols, t_feats):
    """
    Précalcule toutes les probabilités entre toutes les paires d'équipes.
    Retourne deux dicts :
      group_probs[(t1,t2)]   = [p_t2_win, p_draw, p_t1_win]
      ko_probs[(t1,t2)]      = p_t1_win (float)
    """
    print("\n⚡ Précalcul des probabilités pour toutes les paires...")
    all_teams = list(set(t for teams in GROUPS.values() for t in teams))

    # Préparer toutes les rows ML en batch
    rows_group = {}
    rows_ko    = {}

    for t1 in all_teams:
        for t2 in all_teams:
            if t1 == t2: continue
            s1 = get_team_stats(t_feats, t1)
            s2 = get_team_stats(t_feats, t2)
            row = {}
            for k in s1:
                row[f'home_{k}'] = s1[k]
                row[f'away_{k}'] = s2[k]
                row[f'diff_{k}'] = s1[k] - s2[k]
            rows_group[(t1,t2)] = row
            rows_ko[(t1,t2)]    = row

    # Batch XGBoost (groupes)
    pairs_list = [(t1,t2) for t1 in all_teams for t2 in all_teams if t1 != t2]
    X_group = pd.DataFrame([rows_group[p] for p in pairs_list])
    for c in xgb_cols:
        if c not in X_group.columns: X_group[c] = 0.0
    xgb_preds = xgb_model.predict_proba(X_group[xgb_cols])

    # Batch LR (knockout)
    X_ko = pd.DataFrame([rows_ko[p] for p in pairs_list])
    for c in lr_cols:
        if c not in X_ko.columns: X_ko[c] = 0.0
    lr_preds = lr_model.predict_proba(X_ko[lr_cols])

    # Construire les dicts de probabilités
    group_probs = {}
    ko_probs    = {}

    for i, (t1, t2) in enumerate(pairs_list):
        rank1 = FIFA_RANKING.get(t1, 45)
        rank2 = FIFA_RANKING.get(t2, 45)

        # ── Groupes ───────────────────────────────────────────────
        elo_p  = elo_probs(rank1, rank2, knockout=False)
        xgb_p  = xgb_preds[i]
        # Symétrie terrain neutre
        j = pairs_list.index((t2, t1))
        xgb_p2 = xgb_preds[j]
        xgb_p2_inv = np.array([xgb_p2[2], xgb_p2[1], xgb_p2[0]])
        xgb_sym = (xgb_p + xgb_p2_inv) / 2
        combined_g = 0.60 * elo_p + 0.40 * xgb_sym
        group_probs[(t1,t2)] = combined_g / combined_g.sum()

        # ── Knockout ──────────────────────────────────────────────
        elo_ko = elo_probs(rank1, rank2, knockout=True)
        lr_p   = lr_preds[i]
        lr_p2  = lr_preds[j]
        lr_p2_inv = np.array([lr_p2[2], lr_p2[1], lr_p2[0]])
        lr_sym = (lr_p + lr_p2_inv) / 2
        p1_ko  = lr_sym[2] + lr_sym[1] * 0.5
        p2_ko  = lr_sym[0] + lr_sym[1] * 0.5
        lr_ko  = np.array([p2_ko, 0.0, p1_ko]) / (p1_ko + p2_ko)
        combined_ko = 0.60 * elo_ko + 0.40 * lr_ko
        combined_ko = combined_ko / combined_ko.sum()
        ko_probs[(t1,t2)] = combined_ko[2]  # p_t1_win

    print(f"   ✅ {len(group_probs)} paires précalculées\n")
    return group_probs, ko_probs

def simulate_group_fast(group_probs, teams):
    standings = {t: {'pts': 0, 'gf': 0, 'ga': 0, 'gd': 0} for t in teams}
    matchups  = [(teams[i], teams[j]) for i in range(4) for j in range(i+1, 4)]

    for t1, t2 in matchups:
        probs   = group_probs[(t1, t2)]
        outcome = np.random.choice([0, 1, 2], p=probs)
        avg1    = max(2.5 - FIFA_RANKING.get(t1, 30) / 25, 0.6)
        avg2    = max(2.5 - FIFA_RANKING.get(t2, 30) / 25, 0.6)

        if outcome == 2:
            g1 = max(1, int(np.random.poisson(avg1)))
            g2 = max(0, int(np.random.poisson(avg2 * 0.5)))
            if g1 <= g2: g1 = g2 + 1
            standings[t1]['pts'] += 3
        elif outcome == 1:
            g1 = int(np.random.poisson(avg1 * 0.6)); g2 = g1
            standings[t1]['pts'] += 1; standings[t2]['pts'] += 1
        else:
            g2 = max(1, int(np.random.poisson(avg2)))
            g1 = max(0, int(np.random.poisson(avg1 * 0.5)))
            if g2 <= g1: g2 = g1 + 1
            standings[t2]['pts'] += 3

        standings[t1]['gf'] += g1; standings[t1]['ga'] += g2
        standings[t2]['gf'] += g2; standings[t2]['ga'] += g1
        standings[t1]['gd'] += g1 - g2
        standings[t2]['gd'] += g2 - g1

    return sorted(standings.items(),
                  key=lambda x: (x[1]['pts'], x[1]['gd'], x[1]['gf']),
                  reverse=True)

def simulate_all_groups_fast(group_probs, verbose=False):
    group_results = {}
    thirds = []

    if verbose:
        print("\n" + "="*60)
        print("🏟️  PHASE DE GROUPES (12 groupes de 4)")
        print("="*60)

    for g, teams in GROUPS.items():
        ranked = simulate_group_fast(group_probs, teams)
        if verbose:
            print(f"\n  Groupe {g}:")
            for i, (team, s) in enumerate(ranked):
                marker = "✅" if i < 2 else ("🔸" if i == 2 else "❌")
                print(f"    {marker} {i+1}. {team:<25} {s['pts']}pts  GD:{s['gd']:+d}")
        group_results[f"1{g}"] = ranked[0][0]
        group_results[f"2{g}"] = ranked[1][0]
        thirds.append((ranked[2][0], ranked[2][1], g))

    thirds_sorted = sorted(thirds,
                           key=lambda x: (x[1]['pts'], x[1]['gd'], x[1]['gf']),
                           reverse=True)
    best_thirds = thirds_sorted[:8]

    if verbose:
        print(f"\n  🔸 8 meilleurs 3èmes :")
        for t, s, g in best_thirds:
            print(f"    • {t:<25} (Groupe {g}) {s['pts']}pts  GD:{s['gd']:+d}")
        print(f"\n  ✅ 32 équipes qualifiées pour le Round of 32")

    return group_results, [t[0] for t in best_thirds]

def build_bracket(group_results, best_thirds):
    pairs = [
        (group_results["1A"], group_results["2B"]),
        (group_results["1C"], group_results["2D"]),
        (group_results["1E"], group_results["2F"]),
        (group_results["1G"], group_results["2H"]),
        (group_results["1I"], group_results["2J"]),
        (group_results["1K"], group_results["2L"]),
        (group_results["1B"], group_results["2A"]),
        (group_results["1D"], group_results["2C"]),
        (group_results["1F"], group_results["2E"]),
        (group_results["1H"], group_results["2G"]),
        (group_results["1J"], group_results["2I"]),
        (group_results["1L"], group_results["2K"]),
    ]
    t = best_thirds[:]
    random.shuffle(t)
    for i in range(0, 8, 2):
        pairs.append((t[i], t[i+1]))
    return pairs

def simulate_knockout_fast(ko_probs, pairs, verbose=False):
    if verbose:
        print("\n" + "="*60)
        print("⚔️  PHASE ÉLIMINATOIRE")
        print("="*60)

    current     = pairs[:]
    semi_losers = []

    while True:
        n_teams   = len(current) * 2
        rnd_name  = ROUND_NAMES.get(n_teams, f"Tour à {n_teams}")
        if verbose: print(f"\n  🏆 {rnd_name} ({n_teams} équipes)")

        winners = []; losers = []
        for t1, t2 in current:
            p1     = ko_probs[(t1, t2)]
            winner = t1 if random.random() < p1 else t2
            loser  = t2 if winner == t1 else t1
            winners.append(winner); losers.append(loser)
            if verbose:
                print(f"    {t1:<25} ({p1*100:.0f}%) vs {t2:<25} ({(1-p1)*100:.0f}%) → 🏅 {winner}")

        if n_teams == 4:
            semi_losers = losers[:]

        if n_teams == 2:
            champion = winners[0]; finalist = losers[0]
            if verbose: print(f"\n  🥉 Match pour la 3e place")
            p3     = ko_probs[(semi_losers[0], semi_losers[1])]
            third  = semi_losers[0] if random.random() < p3 else semi_losers[1]
            fourth = semi_losers[1] if third == semi_losers[0] else semi_losers[0]
            if verbose:
                print(f"    {semi_losers[0]:<25} ({p3*100:.0f}%) vs {semi_losers[1]:<25} ({(1-p3)*100:.0f}%) → 🏅 {third}")
            return champion, finalist, third, fourth

        current = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]

def save_to_mongodb(db, win_counts, second_counts, third_counts,
                    fourth_counts, semi_counts, final_counts,
                    group_results, best_thirds, group_pos_counts, N):
    print("\n💾 Sauvegarde dans MongoDB...")
    all_teams = set(win_counts) | set(second_counts) | set(third_counts)
    proba_docs = []
    for team in all_teams:
        proba_docs.append({
            "team":           team,
            "fifa_rank":      FIFA_RANKING.get(team, 99),
            "proba_champion": round(win_counts[team]    / N * 100, 1),
            "proba_finalist": round(second_counts[team] / N * 100, 1),
            "proba_third":    round(third_counts[team]  / N * 100, 1),
            "proba_fourth":   round(fourth_counts[team] / N * 100, 1),
            "proba_semis":    round(semi_counts[team]   / N * 100, 1),
            "proba_final":    round(final_counts[team]  / N * 100, 1),
            "simulations":    N,
            "date":           datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
    db["simulation_probabilities"].drop()
    db["simulation_probabilities"].insert_many(proba_docs)
    print(f"   ✅ simulation_probabilities : {len(proba_docs)} équipes")

    group_docs = []
    for key, team in group_results.items():
        group_docs.append({"group": key[1], "position": int(key[0]), "team": team, "qualified": True})
    for team in best_thirds:
        group_docs.append({"group": "best_third", "position": 3, "team": team, "qualified": True})
    db["simulation_groups"].drop()
    db["simulation_groups"].insert_many(group_docs)

    best_c = max(win_counts, key=win_counts.get)
    sc_f   = {k:v for k,v in second_counts.items() if k != best_c}
    best_2 = max(sc_f, key=sc_f.get) if sc_f else "-"
    t3_f   = {k:v for k,v in third_counts.items()  if k not in [best_c, best_2]}
    t4_f   = {k:v for k,v in fourth_counts.items() if k not in [best_c, best_2]}
    best_3 = max(t3_f, key=t3_f.get) if t3_f else "-"
    best_4 = max(t4_f, key=t4_f.get) if t4_f else "-"

    prediction = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "simulations": N,
        "champion": {"team": best_c,  "proba": round(win_counts[best_c]/N*100,  1)},
        "finalist": {"team": best_2,  "proba": round(second_counts[best_2]/N*100, 1)},
        "third":    {"team": best_3,  "proba": round(third_counts[best_3]/N*100,  1)},
        "fourth":   {"team": best_4,  "proba": round(fourth_counts[best_4]/N*100, 1)},
    }
    db["simulation_prediction"].drop()
    db["simulation_prediction"].insert_one(prediction)

    group_proba_docs = []
    for g, teams in GROUPS.items():
        for team in teams:
            c = group_pos_counts[(g, team)]
            group_proba_docs.append({
                "group": g, "team": team,
                "fifa_rank":       FIFA_RANKING.get(team, 99),
                "proba_1st":       round(c[1] / N * 100, 1),
                "proba_2nd":       round(c[2] / N * 100, 1),
                "proba_3rd":       round(c[3] / N * 100, 1),
                "proba_4th":       round(c[4] / N * 100, 1),
                "proba_qualified": round((c[1]+c[2]) / N * 100, 1),
                "simulations": N,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
    db["simulation_group_probas"].drop()
    db["simulation_group_probas"].insert_many(group_proba_docs)
    print(f"   ✅ Toutes les collections sauvegardées !")
    print(f"\n   🥇 {best_c} ({win_counts[best_c]/N*100:.1f}%)")
    print(f"   🥈 {best_2} ({second_counts[best_2]/N*100:.1f}%)")
    print(f"   🥉 {best_3} ({third_counts[best_3]/N*100:.1f}%)")
    print(f"   4e {best_4} ({fourth_counts[best_4]/N*100:.1f}%)")

def main():
    N = 10000
    print("🌍 SIMULATION — COUPE DU MONDE 2026 (v3 — FAST)")
    print("="*60)
    print("⚡ Optimisation : probabilités précalculées en batch")
    print(f"🎲 Simulations : {N:,}")

    db          = get_database()
    xgb_model, xgb_cols, lr_model, lr_cols = load_models()
    t_feats     = pd.DataFrame(list(db['tournament_features'].find({}, {'_id': 0})))
    t_feats['year'] = t_feats['tournament_id'].str.extract(r'(\d{4})').astype(int)

    # ── Précalcul des probs ────────────────────────────────────────
    group_probs, ko_probs = precompute_probs(
        xgb_model, xgb_cols, lr_model, lr_cols, t_feats
    )

    # ── Simulation déterministe 1 fois ────────────────────────────
    group_results, best_thirds = simulate_all_groups_fast(group_probs, verbose=True)
    pairs = build_bracket(group_results, best_thirds)
    champion, finalist, third, fourth = simulate_knockout_fast(ko_probs, pairs, verbose=True)

    print("\n" + "="*60)
    print("🏆 RÉSULTAT SIMULATION DÉTERMINISTE")
    print("="*60)
    print(f"  🥇 Champion   : {champion}")
    print(f"  🥈 2ème place : {finalist}")
    print(f"  🥉 3ème place : {third}")
    print(f"     4ème place : {fourth}")

    # ── Monte Carlo ────────────────────────────────────────────────
    print(f"\n\n🎲 Monte Carlo — {N:,} simulations (version rapide)...")
    win_counts    = defaultdict(int)
    second_counts = defaultdict(int)
    third_counts  = defaultdict(int)
    fourth_counts = defaultdict(int)
    semi_counts   = defaultdict(int)
    final_counts  = defaultdict(int)
    group_pos_counts = defaultdict(lambda: defaultdict(int))

    for sim in range(N):
        # Simuler groupes
        gr_results = {}
        thirds_mc  = []
        for g, teams in GROUPS.items():
            ranked = simulate_group_fast(group_probs, teams)
            for pos, (team, _) in enumerate(ranked, start=1):
                group_pos_counts[(g, team)][pos] += 1
            gr_results[f"1{g}"] = ranked[0][0]
            gr_results[f"2{g}"] = ranked[1][0]
            thirds_mc.append((ranked[2][0], ranked[2][1], g))

        # 8 meilleurs 3èmes
        thirds_sorted = sorted(thirds_mc,
                               key=lambda x: (x[1]['pts'], x[1]['gd'], x[1]['gf']),
                               reverse=True)
        bt = [t[0] for t in thirds_sorted[:8]]

        # Bracket
        p = build_bracket(gr_results, bt)
        current = p[:]
        semi_l  = []

        while True:
            n_t     = len(current) * 2
            winners = []; losers = []
            for t1, t2 in current:
                p1     = ko_probs[(t1, t2)]
                winner = t1 if random.random() < p1 else t2
                loser  = t2 if winner == t1 else t1
                winners.append(winner); losers.append(loser)
                if n_t == 4:
                    semi_counts[t1] += 1; semi_counts[t2] += 1
                if n_t == 2:
                    final_counts[t1] += 1; final_counts[t2] += 1

            if n_t == 4: semi_l = losers[:]

            if n_t == 2:
                win_counts[winners[0]]    += 1
                second_counts[losers[0]]  += 1
                p3    = ko_probs[(semi_l[0], semi_l[1])]
                t3    = semi_l[0] if random.random() < p3 else semi_l[1]
                t4    = semi_l[1] if t3 == semi_l[0] else semi_l[0]
                third_counts[t3]  += 1
                fourth_counts[t4] += 1
                break

            current = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]

        if (sim+1) % 2000 == 0:
            print(f"   ... {sim+1:,}/{N:,} simulations")

    # ── Affichage ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"🏆 PROBABILITÉS ({N:,} simulations)")
    print("="*60)
    print(f"  {'Équipe':<25} {'🥇':>7} {'🥈':>7} {'🥉':>7} {'4e':>7} {'Demies':>8}")
    print("  " + "-"*60)
    all_teams = set(win_counts) | set(second_counts) | set(third_counts)
    for team in sorted(all_teams, key=lambda t: win_counts[t], reverse=True)[:15]:
        bar = "█" * (win_counts[team] // (N//100))
        print(f"  {team:<25} {win_counts[team]/N*100:>6.1f}%"
              f" {second_counts[team]/N*100:>6.1f}%"
              f" {third_counts[team]/N*100:>6.1f}%"
              f" {fourth_counts[team]/N*100:>6.1f}%"
              f" {semi_counts[team]/N*100:>7.1f}%"
              f"  {bar}")

    best_c = max(win_counts, key=win_counts.get)
    sc_f   = {k:v for k,v in second_counts.items() if k != best_c}
    best_2 = max(sc_f, key=sc_f.get) if sc_f else "-"
    t3_f   = {k:v for k,v in third_counts.items()  if k not in [best_c, best_2]}
    t4_f   = {k:v for k,v in fourth_counts.items() if k not in [best_c, best_2]}
    best_3 = max(t3_f, key=t3_f.get) if t3_f else "-"
    best_4 = max(t4_f, key=t4_f.get) if t4_f else "-"

    print("\n" + "="*60)
    print("🎯 PRÉDICTION FINALE")
    print("="*60)
    print(f"  🥇 Champion   : {best_c:<25} ({win_counts[best_c]/N*100:.1f}%)")
    print(f"  🥈 2ème place : {best_2:<25} ({second_counts[best_2]/N*100:.1f}%)")
    print(f"  🥉 3ème place : {best_3:<25} ({third_counts[best_3]/N*100:.1f}%)")
    print(f"     4ème place : {best_4:<25} ({fourth_counts[best_4]/N*100:.1f}%)")

    # ── Probabilités par groupe ────────────────────────────────────
    print("\n\n" + "="*60)
    print("📊 PROBABILITÉS PAR GROUPE")
    print("="*60)
    for g, teams in GROUPS.items():
        print(f"\n  Groupe {g}:")
        print(f"  {'Équipe':<25} {'1er':>7} {'2ème':>7} {'3ème':>7} {'4ème':>7}  Qualif")
        print("  " + "-"*60)
        sorted_teams = sorted(teams, key=lambda t: group_pos_counts[(g, t)][1], reverse=True)
        for team in sorted_teams:
            c      = group_pos_counts[(g, team)]
            p1     = c[1] / N * 100
            p2     = c[2] / N * 100
            p3     = c[3] / N * 100
            p4     = c[4] / N * 100
            qualif = p1 + p2
            bar    = "█" * int(qualif / 5)
            print(f"  {team:<25} {p1:>6.1f}% {p2:>6.1f}% {p3:>6.1f}% {p4:>6.1f}%  {bar} {qualif:.0f}%")

    # ── MongoDB ───────────────────────────────────────────────────
    save_to_mongodb(db, win_counts, second_counts, third_counts,
                    fourth_counts, semi_counts, final_counts,
                    group_results, best_thirds, group_pos_counts, N)

if __name__ == "__main__":
    main()
