import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from itertools import combinations
from scipy.stats import poisson

DATA_DIR = 'data'
np.random.seed(42)

print("Loading model and data...")
with open(f'{DATA_DIR}/model.pkl','rb') as f:
    saved = pickle.load(f)
clf     = saved['classifier']
pr_home = saved['poisson_home']
pr_away = saved['poisson_away']

team_feat = pd.read_csv(f'{DATA_DIR}/team_features.csv')
feat_idx  = team_feat.set_index('team')

GROUPS = {
    'A':['Mexico','South Africa','South Korea','Czech Republic'],
    'B':['Canada','Bosnia and Herzegovina','Qatar','Switzerland'],
    'C':['Brazil','Morocco','Haiti','Scotland'],
    'D':['United States','Paraguay','Australia','Turkey'],
    'E':['Germany','Cote dIvoire','Thailand','Chile'],
    'F':['Spain','Egypt','Tunisia','Costa Rica'],
    'G':['England','Croatia','Ghana','Panama'],
    'H':['France','Senegal','Iraq','Norway'],
    'I':['Argentina','Algeria','Austria','Jordan'],
    'J':['Portugal','DR Congo','Uzbekistan','Colombia'],
    'K':['Netherlands','Iran','New Zealand','Japan'],
    'L':['Belgium','Uruguay','Sweden','Saudi Arabia'],
}
ALL_TEAMS = [t for g in GROUPS.values() for t in g]
avg_gf = team_feat['avg_gf'].mean()
avg_ga = team_feat['avg_ga'].mean()

def gs(team, col, default=0.0):
    try:
        v = feat_idx.loc[team, col]
        return float(v) if not pd.isna(v) else default
    except: return default

def match_probs(home, away, knockout=False):
    hr = gs(home,'rank_score',0.4)
    ar = gs(away,'rank_score',0.4)
    feat = np.array([[
        hr-ar,
        gs(home,'win_rate',0.33)-gs(away,'win_rate',0.33),
        gs(home,'avg_gf',1.2)-gs(away,'avg_gf',1.2),
        gs(home,'avg_ga',1.2)-gs(away,'avg_ga',1.2),
        gs(home,'avg_gd',0.0)-gs(away,'avg_gd',0.0),
        hr, ar, int(knockout),
        gs(home,'avg_xg_scored',1.0)-gs(away,'avg_xg_scored',1.0)
    ]])
    p   = clf.predict_proba(feat)[0]
    ph,pd_,pa = p[0],p[1],p[2]

    # FIFA ranking correction — 50/50 blend
    rank_diff = hr - ar
    rph = 1/(1+np.exp(-4*rank_diff))
    rpa = 1 - rph
    rpd = 0.25
    norm = rph+rpd+rpa
    rph/=norm; rpa/=norm; rpd/=norm
    ph  = 0.3*ph  + 0.7*rph
    pa  = 0.3*pa  + 0.7*rpa
    pd_ = 0.3*pd_ + 0.7*rpd

    if knockout:
        ph += pd_/2; pa += pd_/2; pd_ = 0
    return float(ph), float(pd_), float(pa)

def predict_score(home, away):
    h_atk = gs(home,'avg_gf',avg_gf) / avg_gf
    h_def = gs(home,'avg_ga',avg_ga) / avg_ga
    a_atk = gs(away,'avg_gf',avg_gf) / avg_gf
    a_def = gs(away,'avg_ga',avg_ga) / avg_ga
    lh = max(0.6, h_atk * a_def * avg_gf * 1.1)
    la = max(0.4, a_atk * h_def * avg_gf * 0.9)
    ph,pd_,pa = match_probs(home,away)
    # Adjust lambdas toward expected direction
    # Scale lambdas based on win probability dominance
    prob_diff = ph - pa
    if prob_diff > 0.3:    lh = max(lh, la + 1.2); la = max(0.3, la - 0.3)
    elif prob_diff > 0.15: lh = max(lh, la + 0.7); la = max(0.4, la - 0.2)
    elif prob_diff > 0.05: lh = max(lh, la + 0.3)
    elif prob_diff < -0.3:  la = max(la, lh + 1.2); lh = max(0.3, lh - 0.3)
    elif prob_diff < -0.15: la = max(la, lh + 0.7); lh = max(0.4, lh - 0.2)
    elif prob_diff < -0.05: la = max(la, lh + 0.3)
    best_p, best_s = 0, (1,0)
    for hg in range(9):
        for ag in range(9):
            p = poisson.pmf(hg,lh)*poisson.pmf(ag,la)
            if p > best_p: best_p=p; best_s=(hg,ag)
    return best_s

def simulate_match(home, away, knockout=False):
    ph,pd_,pa = match_probs(home,away,knockout)
    outcome = np.random.choice(['home','draw','away'], p=[ph,pd_,pa])
    h_atk = gs(home,'avg_gf',avg_gf)/avg_gf
    h_def = gs(home,'avg_ga',avg_ga)/avg_ga
    a_atk = gs(away,'avg_gf',avg_gf)/avg_gf
    a_def = gs(away,'avg_ga',avg_ga)/avg_ga
    lh = max(0.6, h_atk*a_def*avg_gf*1.1)
    la = max(0.4, a_atk*h_def*avg_gf*0.9)
    if outcome=='home':
        while True:
            hg,ag = np.random.poisson(lh), np.random.poisson(la)
            if hg > ag: break
        winner = home
    elif outcome=='away':
        while True:
            hg,ag = np.random.poisson(lh), np.random.poisson(la)
            if ag > hg: break
        winner = away
    else:
        if knockout:
            hg = ag = int(np.random.poisson((lh+la)/2))
            winner = home if np.random.random() < ph/(ph+pa) else away
        else:
            while True:
                hg,ag = np.random.poisson(lh), np.random.poisson(la)
                if hg == ag: break
            winner = None
    return hg, ag, winner

def simulate_group(teams):
    st = {t:{'pts':0,'gf':0,'ga':0,'gd':0} for t in teams}
    for home,away in combinations(teams,2):
        hg,ag,_ = simulate_match(home,away)
        st[home]['gf']+=hg; st[home]['ga']+=ag; st[home]['gd']+=hg-ag
        st[away]['gf']+=ag; st[away]['ga']+=hg; st[away]['gd']+=ag-hg
        if hg>ag:   st[home]['pts']+=3
        elif hg==ag: st[home]['pts']+=1; st[away]['pts']+=1
        else:        st[away]['pts']+=3
    df = pd.DataFrame(st).T.reset_index().rename(columns={'index':'team'})
    return df.sort_values(['pts','gd','gf'],ascending=False).reset_index(drop=True)

# ── 10,000 SIMULATIONS ───────────────────────────────────────────
N = 10000
win_counts   = defaultdict(int)
round_counts = defaultdict(lambda: defaultdict(int))
goal_tallies = defaultdict(list)

print(f"Running {N:,} simulations...")
for sim in range(N):
    group_winners = []; group_runners = []; third_place = []

    for grp, teams in GROUPS.items():
        standing = simulate_group(teams)
        group_winners.append(standing.iloc[0]['team'])
        group_runners.append(standing.iloc[1]['team'])
        third_place.append({
            'team': standing.iloc[2]['team'],
            'pts':  float(standing.iloc[2]['pts']),
            'gd':   float(standing.iloc[2]['gd']),
            'gf':   float(standing.iloc[2]['gf']),
        })

    # Best 8 third-place teams
    third_df  = pd.DataFrame(third_place).sort_values(['pts','gd','gf'],ascending=False).head(8)
    best8     = third_df['team'].tolist()
    r32       = group_winners + group_runners + best8
    assert len(r32) == 32

    def run_round(teams, rnd):
        shuffled = teams[:]
        np.random.shuffle(shuffled)
        winners = []
        for i in range(0, len(shuffled), 2):
            h,a = shuffled[i], shuffled[i+1]
            hg,ag,w = simulate_match(h,a,knockout=True)
            if w is None: w = h if np.random.random()<0.5 else a
            round_counts[w][rnd] += 1
            goal_tallies[w].append(hg if w==h else ag)
            winners.append(w)
        return winners

    r16   = run_round(r32, 'r16')
    qf    = run_round(r16, 'qf')
    sf    = run_round(qf,  'sf')
    third_game = run_round([t for t in qf if t not in sf], '3rd')
    final = run_round(sf,  'final')
    champion = final[0]
    win_counts[champion] += 1

    if sim % 2000 == 0 and sim > 0:
        print(f"  {sim:,} done...")

print(f"All {N:,} simulations done!")

# ── RESULTS ──────────────────────────────────────────────────────
results = []
for team in ALL_TEAMS:
    results.append({
        'team':     team,
        'group':    next(g for g,ts in GROUPS.items() if team in ts),
        'fifa_rank':int(gs(team,'fifa_rank',99)),
        'p_r16':    round_counts[team]['r16']   / N,
        'p_qf':     round_counts[team]['qf']    / N,
        'p_sf':     round_counts[team]['sf']    / N,
        'p_final':  round_counts[team]['final'] / N,
        'p_winner': win_counts[team]            / N,
        'avg_ko_goals': np.mean(goal_tallies[team]) if goal_tallies[team] else 0,
    })

results_df = pd.DataFrame(results).sort_values('p_winner',ascending=False).reset_index(drop=True)
results_df['sim_rank'] = results_df.index + 1

print()
print("TOP 16 WIN PROBABILITIES:")
print("="*60)
for _,row in results_df.head(16).iterrows():
    bar = '#'*int(row['p_winner']*300)
    print(f"  {int(row['sim_rank']):>2}. {row['team']:<26} {row['p_winner']:>5.1%}  {bar}")

# ── BRACKET PREDICTIONS ──────────────────────────────────────────
print("\nGenerating predicted bracket...")
bracket_matches = []
np.random.seed(0)

det_winners=[]; det_runners=[]; det_thirds=[]
for grp, teams in GROUPS.items():
    st = {t:{'pts':0,'gf':0,'ga':0,'gd':0} for t in teams}
    for home,away in combinations(teams,2):
        ph,pd_,pa = match_probs(home,away)
        sc = predict_score(home,away)
        hg,ag = sc
        bracket_matches.append({
            'round':f'Group {grp}','home':home,'away':away,
            'pred_home_score':hg,'pred_away_score':ag,
            'home_win_prob':round(ph,3),'draw_prob':round(pd_,3),'away_win_prob':round(pa,3),
            'predicted_winner':home if ph>pa else (away if pa>ph else 'Draw')
        })
        st[home]['gf']+=hg; st[home]['ga']+=ag; st[home]['gd']+=hg-ag
        st[away]['gf']+=ag; st[away]['ga']+=hg; st[away]['gd']+=ag-hg
        if hg>ag: st[home]['pts']+=3
        elif hg==ag: st[home]['pts']+=1; st[away]['pts']+=1
        else: st[away]['pts']+=3
    ranking = sorted(st.keys(), key=lambda t:(-st[t]['pts'],-st[t]['gd'],-st[t]['gf']))
    det_winners.append(ranking[0])
    det_runners.append(ranking[1])
    det_thirds.append({'team':ranking[2],'pts':st[ranking[2]]['pts'],
                       'gd':st[ranking[2]]['gd'],'gf':st[ranking[2]]['gf']})

thirds_sorted = sorted(det_thirds, key=lambda x:(-x['pts'],-x['gd'],-x['gf']))
best8 = [t['team'] for t in thirds_sorted[:8]]
r32 = det_winners + det_runners + best8
np.random.shuffle(r32)

def det_round(teams, rnd):
    winners = []
    for i in range(0, len(teams), 2):
        h,a = teams[i], teams[i+1]
        ph,pd_,pa = match_probs(h,a,knockout=True)
        sc = predict_score(h,a)
        w = h if ph >= pa else a
        bracket_matches.append({
            'round':rnd,'home':h,'away':a,
            'pred_home_score':sc[0],'pred_away_score':sc[1],
            'home_win_prob':round(ph,3),'draw_prob':round(pd_,3),'away_win_prob':round(pa,3),
            'predicted_winner':w
        })
        winners.append(w)
    return winners

r16 = det_round(r32,'R32')
qf  = det_round(r16,'R16')
sf  = det_round(qf,'QF')
det_round([t for t in qf if t not in sf],'3rd Place')
final = det_round(sf,'Final')
champion = final[0]
print(f"Predicted champion: {champion}")

bracket_df = pd.DataFrame(bracket_matches)

# ── FIXTURES ─────────────────────────────────────────────────────
fixtures = []
for grp,teams in GROUPS.items():
    for home,away in combinations(teams,2):
        sc = predict_score(home,away)
        ph,pd_,pa = match_probs(home,away)
        fixtures.append({
            'group':grp,'home':home,'away':away,
            'pred_home':sc[0],'pred_away':sc[1],
            'home_win_prob':round(ph,3),'draw_prob':round(pd_,3),'away_win_prob':round(pa,3),
        })
fixtures_df = pd.DataFrame(fixtures)

# ── SAVE ALL ──────────────────────────────────────────────────────
results_df.to_csv(f'{DATA_DIR}/simulation_results.csv', index=False)
bracket_df.to_csv(f'{DATA_DIR}/bracket_predictions.csv', index=False)
fixtures_df.to_csv(f'{DATA_DIR}/fixtures_2026.csv', index=False)
print(f"\nSaved all files!")
print(f"Champion: {champion}")
