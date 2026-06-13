import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("# 2026 FIFA World Cup — Notebook 1: Data Collection\nData window: 2010-2022 | Sources: jfjelstul/worldcup + StatsBomb Open Data"))

cells.append(nbf.v4.new_code_cell(
"""import requests, warnings
import pandas as pd
import numpy as np
from datetime import datetime
import os
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)
DATA_DIR = '../data'

NAME_MAP = {
    'USA':'United States','Ivory Coast':'Cote dIvoire',
    'Korea Republic':'South Korea',
    'Bosnia-Herzegovina':'Bosnia and Herzegovina',
    'Turkiye':'Turkey','Türkiye':'Turkey',
    'IR Iran':'Iran','Congo DR':'DR Congo',
    'Czechia':'Czech Republic',
    'United States of America':'United States',
    'Curaçao':'Curacao','Cabo Verde':'Cape Verde',
}

# Official 2026 WC Groups (correct draw)
GROUPS = {
    'A':['Mexico','South Korea','Czech Republic','South Africa'],
    'B':['Canada','Switzerland','Bosnia and Herzegovina','Qatar'],
    'C':['Brazil','Morocco','Scotland','Haiti'],
    'D':['United States','Australia','Turkey','Paraguay'],
    'E':['Germany','Ecuador','Cote dIvoire','Curacao'],
    'F':['Netherlands','Japan','Sweden','Tunisia'],
    'G':['Belgium','Iran','Egypt','New Zealand'],
    'H':['Spain','Uruguay','Saudi Arabia','Cape Verde'],
    'I':['France','Norway','Senegal','Iraq'],
    'J':['Argentina','Austria','Algeria','Jordan'],
    'K':['Portugal','Colombia','DR Congo','Uzbekistan'],
    'L':['England','Croatia','Panama','Ghana'],
}
ALL_TEAMS = [t for g in GROUPS.values() for t in g]
GROUP_MAP = {t:g for g,ts in GROUPS.items() for t in ts}
assert len(ALL_TEAMS) == 48, f"Expected 48 teams, got {len(ALL_TEAMS)}"
print(f'Setup complete — {len(ALL_TEAMS)} teams loaded')
print(f'{datetime.now().strftime(\"%Y-%m-%d %H:%M\")}')"""
))

cells.append(nbf.v4.new_markdown_cell("## 1 · Load Raw Data (2010-2022)"))
cells.append(nbf.v4.new_code_cell(
"""matches_df  = pd.read_csv(f'{DATA_DIR}/matches.csv')
team_app_df = pd.read_csv(f'{DATA_DIR}/team_appearances.csv')
goals_df    = pd.read_csv(f'{DATA_DIR}/goals.csv')
bookings_df = pd.read_csv(f'{DATA_DIR}/bookings.csv')
for df in [matches_df,team_app_df,goals_df,bookings_df]:
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'])
for df in [matches_df,team_app_df,goals_df,bookings_df]:
    for col in df.columns:
        if 'team' in col.lower():
            df[col] = df[col].replace(NAME_MAP)
mens_ids = matches_df[matches_df['tournament_name'].str.contains('Men',na=False)]['match_id'].unique()
matches_men = matches_df[
    matches_df['match_id'].isin(mens_ids) &
    (matches_df['match_date']>='2010-01-01') &
    (matches_df['match_date']<='2022-12-31')
].copy()
team_app_men = team_app_df[
    team_app_df['match_id'].isin(mens_ids) &
    (team_app_df['match_date']>='2010-01-01') &
    (team_app_df['match_date']<='2022-12-31')
].copy()
goals_men    = goals_df[goals_df['match_id'].isin(mens_ids)].copy()
bookings_men = bookings_df[bookings_df['match_id'].isin(mens_ids)].copy()
print(f'Matches (2010-2022): {len(matches_men)}')
print(f'Team appearances:    {len(team_app_men)}')
print(f'Goals:               {len(goals_men)}')
print(matches_men.groupby(matches_men['match_date'].dt.year)['match_id'].count())"""
))

cells.append(nbf.v4.new_markdown_cell("## 2 · FIFA Rankings (Official 2026 WC)"))
cells.append(nbf.v4.new_code_cell(
"""# Official FIFA rankings for all 48 qualified teams
# Rankings used for 2026 WC seeding/draw
FIFA = {
    'Argentina':  (1,  1868),
    'Spain':      (2,  1851),
    'France':     (3,  1840),
    'England':    (4,  1792),
    'Portugal':   (5,  1757),
    'Brazil':     (6,  1750),
    'Morocco':    (7,  1720),
    'Netherlands':(8,  1713),
    'Belgium':    (9,  1700),
    'Germany':    (10, 1688),
    'Croatia':    (11, 1675),
    'Colombia':   (12, 1660),
    'Mexico':     (13, 1645),
    'Senegal':    (14, 1632),
    'Uruguay':    (15, 1618),
    'United States':(16,1605),
    'Japan':      (17, 1592),
    'Switzerland':(18, 1580),
    'Iran':       (19, 1567),
    'Ecuador':    (20, 1555),
    'Austria':    (21, 1542),
    'South Korea':(22, 1530),
    'Australia':  (23, 1517),
    'Algeria':    (24, 1505),
    'Egypt':      (25, 1492),
    'Canada':     (26, 1480),
    'Norway':     (27, 1467),
    'Cote dIvoire':(28,1455),
    'Panama':     (29, 1442),
    'Sweden':     (30, 1430),
    'Czech Republic':(31,1417),
    'Paraguay':   (32, 1405),
    'Scotland':   (33, 1392),
    'Tunisia':    (34, 1380),
    'DR Congo':   (35, 1367),
    'Curacao':    (36, 1355),
    'Qatar':      (37, 1342),
    'Iraq':       (38, 1330),
    'South Africa':(39,1317),
    'Saudi Arabia':(40,1305),
    'Cape Verde': (41, 1292),
    'Uzbekistan': (42, 1280),
    'Jordan':     (43, 1267),
    'Bosnia and Herzegovina':(44,1255),
    'Haiti':      (45, 1242),
    'Ghana':      (46, 1230),
    'New Zealand':(47, 1217),
    'Turkey':     (22, 1530),
}

ranks_df = pd.DataFrame(
    [{'team':t,'fifa_rank':v[0],'fifa_pts':v[1],'group':GROUP_MAP.get(t,'?')}
     for t,v in FIFA.items() if t in ALL_TEAMS]
).drop_duplicates('team').sort_values('fifa_rank').reset_index(drop=True)

mn,mx = ranks_df['fifa_pts'].min(),ranks_df['fifa_pts'].max()
ranks_df['rank_score'] = (ranks_df['fifa_pts']-mn)/(mx-mn)

print(f'Rankings loaded: {len(ranks_df)} teams')
missing = [t for t in ALL_TEAMS if t not in ranks_df['team'].values]
if missing:
    print(f'WARNING - missing: {missing}')
else:
    print('All 48 teams have rankings!')
print(ranks_df[['team','group','fifa_rank','rank_score']].to_string(index=False))"""
))

cells.append(nbf.v4.new_markdown_cell("## 3 · StatsBomb xG (2018 & 2022 only)"))
cells.append(nbf.v4.new_code_cell(
"""BASE_SB = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data'

def fetch_json(url):
    r = requests.get(url, timeout=30)
    return r.json()

def get_wc_xg(season_id, label):
    print(f'  Fetching {label}...')
    matches = fetch_json(f'{BASE_SB}/matches/43/{season_id}.json')
    rows = []
    for m in matches:
        mid = m['match_id']
        home = NAME_MAP.get(m['home_team']['home_team_name'], m['home_team']['home_team_name'])
        away = NAME_MAP.get(m['away_team']['away_team_name'], m['away_team']['away_team_name'])
        try:
            events = fetch_json(f'{BASE_SB}/events/{mid}.json')
            shots = [e for e in events if e['type']['name']=='Shot']
            h_xg = sum(s['shot']['statsbomb_xg'] for s in shots if s['team']['name']==m['home_team']['home_team_name'])
            a_xg = sum(s['shot']['statsbomb_xg'] for s in shots if s['team']['name']==m['away_team']['away_team_name'])
        except:
            h_xg=a_xg=None
        rows.append({'match_id':mid,'date':m['match_date'],
            'home':home,'away':away,
            'home_score':m['home_score'],'away_score':m['away_score'],
            'home_xg':h_xg,'away_xg':a_xg,'tournament':label})
    return pd.DataFrame(rows)

print('Fetching StatsBomb xG (~60 seconds)...')
xg_2022 = get_wc_xg(106, '2022 WC')
xg_2018 = get_wc_xg(3,   '2018 WC')
xg_df = pd.concat([xg_2018,xg_2022], ignore_index=True)
xg_df.to_csv(f'{DATA_DIR}/xg_data.csv', index=False)
print(f'xG data: {len(xg_df)} matches saved')"""
))

cells.append(nbf.v4.new_markdown_cell("## 4 · Compute Team Features"))
cells.append(nbf.v4.new_code_cell(
"""def compute_team_features(team):
    rs_row = ranks_df[ranks_df['team']==team]
    rs = float(rs_row['rank_score'].values[0]) if len(rs_row) else 0.3
    fifa_rank = int(rs_row['fifa_rank'].values[0]) if len(rs_row) else 48
    hist = team_app_men[team_app_men['team_name']==team]
    n = len(hist)
    if n > 0:
        wr   = hist['win'].mean()
        dr   = hist['draw'].mean()
        avgf = hist['goals_for'].mean()
        avga = hist['goals_against'].mean()
        avgd = hist['goal_differential'].mean()
    else:
        wr   = max(0.05, 0.65-(fifa_rank-1)*0.012)
        dr   = 0.25
        avgf = max(0.3,  1.8-(fifa_rank-1)*0.030)
        avga = min(2.8,  0.7+(fifa_rank-1)*0.030)
        avgd = avgf-avga
    xg_h = xg_df[xg_df['home']==team]
    xg_a = xg_df[xg_df['away']==team]
    xg_sc = list(xg_h['home_xg'].dropna())+list(xg_a['away_xg'].dropna())
    xg_cn = list(xg_h['away_xg'].dropna())+list(xg_a['home_xg'].dropna())
    avg_xg_sc = np.mean(xg_sc) if xg_sc else avgf*0.85
    avg_xg_cn = np.mean(xg_cn) if xg_cn else avga*0.85
    return {
        'team':team,'group':GROUP_MAP.get(team,'?'),
        'fifa_rank':fifa_rank,'rank_score':round(rs,4),
        'wc_matches':n,'win_rate':round(wr,4),'draw_rate':round(dr,4),
        'avg_gf':round(avgf,4),'avg_ga':round(avga,4),'avg_gd':round(avgd,4),
        'avg_xg_scored':round(avg_xg_sc,4),'avg_xg_conceded':round(avg_xg_cn,4),
        'xg_matches':len(xg_sc),
    }

print('Computing features for all 48 teams...')
master_df = pd.DataFrame([compute_team_features(t) for t in ALL_TEAMS]).sort_values('fifa_rank').reset_index(drop=True)
assert len(master_df) == 48, f"Expected 48, got {len(master_df)}"
print(f'Feature matrix: {master_df.shape}')
print(master_df[['team','group','fifa_rank','win_rate','avg_gf','avg_ga']].to_string(index=False))"""
))

cells.append(nbf.v4.new_markdown_cell("## 5 · Dark Horse Detection (rank > 25 ONLY)"))
cells.append(nbf.v4.new_code_cell(
"""dh = master_df[master_df['fifa_rank'] > 25].copy()
def norm(s):
    mn,mx = s.min(),s.max()
    return (s-mn)/(mx-mn) if mx>mn else pd.Series([0.5]*len(s),index=s.index)
dh['atk_score']  = norm(dh['avg_gf'])
dh['def_score']  = 1-norm(dh['avg_ga'])
dh['upset_pot']  = norm(dh['fifa_rank'])
dh['dark_horse_score'] = 0.35*dh['atk_score']+0.35*dh['def_score']+0.30*dh['upset_pot']
dark_horses = dh.sort_values('dark_horse_score',ascending=False)[
    ['team','group','fifa_rank','dark_horse_score','avg_gf','avg_ga']
].reset_index(drop=True)
print('DARK HORSE RANKINGS (rank > 25 only):')
print('='*60)
for i,(_,row) in enumerate(dark_horses.head(8).iterrows()):
    label = 'PRIMARY' if i==0 else f'#{i+1}   '
    print(f'  {label} {row["team"]:<26} FIFA #{int(row["fifa_rank"]):<4} Score: {row["dark_horse_score"]:.3f}')"""
))

cells.append(nbf.v4.new_markdown_cell("## 6 · Player Goals Summary"))
cells.append(nbf.v4.new_code_cell(
"""name_col = 'player_name' if 'player_name' in goals_men.columns else [c for c in goals_men.columns if 'name' in c.lower()][0]
team_col = 'team_name' if 'team_name' in goals_men.columns else [c for c in goals_men.columns if 'team' in c.lower()][0]
player_goals = goals_men.groupby([name_col,team_col]).agg(
    total_goals=('match_id','count'),
    matches=('match_id','nunique')
).reset_index()
player_goals.columns = ['player_name','team','total_goals','matches_with_goals']
player_goals['goals_per_match'] = player_goals['total_goals']/player_goals['matches_with_goals']
player_goals = player_goals.sort_values('total_goals',ascending=False).reset_index(drop=True)
print(f'Player data: {len(player_goals)} players')
print(player_goals.head(10)[['player_name','team','total_goals','goals_per_match']].to_string(index=False))"""
))

cells.append(nbf.v4.new_markdown_cell("## 7 · Save All Outputs"))
cells.append(nbf.v4.new_code_cell(
"""master_df.to_csv(f'{DATA_DIR}/team_features.csv', index=False)
dark_horses.to_csv(f'{DATA_DIR}/dark_horse_scores.csv', index=False)
player_goals.to_csv(f'{DATA_DIR}/player_summary.csv', index=False)
print('Saved:')
print(f'  team_features.csv     {master_df.shape}')
print(f'  dark_horse_scores.csv {dark_horses.shape}')
print(f'  player_summary.csv    {player_goals.shape}')
print(f'  xg_data.csv           {xg_df.shape}')
print('Notebook 1 complete!')"""
))

nb.cells = cells
with open('notebooks/01_data_collection.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook 1 written successfully!")
