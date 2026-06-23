import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.graph_objects as go
import requests
from itertools import combinations
from scipy.stats import poisson
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="2026 FIFA World Cup Predictor", page_icon="🏆", layout="wide")

st.markdown("""<style>
.stApp{background:#0a0e1a;color:#fff}
[data-testid="stSidebar"]{background:#0d1117;border-right:2px solid #FFD700}
.card{background:linear-gradient(135deg,#0d1117,#1a1f2e);border:1px solid #FFD700;border-radius:12px;padding:18px;margin:6px 0;text-align:center}
.mbox{background:#0d1117;border:1px solid #1e2a3a;border-radius:10px;padding:10px 14px;margin:4px 0}
.winner{color:#FFD700;font-weight:bold}
.loser{color:#555}
.score{color:#FFD700;font-size:1.1em;font-weight:bold;margin:3px 0}
.prob{color:#444;font-size:0.72em;margin-top:3px}
.sec{color:#FFD700;font-size:1.3em;font-weight:bold;margin:12px 0 8px;padding-bottom:5px;border-bottom:1px solid #333}
.gbadge{background:#1a4a1a;color:#00ff88;padding:2px 8px;border-radius:8px;font-size:0.82em}
.rbadge{background:#4a1a1a;color:#ff6666;padding:2px 8px;border-radius:8px;font-size:0.82em}
.ybadge{background:#4a3a00;color:#FFD700;padding:2px 8px;border-radius:8px;font-size:0.82em}
h1,h2,h3{color:#FFD700!important}
div[data-testid="stMetricValue"]{color:#FFD700!important}
</style>""", unsafe_allow_html=True)

API_KEY = "600408d82e9741fab00cd6a8b8d713e9"
BASE_URL = "https://api.football-data.org/v4"
API_HEADERS = {"X-Auth-Token": API_KEY}
NAME_MAP = {"Czechia":"Czech Republic","Bosnia-Herzegovina":"Bosnia and Herzegovina","Turkiye":"Turkey","IR Iran":"Iran","Korea Republic":"South Korea","Congo DR":"DR Congo","Côte d'Ivoire":"Cote dIvoire"}

GROUPS = {
    'A':['Mexico','South Korea','Czech Republic','South Africa'],
    'B':['Bosnia and Herzegovina','Canada','Qatar','Switzerland'],
    'C':['Brazil','Haiti','Morocco','Scotland'],
    'D':['Australia','Paraguay','Turkey','United States'],
    'E':['Curacao','Ecuador','Germany','Cote dIvoire'],
    'F':['Japan','Netherlands','Sweden','Tunisia'],
    'G':['Belgium','Egypt','Iran','New Zealand'],
    'H':['Cape Verde','Saudi Arabia','Spain','Uruguay'],
    'I':['France','Iraq','Norway','Senegal'],
    'J':['Algeria','Argentina','Austria','Jordan'],
    'K':['Colombia','DR Congo','Portugal','Uzbekistan'],
    'L':['Croatia','England','Ghana','Panama'],
}

@st.cache_resource
def load_data():
    tf = pd.read_csv("data/team_features.csv")
    sr = pd.read_csv("data/simulation_results.csv")
    gb = pd.read_csv("data/golden_boot.csv")
    dh = pd.read_csv("data/dark_horse_scores.csv")
    br = pd.read_csv("data/bracket_predictions.csv")
    import xgboost as xgb
    import json
    import numpy as np

    # Load XGBoost from version-independent JSON
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model("data/xgb_model.json")

    # Load Poisson params from pure JSON - no sklearn version dependency
    with open("data/model_params.json","r") as f:
        params = json.load(f)

    # Rebuild Poisson predictors as simple lambda functions using saved coefficients
    home_coef = np.array(params["poisson_home"]["coef"])
    home_int  = params["poisson_home"]["intercept"]
    away_coef = np.array(params["poisson_away"]["coef"])
    away_int  = params["poisson_away"]["intercept"]

    class SimplePoisson:
        def __init__(self, coef, intercept):
            self.coef_ = coef
            self.intercept_ = intercept
        def predict(self, X):
            return np.exp(np.dot(np.array(X), self.coef_) + self.intercept_)

    model = {
        "classifier": xgb_clf,
        "poisson_home": SimplePoisson(home_coef, home_int),
        "poisson_away": SimplePoisson(away_coef, away_int),
        "test_accuracy": params.get("test_accuracy", 0.516),
        "mae": params.get("mae", 0.95),
    }
    return tf, sr, gb, dh, br, model

@st.cache_data(ttl=3600)
def fetch_live():
    try:
        r = requests.get(f"{BASE_URL}/competitions/2000/matches", headers=API_HEADERS, timeout=10)
        rows = []
        for m in r.json().get("matches",[]):
            home = NAME_MAP.get(m["homeTeam"]["name"], m["homeTeam"]["name"])
            away = NAME_MAP.get(m["awayTeam"]["name"], m["awayTeam"]["name"])
            sc = m["score"]["fullTime"]
            rows.append({"date":m["utcDate"][:10],"stage":m["stage"],"group":m.get("group",""),"home":home,"away":away,"status":m["status"],"home_score":sc["home"],"away_score":sc["away"]})
        return pd.DataFrame(rows)
    except: return pd.DataFrame()

tf, sr, gb, dh, br, model = load_data()
clf = model["classifier"]
feat_idx = tf.set_index("team")
avg_gf = tf["avg_gf"].mean(); avg_ga = tf["avg_ga"].mean()
sim_rank = sr.set_index("team")["p_winner"].to_dict()
# Champion = winner of the Final in the bracket (single most likely path)
_final_row = br[br["round"]=="Final"].iloc[0] if len(br[br["round"]=="Final"])>0 else None
champion = _final_row["predicted_winner"] if _final_row is not None else sr.iloc[0]["team"]

def gs(t,c,d=0.0):
    try: v=feat_idx.loc[t,c]; return float(v) if not pd.isna(v) else d
    except: return d

def match_probs(home, away, knockout=False):
    hr=gs(home,"rank_score",0.4); ar=gs(away,"rank_score",0.4)
    feat=np.array([[hr-ar,gs(home,"win_rate",0.33)-gs(away,"win_rate",0.33),gs(home,"avg_gf",1.2)-gs(away,"avg_gf",1.2),gs(home,"avg_ga",1.2)-gs(away,"avg_ga",1.2),gs(home,"avg_gd",0.0)-gs(away,"avg_gd",0.0),hr,ar,int(knockout),gs(home,"avg_xg_scored",1.0)-gs(away,"avg_xg_scored",1.0)]])
    p=clf.predict_proba(feat)[0]; ph,pd_,pa=p[0],p[1],p[2]
    rd=hr-ar; rph=1/(1+np.exp(-4*rd)); rpa=1-rph; rpd=0.25
    norm=rph+rpd+rpa; rph/=norm; rpa/=norm; rpd/=norm
    ph=0.3*ph+0.7*rph; pa=0.3*pa+0.7*rpa; pd_=0.3*pd_+0.7*rpd
    if knockout: ph+=pd_/2; pa+=pd_/2; pd_=0
    return float(ph),float(pd_),float(pa)

def predict_score(home, away):
    h_atk=gs(home,"avg_gf",avg_gf)/avg_gf; h_def=gs(home,"avg_ga",avg_ga)/avg_ga
    a_atk=gs(away,"avg_gf",avg_gf)/avg_gf; a_def=gs(away,"avg_ga",avg_ga)/avg_ga
    lh=max(0.6,h_atk*a_def*avg_gf*1.1); la=max(0.4,a_atk*h_def*avg_gf*0.9)
    ph,pd_,pa=match_probs(home,away)
    d=ph-pa
    if d>0.3: lh=max(lh,la+1.2); la=max(0.3,la-0.3)
    elif d>0.15: lh=max(lh,la+0.7); la=max(0.4,la-0.2)
    elif d>0.05: lh=max(lh,la+0.3)
    elif d<-0.3: la=max(la,lh+1.2); lh=max(0.3,lh-0.3)
    elif d<-0.15: la=max(la,lh+0.7); lh=max(0.4,lh-0.2)
    elif d<-0.05: la=max(la,lh+0.3)
    bp,bs=0,(1,0)
    for hg in range(9):
        for ag in range(9):
            p=poisson.pmf(hg,lh)*poisson.pmf(ag,la)
            if p>bp: bp=p; bs=(hg,ag)
    return bs

# SIDEBAR
with st.sidebar:
    st.markdown("## 🏆 2026 WC Predictor")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Overview","⚽ Group Stage","🏟️ Knockout Bracket","🥇 Golden Boot","🌙 Dark Horse","ℹ️ How It Works"])
    st.markdown("---")
    if st.button("🔄 Refresh Scores"): st.cache_data.clear(); st.rerun()
    st.markdown("<small style='color:#555'>Live data: football-data.org<br>Model: XGBoost + Poisson<br>Training: 2010–2022 WC only</small>", unsafe_allow_html=True)

# ── OVERVIEW ──────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("<h1 style='text-align:center'>🏆 2026 FIFA World Cup Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#aaa'>XGBoost + Poisson · 10,000 Monte Carlo Simulations · Live API Data</p>", unsafe_allow_html=True)
    st.markdown("---")

        # Winner card — use bracket champion for consistency
    winner = champion
    winner_prob = sr[sr["team"]==champion]["p_winner"].values[0] if champion in sr["team"].values else sr.iloc[0]["p_winner"]
    
    # Why the model thinks this
    hr = int(feat_idx.loc[winner,'fifa_rank']) if winner in feat_idx.index else 0
    wr = feat_idx.loc[winner,'win_rate'] if winner in feat_idx.index else 0
    gf = feat_idx.loc[winner,'avg_gf'] if winner in feat_idx.index else 0
    xg = feat_idx.loc[winner,'avg_xg_scored'] if winner in feat_idx.index else 0

    c1,c2 = st.columns([1,1])
    with c1:
        st.markdown(f"""<div class='card' style='border-color:#FFD700;padding:30px'>
            <div style='font-size:3em'>🏆</div>
            <div style='font-size:2.2em;font-weight:bold;color:white'>{winner}</div>
            <div style='font-size:2.8em;font-weight:bold;color:#FFD700;margin:8px 0'>{winner_prob:.1%}</div>
            <div style='color:#aaa'>Predicted Most Likely Winner</div>
            <div style='color:#555;font-size:0.8em;margin-top:8px'>Based on 10,000 tournament simulations</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='padding:20px'><div style='color:#FFD700;font-size:1.1em;font-weight:bold;margin-bottom:12px'>Why the model predicts this:</div>", unsafe_allow_html=True)
        reasons = [
            f"🏅 FIFA Rank #{hr} — one of the highest-ranked teams in the tournament",
            f"⚽ {gf:.2f} average goals per World Cup match (2010–2022)",
            f"📈 {wr:.0%} historical win rate in World Cup matches",
            f"🎯 {xg:.2f} expected goals per match (StatsBomb xG data)",
            f"🔄 Won {winner_prob:.1%} of 10,000 full tournament simulations",
            f"📊 70% FIFA ranking weight ensures top teams are properly favoured",
        ]
        for r in reasons:
            st.markdown(f"<div style='background:#0d1117;border-left:3px solid #FFD700;padding:8px 12px;margin:6px 0;border-radius:0 8px 8px 0;color:#ddd;font-size:0.9em'>{r}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Actual winner from API
    st.markdown("<div class='sec'>✅ Actual World Cup Winner</div>", unsafe_allow_html=True)
    live = fetch_live()
    if len(live) > 0:
        final_m = live[(live["stage"]=="FINAL")&(live["status"]=="FINISHED")]
        if len(final_m) > 0:
            fm = final_m.iloc[0]
            actual_winner = fm["home"] if fm["home_score"] > fm["away_score"] else fm["away"]
            st.markdown(f"""<div class='card' style='max-width:300px;margin:0 auto'>
                <div style='font-size:2em'>🏆</div>
                <div style='font-size:2em;color:#FFD700;font-weight:bold'>{actual_winner}</div>
                <div style='color:#aaa'>2026 World Cup Champion</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("🕐 Tournament in progress — actual winner appears here automatically after the Final.")
    else:
        st.info("🕐 Live data loading...")

# ── GROUP STAGE ───────────────────────────────────────────────────
elif page == "⚽ Group Stage":
    st.markdown("<h1>⚽ Group Stage</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔮 Predictions","📊 Prediction vs Actual"])

    with tab1:
        st.markdown("<div class='sec'>All 72 Predicted Group Stage Scores</div>", unsafe_allow_html=True)
        for grp, teams in GROUPS.items():
            with st.expander(f"Group {grp} — {' · '.join(teams)}", expanded=(grp=="A")):
                c1,c2 = st.columns([1,1])
                with c1:
                    st.markdown("**📊 Predicted Standings**")
                    std = {t:{"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"GD":0,"Pts":0} for t in teams}
                    for home,away in combinations(teams,2):
                        ph,pd_,pa = match_probs(home,away)
                        hg,ag = predict_score(home,away)
                        if ph>pa and ph>pd_ and hg<=ag: hg=ag+1
                        elif pa>ph and pa>pd_ and ag<=hg: ag=hg+1
                        std[home]["P"]+=1; std[away]["P"]+=1
                        std[home]["GF"]+=hg; std[home]["GA"]+=ag; std[home]["GD"]+=hg-ag
                        std[away]["GF"]+=ag; std[away]["GA"]+=hg; std[away]["GD"]+=ag-hg
                        if hg>ag: std[home]["Pts"]+=3; std[home]["W"]+=1; std[away]["L"]+=1
                        elif hg==ag: std[home]["Pts"]+=1; std[away]["Pts"]+=1; std[home]["D"]+=1; std[away]["D"]+=1
                        else: std[away]["Pts"]+=3; std[away]["W"]+=1; std[home]["L"]+=1
                    sdf = pd.DataFrame(std).T.sort_values(["Pts","GD","GF"],ascending=False)
                    def cr(row):
                        i = list(sdf.index).index(row.name)
                        if i<2: return ["background-color:#1a3a1a"]*len(row)
                        elif i==2: return ["background-color:#3a3a1a"]*len(row)
                        else: return ["background-color:#3a1a1a"]*len(row)
                    st.dataframe(sdf.style.apply(cr,axis=1), use_container_width=True)
                    st.markdown("<small>🟢 Qualified · 🟡 Possible 3rd · 🔴 Eliminated</small>", unsafe_allow_html=True)
                with c2:
                    st.markdown("**⚽ Predicted Scores**")
                    for home,away in combinations(teams,2):
                        ph,pd_,pa = match_probs(home,away)
                        hg,ag = predict_score(home,away)
                        if ph>pa and ph>pd_ and hg<=ag: hg=ag+1
                        elif pa>ph and pa>pd_ and ag<=hg: ag=hg+1
                        hc="#FFD700" if ph>=pa else "#888"; ac="#FFD700" if pa>ph else "#888"
                        st.markdown(f"""<div class='mbox'>
                            <div style='display:flex;justify-content:space-between;align-items:center'>
                                <span style='color:{hc};font-weight:bold;min-width:100px;font-size:0.9em'>{home}</span>
                                <span style='color:#FFD700;font-size:1.2em;font-weight:bold;margin:0 8px'>{hg} — {ag}</span>
                                <span style='color:{ac};font-weight:bold;min-width:100px;text-align:right;font-size:0.9em'>{away}</span>
                            </div>
                            <div style='color:#444;font-size:0.75em;margin-top:3px'>{home}: {ph:.0%} · Draw: {pd_:.0%} · {away}: {pa:.0%}</div>
                        </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='sec'>Prediction vs Actual Results</div>", unsafe_allow_html=True)
        live = fetch_live()
        finished = live[(live["stage"]=="GROUP_STAGE")&(live["status"]=="FINISHED")] if len(live)>0 else pd.DataFrame()
        if len(finished)==0:
            st.info("No group stage results yet. Updates automatically as matches are played.")
        else:
            correct=0; rows=[]
            for _,m in finished.iterrows():
                ph,pd_,pa = match_probs(m["home"],m["away"])
                hg,ag = predict_score(m["home"],m["away"])
                if ph>pa and ph>pd_ and hg<=ag: hg=ag+1
                elif pa>ph and pa>pd_ and ag<=hg: ag=hg+1
                ah,aa = int(m["home_score"]),int(m["away_score"])
                ao="H" if ah>aa else ("D" if ah==aa else "A")
                po="H" if ph>pa and ph>pd_ else ("D" if pd_>ph and pd_>pa else "A")
                ok=ao==po; correct+=int(ok)
                if (hg,ag)==(ah,aa): badge="<span class='gbadge'>🟢 Perfect score</span>"
                elif ok: badge="<span class='ybadge'>🟡 Right winner</span>"
                else: badge="<span class='rbadge'>🔴 Wrong</span>"
                rows.append({"Date":m["date"],"Match":f"{m['home']} vs {m['away']}","🔮":f"{hg}–{ag}","✅":f"{ah}–{aa}","Result":badge})
            c1,c2,c3 = st.columns(3)
            c1.metric("Matches Played",len(finished))
            c2.metric("Outcome Accuracy",f"{correct/len(finished):.1%}")
            c3.metric("Correct",f"{correct}/{len(finished)}")
            for r in rows:
                st.markdown(f"""<div class='mbox' style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
                    <span style='color:#aaa;font-size:0.85em'>{r['Date']}</span>
                    <span style='font-weight:bold'>{r['Match']}</span>
                    <span style='color:#aaa'>🔮 {r['🔮']}</span>
                    <span style='color:#FFD700'>✅ {r['✅']}</span>
                    <span>{r['Result']}</span>
                </div>""", unsafe_allow_html=True)

# ── KNOCKOUT BRACKET ──────────────────────────────────────────────
elif page == "🏟️ Knockout Bracket":
    st.markdown("<h1>🏟️ Knockout Bracket</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔮 Predicted Bracket","📊 Bracket vs Actual"])

    with tab1:
        st.markdown("<div class='sec'>Model's Predicted Path to Glory</div>", unsafe_allow_html=True)
        _champ_prob = sr[sr["team"]==champion]["p_winner"].values[0] if champion in sr["team"].values else 0
        st.markdown(f"<p style='color:#aaa;font-size:0.9em'>This shows the model's single most likely path through the tournament. <b style='color:#FFD700'>{champion}</b> wins the Final and has a {_champ_prob:.1%} win probability across 10,000 simulations.</p>", unsafe_allow_html=True)

        def mcard(row, is_final=False):
            h=str(row["home"]); a=str(row["away"])
            hg=int(row["pred_home_score"]); ag=int(row["pred_away_score"])
            w=str(row["predicted_winner"])
            ph=float(row["home_win_prob"]); pa=float(row["away_win_prob"])
            hc="#FFD700" if w==h else "#666"
            ac="#FFD700" if w==a else "#666"
            hfw="bold" if w==h else "normal"
            afw="bold" if w==a else "normal"
            border="2px solid #FFD700" if is_final else "1px solid #1e2a3a"
            bg="#1a1500" if is_final else "#0d1117"
            return f"""<div style='background:{bg};border:{border};border-radius:10px;padding:10px 14px;margin:4px 0'>
                <div style='color:{hc};font-weight:{hfw};font-size:0.88em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{h}</div>
                <div style='color:#FFD700;font-size:1.05em;font-weight:bold;margin:3px 0'>{hg} — {ag}</div>
                <div style='color:{ac};font-weight:{afw};font-size:0.88em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{a}</div>
                <div style='color:#444;font-size:0.7em;margin-top:3px'>{ph:.0%} · {pa:.0%}</div>
            </div>"""

        round_config = [
            ("R32","Round of 32",4),
            ("R16","Round of 16",4),
            ("QF","Quarter Finals",4),
            ("SF","Semi Finals",2),
            ("Final","🏆 Final",1),
        ]
        for rnd, label, ncols in round_config:
            rm = br[br["round"]==rnd]
            if len(rm)==0: continue
            st.markdown(f"### {label}")
            if rnd=="Final":
                row = rm.iloc[0]
                c1,c2,c3 = st.columns([1,2,1])
                with c2:
                    st.markdown(mcard(row, is_final=True), unsafe_allow_html=True)
                prob = sr[sr["team"]==champion]["p_winner"].values[0]
                st.markdown(f"""<div style='text-align:center;margin:20px 0;padding:28px;
                    background:linear-gradient(135deg,#1a1500,#2a2000);
                    border:2px solid #FFD700;border-radius:16px'>
                    <div style='font-size:3em'>🏆</div>
                    <div style='font-size:2.2em;color:#FFD700;font-weight:bold'>{champion}</div>
                    <div style='color:#aaa;margin-top:6px'>Predicted Champion · {prob:.1%} win probability from 10,000 simulations</div>
                </div>""", unsafe_allow_html=True)
            else:
                cols = st.columns(ncols)
                for i,(_,row) in enumerate(rm.iterrows()):
                    with cols[i%ncols]:
                        st.markdown(mcard(row), unsafe_allow_html=True)
            st.markdown("---")

    with tab2:
        st.markdown("<div class='sec'>Predicted vs Actual Knockout Results</div>", unsafe_allow_html=True)
        live = fetch_live()
        ko_stages = ["LAST_32","LAST_16","QUARTER_FINALS","SEMI_FINALS","THIRD_PLACE","FINAL"]
        ko = live[live["stage"].isin(ko_stages)] if len(live)>0 else pd.DataFrame()
        finished_ko = ko[ko["status"]=="FINISHED"] if len(ko)>0 else pd.DataFrame()
        if len(finished_ko)==0:
            st.info("Knockout results appear here automatically as matches are played.")
        else:
            sm={"LAST_32":"R32","LAST_16":"R16","QUARTER_FINALS":"QF","SEMI_FINALS":"SF","FINAL":"Final"}
            correct=0; total=0
            for _,m in finished_ko.iterrows():
                rnd=sm.get(m["stage"],m["stage"])
                ah,aa=int(m["home_score"]),int(m["away_score"])
                aw=m["home"] if ah>aa else m["away"]
                pm=br[(br["round"]==rnd)&(((br["home"]==m["home"])&(br["away"]==m["away"]))|((br["home"]==m["away"])&(br["away"]==m["home"])))]
                if len(pm)>0:
                    pw=pm.iloc[0]["predicted_winner"]; ok=pw==aw; total+=1; correct+=int(ok)
                    badge=f"<span class='gbadge'>✅ Correct</span>" if ok else f"<span class='rbadge'>❌ Wrong</span>"
                    st.markdown(f"""<div class='mbox' style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
                        <span style='color:#FFD700;font-size:0.85em;font-weight:bold'>{rnd}</span>
                        <span style='font-weight:bold'>{m['home']} vs {m['away']}</span>
                        <span style='color:#aaa'>🔮 Pred: <b>{pw}</b></span>
                        <span style='color:#FFD700'>✅ Actual: <b>{aw}</b></span>
                        <span>{badge}</span>
                    </div>""", unsafe_allow_html=True)
            if total>0:
                st.metric("Knockout Accuracy", f"{correct/total:.1%}")

# ── GOLDEN BOOT ───────────────────────────────────────────────────
elif page == "🥇 Golden Boot":
    st.markdown("<h1>🥇 Golden Boot Predictions</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔮 Prediction","📊 Prediction vs Actual"])

    with tab1:
        st.markdown("<div class='sec'>Top 3 Predicted Scorers</div>", unsafe_allow_html=True)
        top3_gb = gb[gb["position"]=="Offence"].head(3) if "position" in gb.columns else gb.head(3)
        if len(top3_gb)<3: top3_gb=gb.head(3)
        c1,c2,c3 = st.columns(3)
        for col,(medal,border),(_,row) in zip([c1,c2,c3],[("🥇","#FFD700"),("🥈","#C0C0C0"),("🥉","#CD7F32")],top3_gb.iterrows()):
            with col:
                st.markdown(f"""<div class='card' style='border-color:{border}'>
                    <div style='font-size:2em'>{medal}</div>
                    <div style='font-size:1.2em;font-weight:bold;color:white'>{row['player']}</div>
                    <div style='color:#aaa;font-size:0.85em'>{row['team']}</div>
                    <div style='font-size:2em;font-weight:bold;color:#FFD700'>{row['goals_per_match']:.3f}</div>
                    <div style='color:#aaa;font-size:0.82em'>Goals per Match</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br><div class='sec'>Top 10 Predicted Scorers</div>", unsafe_allow_html=True)
        t10 = gb[gb["position"]=="Offence"].head(10) if "position" in gb.columns else gb.head(10)
        if len(t10)<5: t10=gb.head(10)
        for i,(_,row) in enumerate(t10.iterrows()):
            rank_emoji = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"][i]
            st.markdown(f"""<div class='mbox' style='display:flex;justify-content:space-between;align-items:center'>
                <span style='font-size:1.1em;min-width:35px'>{rank_emoji}</span>
                <span style='color:#FFD700;font-weight:bold;min-width:180px;font-size:0.95em'>{row['player']}</span>
                <span style='color:#aaa;min-width:120px'>{row['team']}</span>
                <span style='color:#FFD700;font-weight:bold'>{row['goals_per_match']:.3f} goals/match</span>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='sec'>Prediction vs Actual Top Scorers</div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**🔮 Predicted Top 3**")
            p3 = top3_gb[["player","team","expected_matches","projected_goals"]].copy()
            p3.index=["🥇","🥈","🥉"]; p3.columns=["Player","Team","Exp. Games","Proj. Goals"]
            st.dataframe(p3, use_container_width=True)
        with c2:
            st.markdown("**✅ Actual Top Scorers**")
            st.info("Top scorers appear here automatically as goals are scored during the tournament.")

# ── DARK HORSE ────────────────────────────────────────────────────
elif page == "🌙 Dark Horse":
    st.markdown("<h1>🌙 Dark Horse Picks</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔮 Prediction","📊 Prediction vs Actual"])
    dh0 = dh.iloc[0]
    sim_row = sr[sr["team"]==dh0["team"]]
    wp = sim_row["p_winner"].values[0] if len(sim_row)>0 else 0
    qp = sim_row["p_qf"].values[0] if len(sim_row)>0 else 0

    with tab1:
        st.markdown("<div class='sec'>Primary Dark Horse Pick</div>", unsafe_allow_html=True)
        st.markdown(f"""<div class='card' style='max-width:450px;margin:0 auto'>
            <div style='font-size:3em'>👑</div>
            <div style='font-size:2em;color:#FFD700;font-weight:bold'>{dh0['team']}</div>
            <div style='color:#aaa;margin:6px 0'>Primary Dark Horse Pick · FIFA #{int(dh0['fifa_rank'])}</div>
            <div style='display:flex;justify-content:center;gap:30px;margin-top:12px'>
                <div><div style='color:#FFD700;font-size:1.3em;font-weight:bold'>{wp:.1%}</div><div style='color:#555;font-size:0.8em'>Win Prob</div></div>
                <div><div style='color:#FFD700;font-size:1.3em;font-weight:bold'>{qp:.1%}</div><div style='color:#555;font-size:0.8em'>QF Prob</div></div>
                <div><div style='color:#FFD700;font-size:1.3em;font-weight:bold'>{dh0['dark_horse_score']:.3f}</div><div style='color:#555;font-size:0.8em'>DH Score</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br><div class='sec'>Top 5 Dark Horses</div>", unsafe_allow_html=True)
        for i,(_,row) in enumerate(dh.head(5).iterrows()):
            s=sr[sr["team"]==row["team"]]; wp2=s["p_winner"].values[0] if len(s)>0 else 0
            crown="👑" if i==0 else f"#{i+1}"
            st.markdown(f"""<div class='mbox' style='display:flex;justify-content:space-between;align-items:center'>
                <span style='font-size:1.1em;min-width:30px'>{crown}</span>
                <span style='color:#FFD700;font-weight:bold;min-width:150px'>{row['team']}</span>
                <span style='color:#aaa'>FIFA #{int(row['fifa_rank'])}</span>
                <span style='color:#aaa'>Score: {row['dark_horse_score']:.3f}</span>
                <span style='color:#FFD700'>Win: {wp2:.1%}</span>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='sec'>Dark Horse — Prediction vs Actual</div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**🔮 Predicted**")
            st.markdown(f"""<div class='card'><div style='font-size:2em'>👑</div>
                <div style='font-size:1.5em;color:#FFD700'>{dh0['team']}</div>
                <div style='color:#aaa'>FIFA #{int(dh0['fifa_rank'])}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("**✅ Actual Dark Horse**")
            st.info("The actual dark horse emerges as the tournament progresses.")

# ── HOW IT WORKS ──────────────────────────────────────────────────
elif page == "ℹ️ How It Works":
    st.markdown("<h1>ℹ️ How It Works</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sec'>Data Sources (2010–2022 only)</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""<div class='card'><div style='font-size:1.3em;margin-bottom:12px'>📊 Historical Data</div>
        <div style='text-align:left'>
            <div style='color:#FFD700;margin:8px 0'>🏆 jfjelstul/worldcup (GitHub)</div>
            <div style='color:#aaa;font-size:0.88em'>All WC matches 2010–2022 · Goals, bookings, team stats</div>
            <div style='color:#FFD700;margin:8px 0'>⚡ StatsBomb Open Data</div>
            <div style='color:#aaa;font-size:0.88em'>xG & shot data · 2018 & 2022 WC only</div>
            <div style='color:#FFD700;margin:8px 0'>🌍 FIFA Rankings (June 2026)</div>
            <div style='color:#aaa;font-size:0.88em'>Official ranking points · All 48 qualified teams</div>
            <div style='color:#FFD700;margin:8px 0'>🔴 football-data.org (Live API)</div>
            <div style='color:#aaa;font-size:0.88em'>Real-time 2026 match scores · Current squad data</div>
        </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='card'><div style='font-size:1.3em;margin-bottom:12px'>🤖 Model Architecture</div>
        <div style='text-align:left'>
            <div style='color:#FFD700;margin:8px 0'>🎯 XGBoost Classifier</div>
            <div style='color:#aaa;font-size:0.88em'>Predicts: Home Win / Draw / Away Win<br>Features: FIFA rank, xG, win rate, goals, form</div>
            <div style='color:#FFD700;margin:8px 0'>📐 Poisson Scoreline Model</div>
            <div style='color:#aaa;font-size:0.88em'>Attack strength × Defence weakness<br>Method used by professional bookmakers</div>
            <div style='color:#FFD700;margin:8px 0'>🔄 10,000 Monte Carlo Simulations</div>
            <div style='color:#aaa;font-size:0.88em'>Full tournament simulated 10,000 times<br>Win % = how often each team won</div>
            <div style='color:#FFD700;margin:8px 0'>⚖️ 70% FIFA Ranking Weight</div>
            <div style='color:#aaa;font-size:0.88em'>Prevents historical bias · Top teams properly favoured</div>
        </div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    acc=model.get("test_accuracy",0); mae=model.get("mae",0)
    c1,c2,c3 = st.columns(3)
    c1.metric("Model Accuracy on 2022 WC", f"{acc:.1%}", "vs 33% random baseline")
    c2.metric("Scoreline MAE", f"{mae:.2f} goals", "avg error per team")
    c3.metric("Training Window", "2010–2022", "Modern era WC only")
