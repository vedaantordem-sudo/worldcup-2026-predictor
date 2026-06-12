# 2026 FIFA World Cup Predictor

A complete machine-learning pipeline that simulates the 2026 FIFA World Cup 10,000 times and presents results through an interactive Streamlit dashboard.

## Features

- **XGBoost classifier** — predicts Home Win / Draw / Away Win probabilities
- **Poisson regression** — projects expected scoreline for any matchup
- **10,000 tournament simulations** — win probabilities, group progression, golden boot projections
- **Live tracker** — enter real match results; compare predictions vs actuals with green/yellow/red badges
- **Streamlit dashboard** — 6 pages covering the full tournament lifecycle

## Project Structure

```
worldcup-2026-predictor/
├── notebooks/
│   ├── 01_data_collection.ipynb   # scrape data, compute team features
│   ├── 02_model_training.ipynb    # train XGBoost + Poisson, validate on 2022
│   ├── 03_simulation.ipynb        # 10,000 tournament Monte Carlo simulations
│   └── 04_live_tracker.ipynb      # compare predictions vs live results
├── data/                          # generated files (created by notebooks)
│   ├── matches_clean.csv
│   ├── team_features.csv
│   ├── groups_2026.csv
│   ├── xg_statsbomb.csv
│   ├── model.pkl
│   ├── simulation_results.csv
│   ├── golden_boot.csv
│   └── actual_results.csv         # populated as you add live results
├── app.py                         # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order

```bash
# Option A: execute directly
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_collection.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_training.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_simulation.ipynb

# Option B: open JupyterLab and run interactively
jupyter lab
```

> **Note**: Notebook 01 fetches data from GitHub. The StatsBomb xG extraction loops over ~128 match event files and takes 5-10 minutes.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Data Sources

| Source | Content |
|--------|---------|
| jfjelstul/worldcup | Matches, goals, bookings, team appearances 1930-2022 |
| StatsBomb Open Data | Shot-level xG events for 2018 & 2022 World Cups |
| FIFA Rankings (June 2026) | Hardcoded rankings for all 48 teams |

## Methodology

### Feature Engineering (per team)

| Feature | Description |
|---------|-------------|
| win_rate | Fraction of WC matches won (1994-2022) |
| avg_goals_for | Average goals scored per WC match |
| avg_goals_against | Average goals conceded |
| goal_diff | Average goal difference per match |
| avg_xgf | Average expected goals for (StatsBomb 2018+2022) |
| avg_xga | Average expected goals against |
| rank_score | 1 / FIFA_rank (higher = stronger) |

### Model Architecture

**XGBoost Classifier** (outcome: H/D/A)
- 300 estimators, max depth 4, learning rate 0.05
- Train: 1994-2018 World Cups | Validate: 2022 World Cup

**Poisson Regressors** (separate for home and away goals)
- Predicts lambda; actual scoreline drawn from Poisson(lambda)

### Tournament Simulation

1. Group stage: every team plays 3 matches; 3 pts/win, 1/draw
2. Top 2 per group (24) + 8 best 3rd-place = 32 teams advance
3. Knockouts: R32 -> R16 -> QF -> SF -> Final
4. Extra time + weighted penalty shootout for drawn knockouts

### Dark Horse Score

```
dark_horse_score = 0.4 * norm(win_rate)
                 + 0.3 * norm(avg_xgf)
                 - 0.3 * norm(rank_score)
```

## Team Name Standardisation

| Alias | Canonical Name |
|-------|---------------|
| USA | United States |
| Ivory Coast | Cote dIvoire |
| Korea Republic | South Korea |
| Bosnia-Herzegovina | Bosnia and Herzegovina |
| Turkiye | Turkey |
| IR Iran | Iran |
| Congo DR | DR Congo |
| Czechia | Czech Republic |

## 2026 World Cup Groups

| Group | Teams |
|-------|-------|
| A | United States, Panama, Bolivia, Yemen |
| B | Mexico, Ecuador, South Africa, DR Congo |
| C | Canada, Honduras, Algeria, Portugal |
| D | Brazil, Japan, Chile, Croatia |
| E | Spain, Netherlands, Ukraine, New Zealand |
| F | France, Colombia, Belgium, Cameroon |
| G | England, Serbia, Iran, Egypt |
| H | Argentina, Morocco, Nigeria, Venezuela |
| I | Germany, Australia, Senegal, Thailand |
| J | Switzerland, Italy, Denmark, Jamaica |
| K | Uruguay, Turkey, Czech Republic, Saudi Arabia |
| L | South Korea, Poland, Costa Rica, Slovenia |

## Dashboard Pages

| Page | Description |
|------|-------------|
| Tournament Overview | Win probability chart, golden boot top 10, dark horse cards |
| Pre-Match Predictor | Any two teams -> win %, scoreline, reasoning bullets |
| Live Tracker | Enter results, badges, accuracy/MAE metrics |
| Group Standings | Live standings from entered results |
| Player Cards | Key player stats + golden boot projections |
| Model Health | Feature importance, accuracy/MAE trend charts |

## Limitations

- No player-level injury/suspension data
- StatsBomb xG only available for 2018 & 2022 tournaments
- R32 bracket slots are shuffled rather than following exact FIFA seeding rules
- Host advantage not explicitly modelled beyond FIFA rankings
