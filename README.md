# Machine Learning-Driven Probabilistic Forecasting and Monte Carlo Simulation of the 2027 AFC Asian Cup

> CSCI-630.01 · Omar Elouny
> **Status: in progress.** Model testing is complete. The tournament engine (24-team format, score layer, extra time and penalties) is implemented and tested; the final 10,000-run report is next.

## Overview

Predicting international tournament outcomes is hard. Team strength changes over time, head-to-head records are sparse, tournament rules are complex, and single-elimination knockouts are highly random.

This project builds a predictive system trained on international match data (1872–2026). It predicts match-level outcomes (goals scored and win/draw/loss probabilities), then plugs the best-calibrated model into a **10,000-run Monte Carlo simulation** of the **2027 AFC Asian Cup in Saudi Arabia**. The simulation estimates each team's group survival, knockout progression and title probabilities.

## Project pipeline

```
results.csv ──► Elo ratings (1872–2026) ──► Match features ──► Chronological split
                                                                      │
        Monte Carlo (10,000 runs) ◄── Best-calibrated model ◄── Train / compare 3 models
        (24-team format)                                          (log loss, Brier, calibration)
```

| Step | Description | Status |
|---|---|---|
| 1 | Data cleaning and former-name (successor) mapping | Done |
| 2 | Elo ratings with tournament weighting and goal-margin multipliers | Done |
| 3 | Feature engineering (Elo difference, rolling form, rest days, host advantage) | Done |
| 4 | Model training and evaluation (Poisson GLM, Logistic Regression, XGBoost, MLP) | Done |
| 5 | Select the best-calibrated model as the match simulation engine | Done (XGBoost) |
| 6 | Tournament engine: groups, tiebreakers, 3rd-place pool, knockout bracket, score layer, ET and penalties | Done (`afc_sim/`) |
| 6b | Final 10,000-run simulation and reporting | Next |
| 7 | Final report and visualisations | Planned |

## Dataset

[International Football Results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) (Kaggle, Mart Jürisoo). About 49,900 matches.

Files used: `results.csv`, `former_names.csv`. `shootouts.csv` and `goalscorers.csv` are included but not yet used. Place them in a `data/` folder.

## Methodology

**Elo ratings**
- Every team starts at 1500, and the home side gets a +100 bonus on non-neutral ground.
- The update size (K) depends on the competition: World Cup 60, continental finals 50, qualifiers and Nations League 40, other competitive 30, friendlies 20.
- Updates are scaled by the goal margin using the World Football Elo multiplier.
- All features are computed *before* each match is played, so there is no leakage.

**Features**
- Elo ratings and their difference.
- Rolling form: points per game over the last 5 matches, goals for and against over the last 10.
- Rest days, host advantage, neutral venue, friendly flag, and tournament weight.

**Chronological split (full history, 1872+)**

| Set | Period | Matches |
|---|---|---|
| Train | ≤ 2019 | 43,406 |
| Validation | 2020–2022 | 2,441 |
| Test | 2023+ | 4,098 |

**Models**

1. **Poisson GLM**: predicts home and away goals, converted to W/D/L through independent Poisson score grids.
2. **Logistic Regression**: multinomial baseline.
3. **XGBoost**: gradient-boosted trees, with depth and early stopping tuned on the validation set.
4. **MLP**: a feed-forward neural network, with architecture and L2 strength tuned on the validation set.

**Evaluation:** multi-class log loss, Brier score, ranked probability score (RPS), accuracy, and calibration curves with expected calibration error (ECE). The simulation engine is chosen on the validation set and then confirmed on the untouched test set.

## Results so far

Test set, 4,098 matches from 2023 onward (lower is better, except accuracy):

| Model | Log loss | Brier | RPS | Accuracy | ECE |
|---|---|---|---|---|---|
| Class-prior reference (no skill) | 1.054 | 0.636 | 0.230 | 47.2% | n/a |
| Poisson GLM | 0.880 | 0.517 | 0.173 | 60.2% | 0.033 |
| Logistic Regression | 0.867 | 0.509 | 0.170 | 60.4% | 0.026 |
| **XGBoost** | **0.866** | 0.509 | 0.170 | 60.6% | **0.021** |
| MLP | 0.867 | 0.508 | 0.169 | **60.8%** | 0.021 |

**Takeaways**
- **XGBoost** is the selected simulation engine. It was best on validation log loss, Brier score and calibration, and it also had the best test log loss and calibration.
- The gaps between Logistic Regression, XGBoost and the MLP are within about one standard error, so they are close to statistically tied. Most of the predictive signal comes from Elo.
- The Poisson GLM is clearly the weakest. It over-predicts home goals on the test set (1.89 predicted against 1.63 actual). It remains useful for goal predictions.
- On the 1,063 test matches involving an AFC team, the model ranking is broadly the same. The 51 matches from the 2023 Asian Cup alone are too few to separate the models.

Full tables are in [`model_results.csv`](model_results.csv), and calibration plots are in the notebook.

## Repository structure

```
.
├── README.md
├── afc_asian_cup_model_testing.ipynb   # Elo, features, models, evaluation
├── model_results.csv                   # Validation and test metrics
├── run_simulation.py                   # End-to-end: data -> model -> N tournament runs
├── simulation_results.csv              # Preliminary 10,000-run output (see below)
├── afc_sim/
│   ├── config.py      # groups, R16/QF/SF bracket, 15-row third-place table
│   ├── data.py        # Elo + feature pipeline ported from the notebook
│   ├── engine.py      # XGBoost W/D/L tables for all 24 teams (host advantage, neutral symmetrisation)
│   ├── scores.py      # Poisson score layer
│   ├── groupstage.py  # standings, tiebreak cascade, third-place ranking
│   ├── shootouts.py   # penalty shoot-out model fitted on shootouts.csv
│   ├── knockout.py    # bracket construction, extra time, penalties
│   └── simulate.py    # Monte Carlo driver
├── tests/             # 50 unit and end-to-end tests (pytest)
└── data/
    ├── results.csv
    ├── former_names.csv
    ├── shootouts.csv
    └── goalscorers.csv
```

## Getting started

```bash
git clone https://github.com/Omareltouny/IntroAIMonteCarloAsianCup2027.git
cd IntroAIMonteCarloAsianCup2027
pip install pandas numpy scipy scikit-learn xgboost matplotlib jupyter pytest
# download the Kaggle dataset into ./data/
jupyter notebook afc_asian_cup_model_testing.ipynb
```

The full notebook runs in about 1–2 minutes on a laptop CPU.

```bash
python run_simulation.py --sims 10000      # about 20 s
python -m pytest tests -q                  # 50 tests
```

## Tournament engine (`afc_sim/`)

**Format.** Six groups of four (A–F, drawn 9 May 2026). The top two plus the four best third-placed teams reach the Round of 16. Group winners meet third-placed teams according to the official 15-row table keyed on which four groups' thirds qualify (`THIRD_PLACE_TABLE`; each row is checked against the bracket's slot restrictions in the tests). Bracket: R16 matches 37–44, QF 45–48 (W37–W39, W38–W41, W44–W43, W40–W42), SF 49–50 (W45–W46, W47–W48), final 51, no third-place match.

**Tiebreakers** (AFC regulations as published for 2023): head-to-head points, goal difference, goals scored among the tied teams; the same criteria re-applied to any subset still tied; overall goal difference; overall goals scored; a penalty shoot-out if exactly two teams are tied and met in the last round; then disciplinary points and lots. Disciplinary points cannot be simulated, so ties that survive to that point go to lots. Third-placed teams are ranked on points, goal difference, goals scored, then lots.

**Score layer.** For each ordered pairing, independent-Poisson rates are fitted so the implied win/draw/loss probabilities match XGBoost (fit error below 1e-6). The joint score distribution is then reweighted by outcome class so the W/D/L marginals equal XGBoost's exactly, so the classifier decides who wins and the Poisson shape decides the margin. Sampled scores feed goal-difference and goals-scored tiebreakers.

**Extra time and penalties.** A knockout tie level after 90 minutes gets 30 minutes of extra time with each side scoring Poisson(rate × 0.8 / 3). If still level, a symmetric logistic shoot-out model decides it, using the Elo gap and each team's shrunk prior shoot-out win rate, fitted on the ~700 shoot-outs in `shootouts.csv`. It is only slightly better than a coin flip on a 2015+ holdout (log loss 0.685 vs 0.693), which matches the literature's view that shoot-outs are close to random.

**Assumptions to check.**
- Tiebreakers are the 2023 regulations; the 2027 wiki page did not list them.
- The group matchday order was not on the wiki page. The FIFA four-team template is assumed; it only affects the rare last-round shoot-out tiebreak.
- Saudi Arabia plays every match as the genuine home side; all other matches are neutral and symmetrised.
- Ratings are frozen at the last result in the data (6 Oct 2026) and are not updated during the tournament.
- The extra-time intensity of 0.8 is a rough estimate (goalscorer data are too sparse); varying it from 0.4 to 1.2 moves title probabilities by at most about 1.4 percentage points.
- The model sends about 14% of knockout ties to penalties, against 16 of roughly 62 knockout games (26%) in Asian Cups since 2004. The historical sample is small, but it hints that knockout draws may be slightly under-modelled.

## Next steps

- [x] Encode the 24-team AFC Asian Cup format: six groups of four, the best four third-placed teams advance, and the official tiebreakers.
- [x] Add a score-generation layer (Poisson goal model) so goal difference tiebreakers can be simulated.
- [x] Handle knockout draws with extra time and penalties, using the shootout history.
- [ ] Run 10,000 tournament simulations and report group survival, stage progression and championship probabilities.
- [x] Retrain the selected model on all data through 2026 before the final simulation (`train_xgb(df, 2026)`, same tuned hyper-parameters, 487 trees).

## Author

Omar Elouny, CSCI-630.01
