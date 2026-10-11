# Machine Learning-Driven Probabilistic Forecasting and Monte Carlo Simulation of the 2027 AFC Asian Cup

> CSCI-630.01 · Omar Elouny
> **Status: in progress.** Model testing is complete. The Monte Carlo tournament simulation is next.

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
| 6 | Monte Carlo simulation of the 2027 AFC Asian Cup (groups, tiebreakers, 3rd-place pool, knockout bracket) | Next |
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
pip install pandas numpy scipy scikit-learn xgboost matplotlib jupyter
# download the Kaggle dataset into ./data/
jupyter notebook afc_asian_cup_model_testing.ipynb
```

The full notebook runs in about 1–2 minutes on a laptop CPU.

## Next steps

- [ ] Encode the 24-team AFC Asian Cup format: six groups of four, the best four third-placed teams advance, and the official tiebreakers.
- [ ] Add a score-generation layer (Poisson goal model) so goal difference tiebreakers can be simulated.
- [ ] Handle knockout draws with extra time and penalties, using the shootout history.
- [ ] Run 10,000 tournament simulations and report group survival, stage progression and championship probabilities.
- [ ] Retrain the selected model on all data through 2026 before the final simulation.

## Author

Omar Elouny, CSCI-630.01
