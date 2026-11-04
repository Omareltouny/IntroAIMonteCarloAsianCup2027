"""Penalty shoot-out model fitted on ``shootouts.csv`` (about 700 shoot-outs since 1967).

logit P(A beats B) = b_elo * (Elo_A - Elo_B)/100  +  b_rec * (rate_A - rate_B)

* ``Elo`` is each side's pre-match rating from the Elo pipeline.
* ``rate`` is the team's *prior* shoot-out win rate, shrunk towards 0.5 with ``prior_n``
  pseudo-shoot-outs (Beta prior), computed only from shoot-outs played before the match.
* There is no intercept, so the model is symmetric (it cannot favour the listed home side).
* The first-shooter edge is excluded: the order is decided by a coin toss, so it averages
  out when we do not know who kicks first (reported in ``diagnostics`` for reference).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def build_shootout_table(df_feat: pd.DataFrame, sho: pd.DataFrame, prior_n: float = 8.0) -> pd.DataFrame:
    """Join shoot-outs to pre-match Elo and add leakage-free prior shoot-out records."""
    keys = ["date", "home_team", "away_team"]
    m = sho.merge(df_feat[keys + ["elo_home", "elo_away"]].drop_duplicates(keys), on=keys, how="left")
    m = m.dropna(subset=["elo_home", "elo_away"]).sort_values("date", kind="stable").reset_index(drop=True)
    w, n = defaultdict(float), defaultdict(float)
    rh, ra = [], []
    for r in m.itertuples(index=False):
        rh.append((w[r.home_team] + prior_n / 2) / (n[r.home_team] + prior_n))
        ra.append((w[r.away_team] + prior_n / 2) / (n[r.away_team] + prior_n))
        for t in (r.home_team, r.away_team):
            n[t] += 1
        w[r.winner] += 1
    m["rate_home"], m["rate_away"] = rh, ra
    m["y"] = (m.winner == m.home_team).astype(int)
    m["x_elo"] = (m.elo_home - m.elo_away) / 100
    m["x_rec"] = m.rate_home - m.rate_away
    return m


def _fit_logit(X, y, C=1.0):
    # symmetric augmentation: every shoot-out also appears with the sides swapped
    Xa, ya = np.r_[X, -X], np.r_[y, 1 - y]
    return LogisticRegression(fit_intercept=False, C=C).fit(Xa, ya)


class ShootoutModel:
    def __init__(self, b_elo: float, b_rec: float, rate: dict, prior_n: float, diagnostics: dict):
        self.b_elo, self.b_rec, self.rate, self.prior_n = b_elo, b_rec, rate, prior_n
        self.diagnostics = diagnostics

    def team_rate(self, team: str) -> float:
        return self.rate.get(team, 0.5)

    def p_win(self, a: str, b: str, elo_a: float, elo_b: float) -> float:
        z = self.b_elo * (elo_a - elo_b) / 100 + self.b_rec * (self.team_rate(a) - self.team_rate(b))
        return float(_sigmoid(z))

    @classmethod
    def fit(cls, df_feat: pd.DataFrame, sho: pd.DataFrame, prior_n: float = 8.0, holdout_from: int = 2015):
        t = build_shootout_table(df_feat, sho, prior_n)
        X = t[["x_elo", "x_rec"]].values
        y = t.y.values
        # chronological check: fit before `holdout_from`, score after, versus a fair coin
        tr, te = (t.date.dt.year < holdout_from).values, (t.date.dt.year >= holdout_from).values
        mm = _fit_logit(X[tr], y[tr])
        p = np.clip(_sigmoid(X[te] @ mm.coef_[0]), 1e-6, 1 - 1e-6)
        ll = float(-np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)))
        full = _fit_logit(X, y)
        # final per-team prior rates over the whole history
        w, n = defaultdict(float), defaultdict(float)
        for r in t.itertuples(index=False):
            for tm in (r.home_team, r.away_team):
                n[tm] += 1
            w[r.winner] += 1
        rate = {k: (w[k] + prior_n / 2) / (n[k] + prior_n) for k in n}
        fs = sho.dropna(subset=["first_shooter"])
        diag = dict(n_shootouts=len(t), coef_elo_per100=float(full.coef_[0][0]), coef_record=float(full.coef_[0][1]),
                    holdout_from=holdout_from, n_holdout=int(te.sum()), holdout_logloss=ll,
                    coin_logloss=float(np.log(2)),
                    first_shooter_win_rate=float((fs.first_shooter == fs.winner).mean()), n_first_shooter=len(fs))
        return cls(float(full.coef_[0][0]), float(full.coef_[0][1]), rate, prior_n, diag)
