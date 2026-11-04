"""Match engine: XGBoost W/D/L model + per-pairing probability tables for the 24 teams.

``MatchEngine.pair_probs[i, j]`` is P(home win, draw, away win) with team ``i`` as the
nominal home side.  At neutral venues the table is symmetrised (the average of both
orderings) so the arbitrary home/away label cannot matter.  Saudi Arabia, the host,
is treated as the genuine home team (non-neutral, ``host_adv = 1``) in every match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import HOST, TEAMS
from .data import FEATS, TeamState

# Hyper-parameters selected on the validation set in the notebook (depth 2, mcw 20,
# 487 trees at lr 0.03).  Early stopping is replaced by this fixed tree count so the
# model can be refit on more data.
XGB_PARAMS = dict(objective="multi:softprob", n_estimators=487, learning_rate=0.03, max_depth=2,
                  min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                  eval_metric="mlogloss", n_jobs=2)
TOURNAMENT_K = 50  # "AFC Asian Cup" continental-finals weight in the Elo pipeline


def train_xgb(df: pd.DataFrame, train_through_year: int = 2026, seed: int = 42) -> xgb.XGBClassifier:
    tr = df[df.date.dt.year <= train_through_year]
    m = xgb.XGBClassifier(random_state=seed, **XGB_PARAMS)
    m.fit(tr[FEATS].values, tr.y.values, verbose=False)
    return m


def fixture_features(h: str, a: str, state: TeamState, *, home_is_host: bool) -> dict:
    """Pre-match feature row for a (future) Asian Cup match, built from end-of-history state."""
    eh, ea = state.elo.get(h, 1500.0), state.elo.get(a, 1500.0)
    ph, fh, ah = state.form(h)
    pa, fa, aa = state.form(a)
    return dict(
        elo_diff=eh - ea, elo_home=eh, elo_away=ea, form_diff=ph - pa, form_pts_home=ph, form_pts_away=pa,
        atk_def_home=fh - aa, atk_def_away=fa - ah, rest_diff=0.0,
        host_adv=1 if home_is_host else 0, neutral_i=0 if home_is_host else 1,
        K=TOURNAMENT_K, is_friendly=0,
        exp_home=np.log1p(state.n_played.get(h, 0)), exp_away=np.log1p(state.n_played.get(a, 0)))


class MatchEngine:
    def __init__(self, model: xgb.XGBClassifier, state: TeamState, teams: list[str] = TEAMS,
                 host_advantage: bool = True, host: str = HOST):
        self.model, self.state, self.teams = model, state, list(teams)
        self.host = host
        self.host_advantage = host_advantage      # False: treat Saudi Arabia like every other (neutral) team
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.pair_probs = self._build_table()

    def _build_table(self) -> np.ndarray:
        n = len(self.teams)
        rows, keys = [], []
        for i, h in enumerate(self.teams):
            for j, a in enumerate(self.teams):
                if i == j:
                    continue
                if a == self.host and self.host_advantage:       # host always listed as the home side
                    continue
                rows.append(fixture_features(h, a, self.state, home_is_host=(h == self.host and self.host_advantage)))
                keys.append((i, j))
        P = self.model.predict_proba(pd.DataFrame(rows)[FEATS].values)
        raw = np.full((n, n, 3), np.nan)
        for (i, j), p in zip(keys, P):
            raw[i, j] = p
        out = np.full((n, n, 3), np.nan)
        host = self.idx.get(self.host) if self.host_advantage else None
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if i == host:
                    out[i, j] = raw[i, j]
                elif j == host:
                    out[i, j] = raw[j, i][::-1]          # host as home, flipped to (i home, j away)
                else:                                     # neutral: symmetrise the nominal home label
                    out[i, j] = 0.5 * (raw[i, j] + raw[j, i][::-1])
        return out

    def probs(self, home: str, away: str) -> np.ndarray:
        return self.pair_probs[self.idx[home], self.idx[away]]
