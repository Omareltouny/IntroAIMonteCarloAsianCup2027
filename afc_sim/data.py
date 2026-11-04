"""Data loading and the Elo / feature pipeline, ported from the modelling notebook.

The logic mirrors ``afc_asian_cup_model_testing.ipynb`` (sections 1-2) so features
are identical, but it also returns the *end-of-history state* (current Elo, rolling
form, matches played) needed to build features for future fixtures.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HOME_ADV = 100

FEATS = ["elo_diff", "elo_home", "elo_away", "form_diff", "form_pts_home", "form_pts_away",
         "atk_def_home", "atk_def_away", "rest_diff", "host_adv", "neutral_i", "K", "is_friendly",
         "exp_home", "exp_away"]


def k_weight(t: str) -> int:
    t = t.lower()
    if t == "fifa world cup":
        return 60
    if "qualification" in t or "nations league" in t:
        return 40
    if t == "friendly":
        return 20
    if any(s in t for s in ["uefa euro", "copa américa", "african cup of nations", "afc asian cup", "gold cup",
                            "confederations cup", "oceania nations cup", "concacaf championship", "conmebol"]):
        return 50
    return 30


def gm_mult(d: int) -> float:
    d = abs(d)
    return 1.0 if d <= 1 else (1.5 if d == 2 else (11 + d) / 8)


def _apply_former_names(df: pd.DataFrame, fn: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for _, r in fn.iterrows():
        m = (df.date >= r.start_date) & (df.date <= r.end_date)
        for c in cols:
            df.loc[m & (df[c] == r.former), c] = r.current
    return df


def load_raw(data_dir: str | Path = "data"):
    """Return (results, shootouts) with successor-name mapping applied."""
    d = Path(data_dir)
    res = pd.read_csv(d / "results.csv", parse_dates=["date"])
    sho = pd.read_csv(d / "shootouts.csv", parse_dates=["date"])
    fn = pd.read_csv(d / "former_names.csv", parse_dates=["start_date", "end_date"])
    res = res.dropna(subset=["home_score", "away_score"]).sort_values("date", kind="stable").reset_index(drop=True)
    res["neutral"] = res["neutral"].astype(str).str.upper().eq("TRUE")
    res = _apply_former_names(res, fn, ["home_team", "away_team"])
    sho = _apply_former_names(sho, fn, ["home_team", "away_team", "winner"])
    return res, sho


@dataclass
class TeamState:
    """Everything needed to build pre-match features for a team *today*."""
    elo: dict = field(default_factory=dict)
    pts: dict = field(default_factory=dict)      # last 5 points
    gf: dict = field(default_factory=dict)       # last 10 goals for
    ga: dict = field(default_factory=dict)       # last 10 goals against
    n_played: dict = field(default_factory=dict)

    def form(self, team):
        p = np.mean(self.pts[team]) if self.pts.get(team) else 1.0
        f = np.mean(self.gf[team]) if self.gf.get(team) else 1.3
        a = np.mean(self.ga[team]) if self.ga.get(team) else 1.3
        return p, f, a


def build_features(res: pd.DataFrame):
    """Chronological Elo + rolling-form features (no leakage). Returns (df, TeamState)."""
    res = res.copy()
    res["K"] = res.tournament.map(k_weight)
    elo = defaultdict(lambda: 1500.0)
    last_date, n_played = {}, defaultdict(int)
    pts = defaultdict(lambda: deque(maxlen=5))
    gf = defaultdict(lambda: deque(maxlen=10))
    ga = defaultdict(lambda: deque(maxlen=10))
    rows = []

    def form(t):
        p = np.mean(pts[t]) if pts[t] else 1.0
        f = np.mean(gf[t]) if gf[t] else 1.3
        a = np.mean(ga[t]) if ga[t] else 1.3
        return p, f, a

    for r in res.itertuples(index=False):
        h, a = r.home_team, r.away_team
        eh, ea = elo[h], elo[a]
        ph, fh, ah = form(h)
        pa, fa, aa = form(a)
        rest_h = min((r.date - last_date[h]).days, 60) if h in last_date else 60
        rest_a = min((r.date - last_date[a]).days, 60) if a in last_date else 60
        host = 1 if (not r.neutral) or r.country == h else (-1 if r.country == a else 0)
        rows.append((eh, ea, ph, pa, fh, fa, ah, aa, rest_h, rest_a, host,
                     np.log1p(n_played[h]), np.log1p(n_played[a])))
        adv = 0 if r.neutral else HOME_ADV
        exp_h = 1 / (1 + 10 ** (-((eh + adv) - ea) / 400))
        S = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        delta = r.K * gm_mult(r.home_score - r.away_score) * (S - exp_h)
        elo[h] += delta
        elo[a] -= delta
        for t, gfor, gag in [(h, r.home_score, r.away_score), (a, r.away_score, r.home_score)]:
            pts[t].append(3 if gfor > gag else (1 if gfor == gag else 0))
            gf[t].append(gfor)
            ga[t].append(gag)
            last_date[t] = r.date
            n_played[t] += 1

    cols = ["elo_home", "elo_away", "form_pts_home", "form_pts_away", "gf10_home", "gf10_away",
            "ga10_home", "ga10_away", "rest_home", "rest_away", "host_adv", "exp_home", "exp_away"]
    F = pd.DataFrame(rows, columns=cols)
    df = pd.concat([res, F], axis=1)
    df["elo_diff"] = df.elo_home - df.elo_away
    df["form_diff"] = df.form_pts_home - df.form_pts_away
    df["atk_def_home"] = df.gf10_home - df.ga10_away
    df["atk_def_away"] = df.gf10_away - df.ga10_home
    df["rest_diff"] = df.rest_home - df.rest_away
    df["neutral_i"] = df.neutral.astype(int)
    df["is_friendly"] = (df.tournament == "Friendly").astype(int)
    df["y"] = np.where(df.home_score > df.away_score, 0, np.where(df.home_score == df.away_score, 1, 2))
    state = TeamState(elo=dict(elo), pts={k: list(v) for k, v in pts.items()},
                      gf={k: list(v) for k, v in gf.items()}, ga={k: list(v) for k, v in ga.items()},
                      n_played=dict(n_played))
    return df, state
