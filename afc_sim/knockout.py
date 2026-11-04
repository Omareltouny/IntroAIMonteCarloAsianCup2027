"""Knockout rounds: bracket construction, extra time and penalty shoot-outs.

A match level after 90 minutes goes to 30 minutes of extra time.  Each side scores
Poisson(lam * ET_FACTOR / 3) extra-time goals, where ``lam`` is the side's fitted 90-minute
rate from the score layer (so stronger sides also score more in extra time).  If still level,
the winner is drawn from the shoot-out model (Elo gap + shrunk shoot-out record).

ET_FACTOR is the extra-time scoring intensity relative to the regulation per-minute rate.
The goalscorers file is too sparse to pin it down: a conditional-likelihood estimate from
the shoot-out games with recorded goals gives roughly 0.7 (likely biased low because
goals are missing from the file), so 0.8 is used and varied in a sensitivity test.
"""
from __future__ import annotations

import numpy as np

from .config import ALLOWED_THIRD, FINAL, QF, R16, SF, THIRD_PLACE_TABLE

ET_FACTOR = 0.8


def build_r16(winners: dict, runners: dict, thirds: dict) -> dict:
    """Map R16 match number -> (team_a, team_b).

    ``winners`` / ``runners``: group letter -> team.  ``thirds``: group letter -> team for the
    four qualifying third-placed teams (keys decide the row of the official mapping table).
    """
    row = THIRD_PLACE_TABLE[frozenset(thirds)]
    pick = {"W": winners, "R": runners}
    out = {}
    for m, (sa, sb) in R16.items():
        teams = []
        for kind, g in (sa, sb):
            if kind == "T":
                tg = row[g]
                assert tg in ALLOWED_THIRD[g], (g, tg)
                teams.append(thirds[tg])
            else:
                teams.append(pick[kind][g])
        out[m] = tuple(teams)
    return out


def play_knockout(i, j, scores, shootout, elo, names, rng: np.random.Generator,
                  et_factor: float = ET_FACTOR):
    """Play one knockout tie between team indices i and j.

    Returns (winner_index, decided_in) with decided_in in {"90", "ET", "PEN"}.
    """
    gi, gj = scores.sample(i, j, rng.random())
    if gi != gj:
        return (i if gi > gj else j), "90"
    li, lj = scores.lam[i, j] * et_factor / 3.0
    ei, ej = rng.poisson(li), rng.poisson(lj)
    if ei != ej:
        return (i if ei > ej else j), "ET"
    a, b = names[i], names[j]
    p = shootout.p_win(a, b, elo.get(a, 1500.0), elo.get(b, 1500.0))
    return (i if rng.random() < p else j), "PEN"


def play_bracket(r16: dict, play) -> dict:
    """Run R16 -> final. ``play(i, j) -> (winner, method)``; returns detail per match number."""
    res, win = {}, {}
    for m, (a, b) in r16.items():
        w, how = play(a, b)
        win[m], res[m] = w, (a, b, w, how)
    for rnd in (QF, SF, FINAL):
        for m, (f1, f2) in rnd.items():
            a, b = win[f1], win[f2]
            w, how = play(a, b)
            win[m], res[m] = w, (a, b, w, how)
    return res
