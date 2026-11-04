"""Group-stage table, official tiebreakers and the best-third-placed ranking.

Tiebreakers (AFC Asian Cup regulations as published for the 2023 edition; the 2027 wiki
page did not list them, so this assumes the 2027 edition uses the same rules):

 1. points in head-to-head matches among the tied teams
 2. goal difference in those matches
 3. goals scored in those matches
 4. if more than two teams were tied and a subset is still tied, re-apply 1-3 to the subset only
 5. goal difference in all group matches
 6. goals scored in all group matches
 7. penalty shoot-out, if exactly two teams are tied and they met in the last round
 8. disciplinary points  -- NOT simulated (no card model); folded into 9
 9. drawing of lots

The four best third-placed teams are ranked on points, goal difference, goals scored,
then lots (disciplinary points again cannot be simulated).
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

import numpy as np

# result tuple: (team_a, team_b, goals_a, goals_b) -- teams are any hashable ids
Result = tuple


def _table(teams, results):
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    for a, b, x, y in results:
        gf[a] += x; ga[a] += y; gf[b] += y; ga[b] += x
        if x > y:
            pts[a] += 3
        elif x < y:
            pts[b] += 3
        else:
            pts[a] += 1; pts[b] += 1
    return pts, gf, ga


def _partition(items, key):
    """Sort descending by ``key`` and split into blocks of equal key."""
    items = sorted(items, key=key, reverse=True)
    blocks, last = [], object()
    for t in items:
        k = key(t)
        if blocks and k == last:
            blocks[-1].append(t)
        else:
            blocks.append([t])
        last = k
    return blocks


def rank_group(teams, results, rng: np.random.Generator, *,
               last_round: Optional[set] = None,
               shootout_winner: Optional[Callable] = None,
               tie_log: Optional[Counter] = None):
    """Return teams ordered 1st..last, applying the tiebreak cascade above.

    ``last_round``: set of frozenset({a, b}) pairs played on the final matchday.
    ``shootout_winner(a, b)``: callable returning the winner of a shoot-out (rule 7).
    ``tie_log``: optional Counter recording which rule finally split each tie.
    """
    pts, gf, ga = _table(teams, results)
    gd = {t: gf[t] - ga[t] for t in teams}

    def log(name):
        if tie_log is not None:
            tie_log[name] += 1

    def final_ties(block):                       # rules 7-9
        if len(block) == 1:
            return block
        if (len(block) == 2 and last_round is not None and shootout_winner is not None
                and frozenset(block) in last_round):
            w = shootout_winner(block[0], block[1])
            log("shootout")
            return [w, block[1] if w == block[0] else block[0]]
        log("lots")
        block = list(block)
        rng.shuffle(block)
        return block

    def resolve(block):                          # rules 1-6 for one set of teams tied on points
        if len(block) == 1:
            return block
        sub = [r for r in results if r[0] in block and r[1] in block]
        hp, hf, ha = _table(block, sub)
        parts = _partition(block, lambda t: (hp[t], hf[t] - ha[t], hf[t]))
        if len(parts) > 1:                       # head-to-head separated at least one team
            log("head_to_head")
            out = []
            for p in parts:
                out += resolve(p) if len(p) > 1 else p      # rule 4 on the remaining subset
            return out
        parts = _partition(block, lambda t: (gd[t], gf[t]))  # rules 5-6
        if len(parts) > 1:
            log("overall_gd_gf")
        out = []
        for p in parts:
            out += final_ties(p)
        return out

    ordered = []
    for blk in _partition(teams, lambda t: pts[t]):
        ordered += resolve(blk)
    return ordered, {"pts": pts, "gf": gf, "ga": ga, "gd": gd}


def rank_thirds(entries, rng: np.random.Generator):
    """Rank third-placed teams. ``entries`` = list of (group, pts, gd, gf); best first."""
    noise = rng.random(len(entries))
    order = sorted(range(len(entries)), key=lambda i: (entries[i][1], entries[i][2], entries[i][3], noise[i]),
                   reverse=True)
    return [entries[i] for i in order]
