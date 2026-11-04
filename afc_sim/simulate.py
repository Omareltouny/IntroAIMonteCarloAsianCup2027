"""Monte Carlo driver: groups -> best thirds -> knockout bracket -> champion."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import GROUPS, LAST_ROUND_PAIRS, MATCHDAYS, N_THIRD_ADVANCING, TEAMS
from .engine import MatchEngine
from .groupstage import rank_group, rank_thirds
from .knockout import ET_FACTOR, build_r16, play_bracket, play_knockout
from .scores import ScoreModel
from .shootouts import ShootoutModel

STAGES = ["Win group", "Advance", "QF", "SF", "Final", "Champion"]


@dataclass
class SimResult:
    table: pd.DataFrame          # per-team stage probabilities
    diagnostics: dict
    extras: dict | None = None   # raw counters / arrays for reporting (see Simulator.run)


class Simulator:
    def __init__(self, engine: MatchEngine, shootout: ShootoutModel, et_factor: float = ET_FACTOR,
                 scores: ScoreModel | None = None, groups: dict | None = None):
        self.engine, self.shootout, self.et_factor = engine, shootout, et_factor
        self.groups = groups or GROUPS          # default: the 2027 draw; other draws are used for back-tests
        self.names = engine.teams
        self.idx = engine.idx
        self.scores = scores or ScoreModel(engine.pair_probs)
        self.elo = engine.state.elo
        # fixed group fixtures as (group, team_i, team_j) index triples
        self.fixtures = []
        for g, ts in self.groups.items():
            for md in MATCHDAYS:
                for a, b in md:
                    self.fixtures.append((g, self.idx[ts[a - 1]], self.idx[ts[b - 1]]))
        self.fi = np.array([f[1] for f in self.fixtures])
        self.fj = np.array([f[2] for f in self.fixtures])

    def run(self, n_sims: int = 10_000, seed: int = 2027) -> SimResult:
        rng = np.random.default_rng(seed)
        nT = len(self.names)
        stage = np.zeros((nT, len(STAGES)))
        pos = np.zeros((nT, 4))                     # finishing position 1..4 in group
        third_adv = np.zeros(nT)
        ties, methods = Counter(), Counter()
        champions = np.zeros(n_sims, dtype=int)
        final_pairs, r16_pairs, ko_round_how = Counter(), Counter(), Counter()
        third_pts = Counter()                      # (points, gd, qualified) of every third-placed team
        pts_by_pos = np.zeros(4); goals = 0; draws = 0; n_group_games = 0
        champ_how = Counter()
        # draw every group-stage score for every simulation up front
        U = rng.random((n_sims, len(self.fixtures)))
        GI, GJ = self.scores.sample(self.fi[None, :], self.fj[None, :], U)

        group_slots = {g: [k for k, f in enumerate(self.fixtures) if f[0] == g] for g in self.groups}
        sw = lambda a, b: (a if rng.random() < self.shootout.p_win(
            self.names[a], self.names[b], self.elo.get(self.names[a], 1500.0), self.elo.get(self.names[b], 1500.0)) else b)
        play = lambda a, b: play_knockout(a, b, self.scores, self.shootout, self.elo, self.names, rng, self.et_factor)

        for s in range(n_sims):
            winners, runners, thirds_entries = {}, {}, []
            for g, ts in self.groups.items():
                tid = [self.idx[t] for t in ts]
                results = [(self.fixtures[k][1], self.fixtures[k][2], int(GI[s, k]), int(GJ[s, k]))
                           for k in group_slots[g]]
                last = {frozenset((self.idx[ts[a - 1]], self.idx[ts[b - 1]])) for a, b in MATCHDAYS[-1]}
                order, st = rank_group(tid, results, rng, last_round=last, shootout_winner=sw, tie_log=ties)
                for p, t in enumerate(order):
                    pos[t, p] += 1
                    pts_by_pos[p] += st["pts"][t]
                for _, _, x, y in results:
                    goals += x + y; draws += x == y; n_group_games += 1
                winners[g], runners[g] = order[0], order[1]
                t3 = order[2]
                thirds_entries.append((g, t3, st["pts"][t3], st["gd"][t3], st["gf"][t3]))
                stage[order[0], 0] += 1
            ranked = rank_thirds([(g, p, d, f, t) for g, t, p, d, f in thirds_entries], rng)
            qual = ranked[:N_THIRD_ADVANCING]
            thirds = {e[0]: e[4] for e in qual}
            for e in qual:
                third_adv[e[4]] += 1
            qset = {e[4] for e in qual}
            for e in ranked:
                third_pts[(e[1], e[2], e[4] in qset)] += 1
            for t in list(winners.values()) + list(runners.values()) + list(thirds.values()):
                stage[t, 1] += 1
            r16 = build_r16(winners, runners, thirds)
            res = play_bracket(r16, play)
            for m, (a, b, w, how) in res.items():
                methods[how] += 1
                rnd = "R16" if m < 45 else "QF" if m < 49 else "SF" if m < 51 else "Final"
                ko_round_how[(rnd, how)] += 1
                if m < 45:
                    r16_pairs[frozenset((self.names[a], self.names[b]))] += 1
                if 45 <= m <= 48:
                    stage[a, 2] += 1; stage[b, 2] += 1
                elif m in (49, 50):
                    stage[a, 3] += 1; stage[b, 3] += 1
                elif m == 51:
                    stage[a, 4] += 1; stage[b, 4] += 1; stage[w, 5] += 1
                    champions[s] = w; champ_how[how] += 1
                    final_pairs[frozenset((self.names[a], self.names[b]))] += 1
        # QF/SF/Final columns above count *appearances*: a team appears in QF = won its R16, etc.
        tab = pd.DataFrame(stage / n_sims, index=self.names, columns=STAGES)
        tab.insert(0, "Group", [next(g for g, ts in self.groups.items() if t in ts) for t in self.names])
        tab.insert(1, "Elo", [round(self.elo.get(t, 1500.0)) for t in self.names])
        for p in range(4):
            tab[f"Pos{p + 1}"] = pos[:, p] / n_sims
        tab["Adv as 3rd"] = third_adv / n_sims
        tab = tab.sort_values("Champion", ascending=False)
        diag = dict(n_sims=n_sims, tiebreak_resolution=dict(ties),
                    tiebreak_per_sim=sum(ties.values()) / n_sims,
                    knockout_decided_in={k: v / sum(methods.values()) for k, v in methods.items()},
                    mean_fit_error=float(np.nanmean(self.scores.fit_error)),
                    max_fit_error=float(np.nanmax(self.scores.fit_error)))
        extras = dict(champions=champions, names=list(self.names), final_pairs=final_pairs, r16_pairs=r16_pairs,
                      ko_round_how=ko_round_how, third_pts=third_pts, champ_how=champ_how,
                      pts_by_pos=pts_by_pos / (n_sims * len(self.groups)), goals_per_group_game=goals / n_group_games,
                      group_draw_rate=draws / n_group_games, tiebreak_resolution=dict(ties))
        return SimResult(tab, diag, extras)
