"""Validation: (1) replay the real 2023 Asian Cup group stage through the format code,
(2) back-test the full pipeline on the 2023 Asian Cup using only pre-tournament data,
(3) check whether defending champions beat their Elo expectation (the "Qatar question").

    python validation.py            # about 1 minute; writes results/validation.json, results/backtest_2023.csv,
                                    # figures/10_backtest_2023.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from afc_sim import MatchEngine, ShootoutModel, Simulator, build_features, load_raw, train_xgb
from afc_sim.groupstage import rank_group, rank_thirds
from afc_sim.knockout import build_r16

DATA, OUT, FIG = "data", Path("results"), Path("figures")
START, END, KICKOFF = "2024-01-12", "2024-02-10", pd.Timestamp("2024-01-12")   # 2023 AFC Asian Cup (Qatar)
# 2023 draw (letters matter: they fix the bracket). The code checks these sets against the results file.
GROUPS_2023 = {
    "A": ["Qatar", "China", "Tajikistan", "Lebanon"],
    "B": ["Australia", "Uzbekistan", "Syria", "India"],
    "C": ["Iran", "United Arab Emirates", "Hong Kong", "Palestine"],
    "D": ["Japan", "Indonesia", "Iraq", "Vietnam"],
    "E": ["South Korea", "Malaysia", "Jordan", "Bahrain"],
    "F": ["Saudi Arabia", "Thailand", "Oman", "Kyrgyzstan"],
}
HOST_2023 = "Qatar"


def winner_of(m, sho):
    if m.home_score != m.away_score:
        return m.home_team if m.home_score > m.away_score else m.away_team
    s = sho[(sho.date == m.date) & (sho.home_team == m.home_team) & (sho.away_team == m.away_team)]
    return s.winner.iloc[0]


def load_2023(res, sho):
    t = res[(res.tournament == "AFC Asian Cup") & (res.date >= START) & (res.date <= END)].sort_values("date", kind="stable")
    assert len(t) == 51, len(t)
    grp, ko = t.iloc[:36], t.iloc[36:]
    for ts in GROUPS_2023.values():                         # groups must match who actually played whom
        assert {frozenset((a, b)) for a, b in zip(grp.home_team, grp.away_team) if a in ts} == \
               {frozenset((a, b)) for i, a in enumerate(ts) for b in ts[i + 1:]}
    return grp, ko


# ------------------------------------------------------------------ (1) replay of the real group stage
def replay(grp, ko, sho):
    rng = np.random.default_rng(0)
    W, R, thirds = {}, {}, []
    for g, ts in GROUPS_2023.items():
        res = [(r.home_team, r.away_team, int(r.home_score), int(r.away_score))
               for r in grp.itertuples() if r.home_team in ts]
        order, st = rank_group(ts, res, rng)
        W[g], R[g] = order[0], order[1]
        thirds.append((g, st["pts"][order[2]], st["gd"][order[2]], st["gf"][order[2]], order[2]))
    ranked = rank_thirds([(g, p, d, f, t) for g, p, d, f, t in thirds], rng)[:4]
    T = {e[0]: e[4] for e in ranked}
    pred = {frozenset(p) for p in build_r16(W, R, T).values()}
    actual = {frozenset((r.home_team, r.away_team)) for r in ko.iloc[:8].itertuples()}
    return dict(winners=W, runners_up=R, best_thirds=T, matched_r16_ties=len(pred & actual), of=8)


# ------------------------------------------------------------------ (2) pre-tournament back-test
def actual_stage(grp, ko, sho):
    teams = set(grp.home_team) | set(grp.away_team)
    stage = {t: 0 for t in teams}                           # 0 group exit .. 5 champion (rounds of the knockout reached)
    rounds = [(0, 8, 1), (8, 12, 2), (12, 14, 3), (14, 15, 4)]
    for a, b, k in rounds:
        for m in ko.iloc[a:b].itertuples():
            for t in (m.home_team, m.away_team):
                stage[t] = max(stage[t], k)
    stage[winner_of(ko.iloc[-1], sho)] = 5
    return stage


def backtest(res, sho, grp, ko, n=10_000, seed=2023):
    past = res[res.date < KICKOFF]
    sho_p = sho[sho.date < KICKOFF]
    df, state = build_features(past)
    teams = [t for ts in GROUPS_2023.values() for t in ts]
    model = train_xgb(df, 2023)
    engine = MatchEngine(model, state, teams=teams, host=HOST_2023)
    shoot = ShootoutModel.fit(df, sho_p)
    out = Simulator(engine, shoot, groups=GROUPS_2023).run(n, seed)
    tab = out.table
    act = actual_stage(grp, ko, sho)
    tab["Actual stage"] = [act[t] for t in tab.index]
    cols = ["Advance", "QF", "SF", "Final", "Champion"]
    tab["Expected stage"] = tab[cols].sum(axis=1)
    base = np.array([16, 8, 4, 2, 1]) / 24
    y = np.array([[act[t] >= k for k in range(1, 6)] for t in tab.index], float)
    p = tab[cols].values
    brier, brier0 = ((p - y) ** 2).sum(axis=1).mean(), ((base - y) ** 2).sum(axis=1).mean()
    champ = ko.iloc[-1]; champ_name = winner_of(champ, sho)
    finalists = {champ.home_team, champ.away_team}
    top16 = set(tab.sort_values("Advance", ascending=False).index[:16])
    actual16 = {t for t, s in act.items() if s >= 1}
    # group-stage match probabilities (clean 90-minute results) from the pre-tournament engine
    ll, ll0, hit = [], [], []
    prior = np.bincount(df[df.date.dt.year <= 2023].y, minlength=3) / len(df[df.date.dt.year <= 2023])
    for m in grp.itertuples():
        i, j = engine.idx[m.home_team], engine.idx[m.away_team]
        pw = engine.pair_probs[i, j]
        o = 0 if m.home_score > m.away_score else 1 if m.home_score == m.away_score else 2
        ll.append(-np.log(pw[o])); ll0.append(-np.log(prior[o])); hit.append(pw.argmax() == o)
    metrics = dict(
        champion=champ_name, champion_title_prob=float(tab.Champion[champ_name]),
        champion_rank=int(tab.Champion.rank(ascending=False)[champ_name]),
        uniform_title_prob=1 / 24, finalists=sorted(finalists),
        r16_teams_in_model_top16=len(top16 & actual16),
        expected_r16_hits_under_model=float(tab.Advance[list(actual16)].sum()),
        brier_stage=float(brier), brier_stage_baseline=float(brier0), brier_skill=float(1 - brier / brier0),
        spearman_expected_vs_actual=float(tab["Expected stage"].corr(tab["Actual stage"], method="spearman")),
        group_match_logloss=float(np.mean(ll)), group_match_logloss_prior=float(np.mean(ll0)),
        group_match_accuracy=float(np.mean(hit)), n_group_matches=len(hit), n_sims=n)
    return tab, metrics


# ------------------------------------------------------------------ (3) defending-champion check
CONT = ["AFC Asian Cup", "African Cup of Nations", "UEFA Euro", "Copa América", "Gold Cup", "FIFA World Cup"]


def defending_champions(res, sho, since=1996, seed=0):
    df, _ = build_features(res)
    df = df.sort_values("date", kind="stable")
    rows = []
    for tour in CONT:
        t = df[df.tournament == tour]
        ed = (t.date.diff().dt.days.fillna(0) > 120).cumsum()
        prev_champ = None
        for e, g in t.groupby(ed):
            g = g.sort_values("date", kind="stable")
            if g.date.dt.year.min() >= since and prev_champ is not None:
                m = g[(g.home_team == prev_champ) | (g.away_team == prev_champ)]
                for r in m.itertuples():
                    home = r.home_team == prev_champ
                    adv = 0 if r.neutral else (100 if home else -100)
                    d = (r.elo_home - r.elo_away) if home else (r.elo_away - r.elo_home)
                    exp = 1 / (1 + 10 ** (-(d + adv) / 400))
                    gf, ga = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
                    S = 1.0 if gf > ga else 0.5 if gf == ga else 0.0
                    rows.append((tour, g.date.min().year, prev_champ, S - exp))
            f = g.iloc[-1]
            try:                                            # the final is the last match of an edition
                prev_champ = winner_of(f, sho)
            except IndexError:                              # level and no shoot-out record: skip next edition
                prev_champ = None
    r = pd.DataFrame(rows, columns=["tournament", "year", "team", "resid"])
    ed_means = r.groupby(["tournament", "year"]).resid.agg(["mean", "size"])
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ed_means))
    boots = []
    for _ in range(5000):                                   # cluster bootstrap over tournament editions
        b = rng.choice(idx, len(idx)); w = ed_means["size"].values[b]
        boots.append((ed_means["mean"].values[b] * w).sum() / w.sum())
    return r, dict(n_matches=len(r), n_editions=len(ed_means), mean_residual=float(r.resid.mean()),
                   ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                   pts_per_game_gain=float(r.resid.mean()))


# ------------------------------------------------------------------ figure
def figure(tab, metrics):
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlelocation": "left", "axes.titleweight": "bold",
                         "font.size": 10})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[1, 1.1]))
    d = tab.sort_values("Champion", ascending=False).head(10).iloc[::-1]
    cols = ["#e87b34" if t == metrics["champion"] else "#2a78d6" for t in d.index]
    a.barh(d.index, d.Champion * 100, color=cols, height=0.62)
    for y, v in enumerate(d.Champion * 100):
        a.text(v + 0.4, y, f"{v:.1f}%", va="center", fontsize=9)
    a.set_xlabel("Pre-tournament title probability (%)")
    a.set_title("Back-test: 2023 Asian Cup title odds", pad=12)
    a.text(0.97, 0.05, f"Orange: actual champion\n(rank {metrics['champion_rank']} of 24)", transform=a.transAxes,
           ha="right", fontsize=9, color="#52514e")
    jit = np.random.default_rng(1).uniform(-0.12, 0.12, len(tab))
    b.plot([0, 5], [0, 5], color="#c9c8c3", lw=1)
    b.scatter(tab["Expected stage"], tab["Actual stage"] + jit, s=42, color="#2a78d6", edgecolor="#fcfcfb", linewidth=1)
    off = {"Iran": (6, 6), "South Korea": (-62, -14), "Australia": (-58, 2), "Japan": (6, 4)}
    for t in ["Qatar", "Jordan", "Iran", "Japan", "Australia", "South Korea", "Saudi Arabia"]:
        b.annotate(t, (tab["Expected stage"][t], tab["Actual stage"][t]), textcoords="offset points",
                   xytext=off.get(t, (5, 4)), fontsize=8.5)
    b.set_xlabel("Model's expected knockout rounds reached")
    b.set_ylabel("Actual rounds reached\n(0 group exit, 1 R16, ... 5 champion)")
    b.set_title("Expected vs actual progress, 24 teams", pad=12)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "10_backtest_2023.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    res, sho = load_raw(DATA)
    grp, ko = load_2023(res, sho)
    rp = replay(grp, ko, sho)
    print("replay:", rp)
    tab, bt = backtest(res, sho, grp, ko)
    print(json.dumps(bt, indent=1))
    r, dc = defending_champions(res, sho)
    print("defending champions:", dc)
    OUT.mkdir(exist_ok=True)
    tab.round(5).to_csv(OUT / "backtest_2023.csv")
    (OUT / "validation.json").write_text(json.dumps(dict(replay_2023=rp, backtest_2023=bt, defending_champions=dc),
                                                    indent=1, default=str))
    figure(tab, bt)
