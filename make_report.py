"""Run the final 10,000-run Monte Carlo, robustness checks, and write results/, figures/ and REPORT.md.

    python make_report.py                # full run (about 2-3 minutes)
    python make_report.py --skip-sim     # re-draw figures/report from results/cache.pkl
"""
import argparse, json, pickle, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from afc_sim import MatchEngine, ShootoutModel, Simulator, build_features, load_raw, train_xgb
from afc_sim.config import GROUPS, HOST
from afc_sim.scores import ScoreModel

N, SEED = 10_000, 2027
OUT, FIG = Path("results"), Path("figures")
CACHE = OUT / "cache.pkl"

# ---- palette (reference categorical slots 1-5; single-hue blue ramp for magnitude) -------------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE = LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#9cc0ee", "#2a78d6", "#123f78"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.grid": False, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 130})


def ci95(p, n=N):
    return 1.96 * np.sqrt(p * (1 - p) / n)


# =============================================================================== simulation
def simulate():
    t0 = time.time()
    res, sho = load_raw("data")
    df, state = build_features(res)
    model = train_xgb(df, 2026)
    engine = MatchEngine(model, state)
    shoot = ShootoutModel.fit(df, sho)
    scores = ScoreModel(engine.pair_probs)
    main = Simulator(engine, shoot, scores=scores).run(N, SEED)
    cache = dict(main=main, shoot_diag=shoot.diagnostics, elo=state.elo, seeds={}, et={}, host=None)
    for s in (1, 2, 3, 4):                                           # seed stability
        cache["seeds"][s] = Simulator(engine, shoot, scores=scores).run(N, s).table["Champion"]
    for et in (0.4, 1.2):                                            # extra-time intensity
        cache["et"][et] = Simulator(engine, shoot, et_factor=et, scores=scores).run(N, SEED)
    eng2 = MatchEngine(model, state, host_advantage=False)           # no host advantage
    cache["host"] = Simulator(eng2, shoot).run(N, SEED)
    cache["pair_probs"], cache["teams"] = engine.pair_probs, engine.teams
    cache["runtime_s"] = time.time() - t0
    OUT.mkdir(exist_ok=True)
    pickle.dump(cache, open(CACHE, "wb"))
    return cache


# =============================================================================== figures
def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def fig_title_odds(t):
    d = t.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(d.index, d.Champion * 100, xerr=ci95(d.Champion) * 100, color=C[0], height=0.62,
            error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    for y, v in enumerate(d.Champion * 100):
        ax.text(v + ci95(d.Champion.iloc[y]) * 100 + 0.5, y, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Probability of winning the 2027 Asian Cup (%)  -  bars: 95% Monte Carlo interval")
    ax.set_title("Title odds: Japan is the clear favourite", pad=12)
    ax.set_xlim(0, d.Champion.max() * 100 * 1.2)
    save(fig, "01_title_odds.png")


def fig_heatmap(t):
    cols = ["Advance", "QF", "SF", "Final", "Champion"]
    d = t[cols] * 100
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(d.values, cmap=BLUE, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(cols)), ["Round of 16", "Quarter-final", "Semi-final", "Final", "Champion"], rotation=25, ha="right")
    ax.set_yticks(range(len(d)), [f"{n}  ({t.Group[n]})" for n in d.index])
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.values[i, j]
            ax.text(j, i, f"{v:.0f}" if v >= 1 else ("<1" if v > 0 else "0"), ha="center", va="center",
                    fontsize=8, color="white" if v > 55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Probability (%) of reaching each stage", pad=12)
    save(fig, "02_stage_heatmap.png")


def fig_groups(t):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    shades = [BLUE(0.95), BLUE(0.62), BLUE(0.32), BLUE(0.1)]
    for ax, (g, ts) in zip(axes.ravel(), GROUPS.items()):
        d = t.loc[ts].sort_values("Pos1", ascending=False).iloc[::-1]
        left = np.zeros(len(d))
        for k in range(4):
            v = d[f"Pos{k + 1}"].values * 100
            ax.barh(d.index, v, left=left, color=shades[k], edgecolor=SURF, linewidth=1.5, height=0.7,
                    label=["1st", "2nd", "3rd", "4th"][k])
            for y, (l, w) in enumerate(zip(left, v)):
                if w >= 9:
                    ax.text(l + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if k < 2 else INK)
            left += v
        ax.set_title(f"Group {g}", fontsize=11)
        ax.set_xlim(0, 100)
    axes[1, 1].set_xlabel("Probability of finishing in each group position (%)")
    axes[0, 0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.12), frameon=False, fontsize=8)
    fig.suptitle("Group-stage finishing positions", x=0.01, ha="left", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "03_group_positions.png")


def fig_thirds(ex):
    rows = pd.DataFrame([(p, gd, q, n) for (p, gd, q), n in ex["third_pts"].items()], columns=["pts", "gd", "q", "n"])
    byp = rows.groupby("pts").apply(lambda r: pd.Series(dict(n=r.n.sum(), pq=r[r.q].n.sum() / r.n.sum())), include_groups=False)
    byp = byp[byp.n > 50]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(range(len(byp)), byp.pq * 100, color=C[0], width=0.6)
    for x, pq in enumerate(byp.pq):
        ax.text(x, pq * 100 + 2, f"{pq * 100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(byp)), [f"{p}\n(n={int(n):,})" for p, n in zip(byp.index, byp.n)])
    ax.set_ylim(0, 112)
    ax.set_xlabel("Points of a third-placed team after three games (n = third-placed teams observed)")
    ax.set_ylabel("Chance it is one of the best four thirds (%)")
    ax.set_title("How many points does a third place need?", pad=12)
    save(fig, "04_third_place.png")
    return byp


def fig_knockout(ex):
    rounds = ["R16", "QF", "SF", "Final"]
    how = ["90", "ET", "PEN"]
    lab = {"90": "Decided in 90 min", "ET": "Extra time", "PEN": "Penalties"}
    d = pd.DataFrame({h: [ex["ko_round_how"][(r, h)] for r in rounds] for h in how}, index=rounds)
    d = d.div(d.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    left = np.zeros(len(d))
    for k, h in enumerate(how):
        ax.barh(d.index[::-1], d[h].values[::-1], left=left[::-1], color=C[k], edgecolor=SURF, linewidth=2,
                height=0.62, label=lab[h])
        for y, (l, w) in enumerate(zip(left[::-1], d[h].values[::-1])):
            if w > 5:
                ax.text(l + w / 2, y, f"{w:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left += d[h].values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of knockout ties (%)")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    ax.set_title("How knockout ties are settled", pad=30)
    save(fig, "05_knockout_resolution.png")
    return d


def fig_convergence(c):
    ex, t = c["main"].extras, c["main"].table
    top = list(t.index[:5])
    champs = ex["champions"]
    names = ex["names"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    n = np.arange(1, len(champs) + 1)
    for k, tm in enumerate(top):
        run = np.cumsum(champs == names.index(tm)) / n
        a.plot(n, run * 100, color=C[k], lw=1.6, label=tm)
    a.set_xscale("log")
    a.set_xlim(100, len(champs))
    a.set_ylim(0, 45)
    a.set_xlabel("Number of simulated tournaments (log scale)")
    a.set_ylabel("Running title probability (%)")
    a.set_title("Title estimates settle by about 3,000 runs", pad=12)
    a.legend(frameon=False, ncol=3, loc="center", bbox_to_anchor=(0.5, 0.54))
    # seed stability: dot plot of the five independent 10,000-run estimates for the top 8
    top8 = list(t.index[:8])[::-1]
    allseeds = pd.concat([t["Champion"].rename("2027")] + [s.rename(str(k)) for k, s in c["seeds"].items()], axis=1)
    for y, tm in enumerate(top8):
        v = allseeds.loc[tm].values * 100
        b.plot([v.min(), v.max()], [y, y], color=INK2, lw=1.5, zorder=1)
        b.scatter(v, [y] * len(v), s=70, color=C[0], alpha=0.55, edgecolor=SURF, linewidth=1.0, zorder=2)
    b.set_yticks(range(len(top8)), top8)
    b.set_xlabel("Title probability (%), five independent seeds")
    b.set_title("Five seeds agree to about 1 point or less", pad=12)
    fig.tight_layout()
    save(fig, "06_convergence_and_seeds.png")
    return allseeds


def fig_elo(t):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    d = t[t.Champion > 0]
    ax.scatter(t.Elo, (t.Champion.clip(lower=0.0003)) * 100, s=42, color=C[0], edgecolor=SURF, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 60)
    ax.set_yticks([0.1, 1, 10, 50], ["0.1", "1", "10", "50"])
    offs = {"Japan": (6, -2), "Australia": (6, -4), "Iran": (-8, 8), "Saudi Arabia": (-70, -2),
            "South Korea": (-78, -3), "Uzbekistan": (6, 4), "Jordan": (6, -9)}
    for tm, (dx, dy) in offs.items():
        ax.annotate(tm, (t.Elo[tm], t.Champion[tm] * 100), textcoords="offset points", xytext=(dx, dy), fontsize=8.5, color=INK)
    ax.set_xlabel("Elo rating entering the tournament")
    ax.set_ylabel("Title probability (%, log scale)")
    ax.set_title("Elo drives title odds; Saudi Arabia (host) is the outlier", pad=12)
    ax.text(0.02, 0.95, "Teams with zero titles in 10,000 runs are drawn at the axis floor", transform=ax.transAxes,
            fontsize=8, color=INK2)
    save(fig, "07_elo_vs_title.png")


def fig_finals(ex):
    top = ex["final_pairs"].most_common(10)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh([" v ".join(sorted(p)) for p, _ in top], [n / N * 100 for _, n in top], color=C[0], height=0.62)
    for y, (_, n) in enumerate(top):
        ax.text(n / N * 100 + 0.1, y, f"{n / N * 100:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Share of simulated tournaments (%)")
    ax.set_title("Most likely finals", pad=12)
    ax.set_xlim(0, top[-1][1] / N * 100 * 1.15)
    save(fig, "08_likely_finals.png")


def fig_sensitivity(c):
    t = c["main"].table
    top = list(t.index[:8])
    rows = {"Baseline (ET factor 0.8)": t["Champion"],
            "ET factor 0.4": c["et"][0.4].table["Champion"], "ET factor 1.2": c["et"][1.2].table["Champion"],
            "No host advantage": c["host"].table["Champion"]}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ys = np.arange(len(top))[::-1]
    for k, (name, s) in enumerate(rows.items()):
        ax.scatter(s[top].values * 100, ys + (k - 1.5) * 0.14, s=34, color=C[k], edgecolor=SURF, linewidth=1, label=name)
    ax.set_yticks(ys, top)
    ax.set_xlabel("Title probability (%)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Sensitivity of title odds to modelling choices", pad=12)
    save(fig, "09_sensitivity.png")
    return rows


# =============================================================================== report
def md_table(df, fmt=None):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    for idx, r in df.iterrows():
        out.append("| " + " | ".join([str(idx)] + [(fmt or {}).get(c, "{}").format(r[c]) for c in cols]) + " |")
    return "\n".join(out)


def build(c):
    main = c["main"]; t, ex, diag = main.table, main.extras, main.diagnostics
    fig_title_odds(t); fig_heatmap(t); fig_groups(t)
    byp = fig_thirds(ex); kod = fig_knockout(ex); seeds = fig_convergence(c)
    fig_elo(t); fig_finals(ex); sens = fig_sensitivity(c)

    # ---- csv outputs
    t.round(5).to_csv(OUT / "team_probabilities.csv")
    pd.DataFrame([(" v ".join(sorted(p)), n / N) for p, n in ex["final_pairs"].most_common(25)],
                 columns=["final", "probability"]).to_csv(OUT / "likely_finals.csv", index=False)
    pd.DataFrame([(" v ".join(sorted(p)), n / N) for p, n in ex["r16_pairs"].most_common(40)],
                 columns=["round_of_16_tie", "probability"]).to_csv(OUT / "likely_r16_ties.csv", index=False)
    sens_df = pd.DataFrame(sens).mul(100).round(2)
    sens_df.to_csv(OUT / "sensitivity_title_pct.csv")
    seeds.mul(100).round(2).to_csv(OUT / "seed_stability_title_pct.csv")
    (OUT / "diagnostics.json").write_text(json.dumps(dict(
        n_sims=N, seed=SEED, runtime_s=round(c["runtime_s"], 1), diagnostics=diag,
        goals_per_group_game=ex["goals_per_group_game"], group_draw_rate=ex["group_draw_rate"],
        mean_points_by_position=list(ex["pts_by_pos"]), shootout_model=c["shoot_diag"],
        champion_decided_in_final=dict(ex["champ_how"])), indent=1, default=float))

    # ---- numbers for the narrative
    top = t.head(10)
    hi = lambda tm: f"{t.Champion[tm] * 100:.1f}% (±{ci95(t.Champion[tm]) * 100:.1f})"
    host_t = c["host"].table
    grp_elo = {g: np.mean([c["elo"][x] for x in ts]) for g, ts in GROUPS.items()}
    gstrong = max(grp_elo, key=grp_elo.get); gweak = min(grp_elo, key=grp_elo.get)
    # group where the best team is least dominant
    lead = {g: t.loc[ts, "Win group"].max() for g, ts in GROUPS.items()}
    opengrp = min(lead, key=lead.get)
    pen_final = ex["champ_how"]["PEN"] / N
    ko_pen = diag["knockout_decided_in"]["PEN"]; ko_et = diag["knockout_decided_in"]["ET"]
    spread = (seeds.max(axis=1) - seeds.min(axis=1)).max() * 100
    third_cut = byp.pq
    three = lambda p: f"{byp.pq.get(p, float('nan')) * 100:.0f}%"
    rtab = t[["Group", "Elo", "Advance", "QF", "SF", "Final", "Champion"]].head(24).copy()
    for k in ["Advance", "QF", "SF", "Final", "Champion"]:
        rtab[k] = (rtab[k] * 100).round(1)
    rtab_md = md_table(rtab, {"Advance": "{:.1f}", "QF": "{:.1f}", "SF": "{:.1f}", "Final": "{:.1f}", "Champion": "{:.1f}"})
    gmd = []
    for g, ts in GROUPS.items():
        d = t.loc[ts].sort_values("Win group", ascending=False)[["Elo", "Win group", "Advance", "Adv as 3rd"]]
        d = (d.assign(**{k: (d[k] * 100).round(1) for k in ["Win group", "Advance", "Adv as 3rd"]}))
        d = d.rename(columns={"Win group": "Win group %", "Advance": "Advance %", "Adv as 3rd": "Advance as 3rd %"})
        gmd.append(f"**Group {g}** (mean Elo {grp_elo[g]:.0f})\n\n" + md_table(d))
    fin = "\n".join(f"- {' v '.join(sorted(p))}: {n / N * 100:.1f}%" for p, n in ex["final_pairs"].most_common(8))
    r16 = "\n".join(f"- {' v '.join(sorted(p))}: {n / N * 100:.1f}%" for p, n in ex["r16_pairs"].most_common(8))
    sens_md = md_table(sens_df.loc[list(t.index[:8])], {k: "{:.1f}" for k in sens_df.columns})
    tb = ex["tiebreak_resolution"]
    saudi_adv = t.Champion[HOST] * 100; saudi_nohost = host_t.Champion[HOST] * 100
    saudi_adv_g = t.Advance[HOST] * 100; saudi_adv_g_n = host_t.Advance[HOST] * 100

    vp = OUT / "validation.json"
    tab_bt_japan = pd.read_csv(OUT / "backtest_2023.csv", index_col=0).Champion["Japan"] * 100 if vp.exists() else 0
    val_md = ""
    if vp.exists():
        v = json.loads(vp.read_text()); b, d, rp = v["backtest_2023"], v["defending_champions"], v["replay_2023"]
        val_md = f"""## 7. Validation

![Back-test](figures/10_backtest_2023.png)

- **Replay of the real 2023 group stage.** Feeding the actual 2023 results through the format code gave the correct winner and runner-up of all six groups, the correct four best third-placed teams (Jordan, Palestine, Syria, Indonesia) and **{rp['matched_r16_ties']} of {rp['of']} real Round-of-16 ties**. This checks the tiebreakers, third-place ranking and bracket table against a real tournament (2023 used the same bracket as 2027).
- **Pre-tournament back-test of the 2023 Asian Cup** (data before kick-off only, {b['n_sims']:,} simulations). Qatar, the champion, was given {b['champion_title_prob'] * 100:.1f}% (rank {b['champion_rank']} of 24; a uniform guess is 4.2%). Japan was the favourite at {tab_bt_japan:.1f}% and lost in the quarter-finals; runner-up Jordan had 0.5%. Over all stages the Brier score was {b['brier_stage']:.3f} against {b['brier_stage_baseline']:.3f} for a base-rate guess (a {b['brier_skill'] * 100:.0f}% improvement), {b['r16_teams_in_model_top16']} of the 16 real Round-of-16 teams were in the model's top 16, and group-match log loss was {b['group_match_logloss']:.2f} against {b['group_match_logloss_prior']:.2f} for class frequencies ({b['n_group_matches']} matches). One tournament is a small sample: it shows the forecast carries real information, not that the title odds are perfectly calibrated.
- **Defending champions.** Across {d['n_matches']} matches by defending champions in {d['n_editions']} continental tournaments and World Cups since 1996, they scored {d['mean_residual']:+.3f} points per game relative to their Elo expectation (95% interval {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}). There is no evidence of a champion effect beyond rating, so Qatar's low odds (Elo {c['elo']['Qatar']:.0f}, in the strongest group) reflect its current strength rather than a missing feature.

"""
    rep = f"""# Forecasting the 2027 AFC Asian Cup: Monte Carlo report

*Omar Elouny · CSCI-630.01 · {N:,} simulated tournaments (seed {SEED}) · generated by `make_report.py`*

## 1. Summary

I simulated the 24-team 2027 AFC Asian Cup {N:,} times. Each match is drawn from an XGBoost win/draw/loss model (trained on every international result through 6 October 2026), turned into scorelines by a Poisson layer, and played through the official group tiebreakers, the best-four-thirds rule, the published bracket, extra time and penalties.

- **Japan is the favourite at {hi('Japan')}**, ahead of Australia ({hi('Australia')}), Saudi Arabia ({hi('Saudi Arabia')}), Iran ({hi('Iran')}) and South Korea ({hi('South Korea')}). Those five hold {t.head(5).Champion.sum() * 100:.0f}% of the title probability.
- **The field is top-heavy.** Teams outside the top ten win the tournament in {(1 - top.Champion.sum()) * 100:.1f}% of runs.
- **Saudi Arabia's host advantage is worth about {saudi_adv - saudi_nohost:+.1f} points of title probability** ({saudi_adv:.1f}% with it, {saudi_nohost:.1f}% if it is treated like a neutral side), mostly because it lifts them out of Group A more reliably ({saudi_adv_g:.0f}% vs {saudi_adv_g_n:.0f}% to reach the Round of 16).
- **About {(ko_pen + ko_et) * 100:.0f}% of knockout ties go beyond 90 minutes**, and {ko_pen * 100:.0f}% reach penalties, so a coin-flip-like shoot-out decides roughly one knockout game in {round(1 / ko_pen)}.
- **Results are numerically stable**: five independent 10,000-run seeds differ by at most {spread:.1f} percentage points for any team's title odds.

![Title odds](figures/01_title_odds.png)

## 2. Method in one page

| Stage | What it does |
|---|---|
| Ratings and features | Elo from 49,945 matches (1872-2026) with tournament weights, goal-margin multiplier and host bonus; rolling form, experience, rest, neutral/host flags. Frozen at the last result in the data. |
| Match model | XGBoost (depth 2, 487 trees, tuned on 2020-22 validation; best-calibrated model in the model-testing notebook) refit on all matches through 2026. Neutral-venue probabilities are symmetrised; Saudi Arabia is the true home side in all its games. |
| Score layer | For each pairing, independent-Poisson rates are fitted so the implied W/D/L matches XGBoost (max error < 1e-6), then the score distribution is reweighted so the W/D/L marginals are exactly XGBoost's. Scores feed goal-difference tiebreakers. |
| Groups | Six groups of four; points, then head-to-head points / goal difference / goals, re-applied to subsets, then overall goal difference and goals scored, then a penalty shoot-out (if two teams met in the last round) or lots. |
| Best thirds | Four best of six on points, goal difference, goals scored, then lots; placed in the bracket with the official 15-row table. |
| Knockouts | R16 matches 37-44, QF 45-48, SF 49-50, final 51 (no third-place game). A tie after 90 minutes gets 30 minutes of extra time (Poisson, intensity 0.8 of the regulation rate), then a symmetric shoot-out model using Elo gap and each team's shrunk shoot-out history. |

Full details and assumptions are in the README.

## 3. Who wins the tournament

{md_table(rtab, {"Advance": "{:.1f}", "QF": "{:.1f}", "SF": "{:.1f}", "Final": "{:.1f}", "Champion": "{:.1f}"})}

*Percentages. Monte Carlo 95% interval is at most ±{ci95(0.5) * 100:.1f} points for any entry (±{ci95(t.Champion.iloc[0]) * 100:.1f} for the leader's title odds).*

![Stage heatmap](figures/02_stage_heatmap.png)

**Reading the table.** Japan's edge comes from rating: Elo {c['elo']['Japan']:.0f}, the highest in the field by {c['elo']['Japan'] - c['elo']['Australia']:.0f} points. Saudi Arabia (Elo {c['elo'][HOST]:.0f}) reaches the quarter-finals about as often as Iran (Elo {c['elo']['Iran']:.0f}) and Australia ({t.QF[HOST] * 100:.0f}% vs {t.QF['Iran'] * 100:.0f}% and {t.QF['Australia'] * 100:.0f}%) despite a much lower rating. That is the host effect: with home advantage switched off its quarter-final chance falls to {host_t.QF[HOST] * 100:.0f}% (section 6). The size of that effect comes from the model's host features, which are learned from matches with a home side, so it is the least certain number in this report.

## 4. Group stage

{chr(10).join(gmd)}

![Group positions](figures/03_group_positions.png)

- **Strongest group by mean Elo:** Group {gstrong} ({grp_elo[gstrong]:.0f}). **Weakest:** Group {gweak} ({grp_elo[gweak]:.0f}).
- **Most open group:** Group {opengrp}, where the favourite wins the group only {lead[opengrp] * 100:.0f}% of the time.
- Average group-stage match: {ex['goals_per_group_game']:.2f} goals, {ex['group_draw_rate'] * 100:.0f}% draws (Asian Cups since 2004: 2.68 goals, 21% draws).
- Mean points: winners {ex['pts_by_pos'][0]:.1f}, runners-up {ex['pts_by_pos'][1]:.1f}, third {ex['pts_by_pos'][2]:.1f}, fourth {ex['pts_by_pos'][3]:.1f}.

### Best third-placed teams

Four of six third-placed teams advance, so even a modest haul often survives. A third-placed team with 4 points advances {three(4)} of the time, with 3 points {three(3)}, with 2 points {three(2)} and with 6 points {three(6)}.

![Third place](figures/04_third_place.png)

### Tiebreakers

Tiebreakers matter constantly in a four-team group: across {N:,} tournaments the cascade was needed {sum(tb.values()) / N:.1f} times per tournament. Resolution counts: head-to-head {tb.get('head_to_head', 0):,}, overall goal difference / goals scored {tb.get('overall_gd_gf', 0):,}, last-round shoot-out {tb.get('shootout', 0):,}, drawing of lots {tb.get('lots', 0):,}. Lots includes disciplinary points, which cannot be simulated, so treat it as an upper bound on how often cards would decide a placing.

## 5. Knockout rounds

![Knockout resolution](figures/05_knockout_resolution.png)

Across all knockout games, {diag['knockout_decided_in']['90'] * 100:.1f}% are settled in 90 minutes, {ko_et * 100:.1f}% in extra time and {ko_pen * 100:.1f}% on penalties. The champion needed a shoot-out to win the final in {pen_final * 100:.1f}% of tournaments.

The shoot-out model (fitted on {c['shoot_diag']['n_shootouts']} historical shoot-outs) gives a stronger Elo side only a small edge: on a held-out 2015+ sample its log loss is {c['shoot_diag']['holdout_logloss']:.3f} versus {c['shoot_diag']['coin_logloss']:.3f} for a fair coin. The first shooter won {c['shoot_diag']['first_shooter_win_rate'] * 100:.1f}% of {c['shoot_diag']['n_first_shooter']} recorded shoot-outs, not a significant edge, so it is left out.

**Most likely finals**

{fin}

**Most likely Round-of-16 ties**

{r16}

![Likely finals](figures/08_likely_finals.png)

## 6. Robustness

{md_table(sens_df.loc[list(t.index[:8])], {k: "{:.1f}" for k in sens_df.columns})}

*Title probability (%). Each column is a separate {N:,}-run simulation with the same seed.*

![Sensitivity](figures/09_sensitivity.png)

- **Extra-time intensity** (0.4 to 1.2 of the regulation rate) moves no team's title odds by more than {max((sens_df['ET factor 1.2'] - sens_df['ET factor 0.4']).abs()):.1f} points.
- **Host advantage** matters mainly for Saudi Arabia, and slightly redistributes title odds to the other favourites without it.
- **Seed stability**: five independent seeds agree within {spread:.1f} points.

![Convergence and seeds](figures/06_convergence_and_seeds.png)

![Elo vs title](figures/07_elo_vs_title.png)

{val_md}## 8. Limitations

1. **Ratings are frozen** at 6 October 2026. Form, injuries, squad changes and the three months before kick-off are not captured. Ratings are not updated during the simulated tournament.
2. **Tiebreakers follow the 2023 regulations** because the 2027 wiki page did not list them; the group matchday order is assumed (it only affects the rare last-round shoot-out rule).
3. **Disciplinary points are not modelled**; ties that would reach that rule are decided by lots.
4. **Knockout draws are probably under-modelled.** The model sends {ko_pen * 100:.0f}% of knockout ties to penalties; 16 of about 62 Asian Cup knockout games since 2004 (26%) did. The sample is small, so this is a hint rather than a finding, but knockouts may be cagier than the average match the model was trained on.
5. **Model uncertainty is not propagated.** The Monte Carlo intervals reflect simulation noise only, not uncertainty in the XGBoost probabilities or Elo ratings, so true uncertainty about each team's odds is wider than the intervals shown.
6. **Probabilities are not betting advice**; they are outputs of a simple, transparent model.

## 9. Reproduce

```bash
pip install -r requirements.txt            # put the Kaggle CSVs in data/
python validation.py                       # 2023 replay and back-test, defending-champion check
python make_report.py                      # about {c['runtime_s'] / 60:.0f} min: results/, figures/, REPORT.md
python -m pytest tests -q
```
"""
    Path("REPORT.md").write_text(rep)
    print(rep[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sim", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cache = pickle.load(open(CACHE, "rb")) if a.skip_sim else simulate()
    build(cache)
