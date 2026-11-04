"""Small full-tournament simulation with a synthetic Elo engine (no data files needed)."""
import numpy as np, pytest
from types import SimpleNamespace
from afc_sim.config import TEAMS, GROUPS
from afc_sim.scores import ScoreModel
from afc_sim.shootouts import ShootoutModel
from afc_sim.simulate import Simulator

def synthetic_engine(seed=3):
    rng = np.random.default_rng(seed)
    elo = {t: 1500 + rng.normal(0, 150) for t in TEAMS}
    n = len(TEAMS); pp = np.full((n, n, 3), np.nan)
    for i, a in enumerate(TEAMS):
        for j, b in enumerate(TEAMS):
            if i != j:
                e = 1 / (1 + 10 ** (-(elo[a] - elo[b]) / 400)); d = 0.26 - 0.1 * abs(e - .5)
                pp[i, j] = [e * (1 - d) , d, (1 - e) * (1 - d)]
    pp /= pp.sum(-1, keepdims=True)
    return SimpleNamespace(teams=list(TEAMS), idx={t: i for i, t in enumerate(TEAMS)},
                           state=SimpleNamespace(elo=elo), pair_probs=pp), elo

@pytest.fixture(scope="module")
def result():
    eng, _ = synthetic_engine()
    sh = ShootoutModel(0.1, 0.1, {}, 8, {})
    return Simulator(eng, sh).run(1500, seed=11)

def test_stage_probabilities_conserve_mass(result):
    t = result.table
    assert t["Win group"].sum() == pytest.approx(6)
    assert t["Advance"].sum() == pytest.approx(16)
    assert t["QF"].sum() == pytest.approx(8)
    assert t["SF"].sum() == pytest.approx(4)
    assert t["Final"].sum() == pytest.approx(2)
    assert t["Champion"].sum() == pytest.approx(1)
    assert t["Adv as 3rd"].sum() == pytest.approx(4)
    for g, ts in GROUPS.items():
        assert t.loc[ts, "Win group"].sum() == pytest.approx(1)
        assert t.loc[ts, [f"Pos{p}" for p in (1, 2, 3, 4)]].sum().sum() == pytest.approx(4)

def test_monotone_progression_and_elo_ordering(result):
    t = result.table
    cols = ["Advance", "QF", "SF", "Final", "Champion"]
    assert (t[cols].diff(axis=1).iloc[:, 1:] <= 1e-12).all().all()
    assert t["Champion"].corr(t["Elo"], method="spearman") > 0.7

def test_reproducible_given_seed():
    eng, _ = synthetic_engine(); sh = ShootoutModel(0.1, 0.1, {}, 8, {})
    a = Simulator(eng, sh).run(200, seed=5).table; b = Simulator(eng, sh).run(200, seed=5).table
    assert a.equals(b)
