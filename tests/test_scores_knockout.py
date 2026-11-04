import numpy as np, pytest
from afc_sim.scores import MAX_GOALS, ScoreModel, fit_rates, poisson_wdl, score_pmf, _HW, _DR, _AW
from afc_sim.knockout import play_knockout
from afc_sim.shootouts import ShootoutModel

P = [np.array(p) for p in ([.62, .22, .16], [.34, .30, .36], [.08, .17, .75], [.90, .07, .03], [.45, .27, .28])]

@pytest.mark.parametrize("p", P)
def test_poisson_rates_reproduce_engine_probabilities(p):
    lh, la, err = fit_rates(p)
    assert err < 1e-5 and lh > 0 and la > 0
    assert np.allclose(poisson_wdl(lh, la), p, atol=1e-5)

@pytest.mark.parametrize("p", P)
def test_score_pmf_has_exact_wdl_marginals(p):
    lh, la, _ = fit_rates(p)
    M = score_pmf(p, lh, la)
    assert M.sum() == pytest.approx(1)
    assert [M[_HW].sum(), M[_DR].sum(), M[_AW].sum()] == pytest.approx(list(p / p.sum()))

def _model():
    pp = np.full((3, 3, 3), np.nan)
    for i in range(3):
        for j in range(3):
            if i != j:
                pp[i, j] = P[(i + j) % len(P)]
    return pp, ScoreModel(pp)

def test_sampled_scores_match_pmf_and_wdl():
    pp, sm = _model()
    u = np.random.default_rng(0).random(200_000)
    gi, gj = sm.sample(0, 1, u)
    wdl = [(gi > gj).mean(), (gi == gj).mean(), (gi < gj).mean()]
    assert wdl == pytest.approx(list(pp[0, 1]), abs=0.005)
    assert gi.max() <= MAX_GOALS
    emp = np.zeros((MAX_GOALS + 1,) * 2); np.add.at(emp, (gi, gj), 1); emp /= len(u)
    assert np.abs(emp - sm.pmf(0, 1)).max() < 0.004

def test_goal_difference_varies_so_tiebreaks_are_possible():
    _, sm = _model()
    gi, gj = sm.sample(0, 1, np.random.default_rng(1).random(50_000))
    assert len(np.unique(gi - gj)) > 5

# ---- knockout -----------------------------------------------------------------------
class AlwaysDraw:
    """Score stub: every 90-minute game finishes level; rates configurable."""
    def __init__(self, lam): self.lam = np.full((2, 2, 2), lam)
    def sample(self, i, j, u): return 1, 1

class StubShootout:
    def __init__(self, p): self.p = p
    def p_win(self, a, b, ea, eb): return self.p

def run(model, shoot, n=20_000, et=0.8, seed=0):
    rng = np.random.default_rng(seed)
    outs = [play_knockout(0, 1, model, shoot, {}, ["a", "b"], rng, et) for _ in range(n)]
    return outs

def test_draw_goes_to_extra_time_then_penalties():
    outs = run(AlwaysDraw(1.35), StubShootout(0.5))
    how = np.array([h for _, h in outs])
    # P(level after ET) = sum_k Poisson(m;k)^2 with m = 1.35*0.8/3 per side
    from scipy.stats import poisson
    m = 1.35 * 0.8 / 3; ks = np.arange(40)
    p_pen = float((poisson.pmf(ks, m) ** 2).sum())
    assert (how == "PEN").mean() == pytest.approx(p_pen, abs=0.01)
    assert (how == "90").sum() == 0

def test_zero_extra_time_scoring_sends_every_draw_to_penalties():
    outs = run(AlwaysDraw(1.35), StubShootout(0.5), n=2000, et=0.0)
    assert {h for _, h in outs} == {"PEN"}

def test_penalty_winner_follows_shootout_probability():
    outs = run(AlwaysDraw(0.0), StubShootout(0.7), n=20_000)
    assert np.mean([w == 0 for w, _ in outs]) == pytest.approx(0.7, abs=0.01)

def test_more_extra_time_scoring_means_fewer_penalties():
    f = lambda et: np.mean([h == "PEN" for _, h in run(AlwaysDraw(1.35), StubShootout(.5), 10_000, et)])
    assert f(0.4) > f(0.8) > f(1.2)

def test_shootout_model_is_symmetric():
    sm = ShootoutModel(0.14, 0.13, {"a": 0.62, "b": 0.45}, 8, {})
    assert sm.p_win("a", "b", 1900, 1700) + sm.p_win("b", "a", 1700, 1900) == pytest.approx(1)
    assert sm.p_win("a", "b", 1800, 1800) > 0.5          # better shoot-out record helps
    assert sm.p_win("x", "y", 1800, 1800) == pytest.approx(0.5)
