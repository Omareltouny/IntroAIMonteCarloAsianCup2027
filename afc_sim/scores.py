"""Score-generation layer: turn W/D/L probabilities into scorelines.

The XGBoost engine only predicts the result class.  Goal-difference and goals-scored
tiebreakers need actual scores, so for every ordered pairing we

1. fit independent-Poisson rates (lam_home, lam_away) whose implied W/D/L probabilities
   match the engine's probabilities (least squares on the two free probabilities);
2. build the joint score pmf  P(a, b) = Poisson(a; lam_h) * Poisson(b; lam_a) * w[class(a, b)]
   where the class weights w make the W/D/L marginals *exactly* the engine's.

So the engine decides who wins, and the Poisson model decides the margin and total.
The fitted rates are also what drive extra-time scoring in the knockout rounds.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson

MAX_GOALS = 12


def _class_masks(maxg: int = MAX_GOALS):
    a = np.arange(maxg + 1)
    A, B = np.meshgrid(a, a, indexing="ij")          # A = home goals, B = away goals
    return A > B, A == B, A < B


_HW, _DR, _AW = _class_masks()


def poisson_grid(lh: float, la: float, maxg: int = MAX_GOALS) -> np.ndarray:
    g = np.arange(maxg + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    return M / M.sum()


def poisson_wdl(lh: float, la: float) -> np.ndarray:
    M = poisson_grid(lh, la)
    return np.array([M[_HW].sum(), M[_DR].sum(), M[_AW].sum()])


def fit_rates(p_wdl: np.ndarray) -> tuple[float, float, float]:
    """Return (lam_home, lam_away, max_abs_error) matching Poisson W/D/L to ``p_wdl``."""
    target = np.asarray(p_wdl, float)

    def resid(x):
        return (poisson_wdl(np.exp(x[0]), np.exp(x[1])) - target)[:2]

    best = None
    for x0 in ([0.3, 0.1], [0.0, 0.0], [0.8, -0.5], [-0.3, 0.5]):   # multi-start; cheap
        s = least_squares(resid, x0, bounds=([-3, -3], [2.2, 2.2]))
        if best is None or s.cost < best.cost:
            best = s
        if best.cost < 1e-14:
            break
    lh, la = np.exp(best.x)
    err = float(np.abs(poisson_wdl(lh, la) - target).max())
    return float(lh), float(la), err


def score_pmf(p_wdl: np.ndarray, lh: float, la: float) -> np.ndarray:
    """Joint pmf over (home goals, away goals) whose W/D/L marginals equal ``p_wdl``."""
    M = poisson_grid(lh, la)
    out = np.zeros_like(M)
    for mask, p in zip((_HW, _DR, _AW), np.asarray(p_wdl, float) / np.sum(p_wdl)):
        mass = M[mask].sum()
        out[mask] = M[mask] * (p / mass) if mass > 0 else 0.0
    return out / out.sum()


class ScoreModel:
    """Pre-computed score distributions for every ordered pairing in a ``MatchEngine``."""

    def __init__(self, pair_probs: np.ndarray):
        n = pair_probs.shape[0]
        self.n = n
        self.lam = np.full((n, n, 2), np.nan)
        self.fit_error = np.zeros((n, n))
        self._cdf = np.zeros((n, n, (MAX_GOALS + 1) ** 2))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p = pair_probs[i, j]
                lh, la, err = fit_rates(p)
                self.lam[i, j] = (lh, la)
                self.fit_error[i, j] = err
                self._cdf[i, j] = np.cumsum(score_pmf(p, lh, la).ravel())
        self._cdf[..., -1] = 1.0

    def sample(self, i, j, u):
        """Vectorised draw of (goals_i, goals_j) for pairings (i, j) with uniforms ``u``."""
        i, j, u = np.broadcast_arrays(np.asarray(i), np.asarray(j), np.asarray(u, float))
        cdf = self._cdf[i, j]
        k = (cdf < u[..., None]).sum(-1)
        return k // (MAX_GOALS + 1), k % (MAX_GOALS + 1)

    def pmf(self, i, j) -> np.ndarray:
        c = self._cdf[i, j]
        return np.diff(np.r_[0.0, c]).reshape(MAX_GOALS + 1, MAX_GOALS + 1)
