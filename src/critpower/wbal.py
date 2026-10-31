"""W' balance: how much of the capacity above CP is left, moment by moment.

Two published models:

``"differential"`` (default; Clarke & Skiba 2013, Skiba et al. 2014)
    Above CP, W'bal falls by (P − CP)·dt. Below CP, it recovers at a rate
    proportional to what is missing and to how far below CP the athlete is:
    dW'bal/dt = (W' − W'bal)/W' · (CP − P). Unit-agnostic: works for speed
    (CS, D') as well as power.
``"integral"`` (Skiba et al. 2012)
    W'bal = W' − Σ W'expended(u)·exp(−(t − u)/τ), with
    τ = 546·exp(−0.01·D_CP) + 316 s and D_CP the mean gap below CP during
    recovery. τ was established in cycling, in watts: for speed, pass ``tau``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class WBalance:
    time: np.ndarray
    balance: np.ndarray
    w_prime: float
    model: str
    tau: float | None = None

    @property
    def minimum(self) -> float:
        return float(np.min(self.balance))

    @property
    def minimum_pct(self) -> float:
        return 100.0 * self.minimum / self.w_prime

    def time_below(self, fraction: float) -> float:
        """Seconds spent with less than ``fraction`` of W' left."""
        dt = np.diff(self.time, append=self.time[-1])
        return float(dt[self.balance < fraction * self.w_prime].sum())


def w_prime_balance(time: ArrayLike, intensity: ArrayLike, cp: float, w_prime: float, *,
                    model: str = "differential", tau: float | None = None) -> WBalance:
    """W' balance over a session (``time`` in s, ``intensity`` in W or m/s)."""
    t = np.asarray(time, dtype=float).ravel()
    x = np.asarray(intensity, dtype=float).ravel()
    order = np.argsort(t)
    t, x = t[order], np.nan_to_num(x[order])
    dt = np.diff(t, prepend=t[0])
    bal = np.empty(t.size)

    if model == "differential":
        b = w_prime
        for i in range(t.size):
            if x[i] > cp:
                b -= (x[i] - cp) * dt[i]
            else:
                b += (w_prime - b) / w_prime * (cp - x[i]) * dt[i]
            bal[i] = min(b, w_prime)
        return WBalance(t, bal, w_prime, model)

    if model == "integral":
        if tau is None:
            below = x < cp
            d_cp = float(np.mean(cp - x[below])) if below.any() else 0.0
            tau = 546.0 * np.exp(-0.01 * d_cp) + 316.0
        spent = 0.0
        for i in range(t.size):
            spent = spent * np.exp(-dt[i] / tau) + max(x[i] - cp, 0.0) * dt[i]
            bal[i] = w_prime - spent
        return WBalance(t, bal, w_prime, model, float(tau))

    raise ValueError("model must be 'differential' or 'integral'")
