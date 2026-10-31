"""Critical power or speed from field data: the 3-min all-out test, and
mean-maximal curves of training or match recordings.

* :func:`all_out_test` — Vanhatalo, Doust & Burnley (2007): during a 3-min
  all-out effort, the intensity falls to CP; the mean of the last 30 s is CP
  and the work done above it is W'.
* :func:`mean_maximal` — the best mean intensity over each duration
  (the "mean-maximal power" or "maximal mean speed" curve).
* :func:`fit_field` — mean-maximal curve, then a model fit: CP from
  training files (Smyth & Muniz-Pumares 2020 for running), or CS from GPS
  match data. Field efforts are not always maximal: the result is a lower
  bound and should be read as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .models import CriticalFit, fit_critical


def _uniform(time: ArrayLike, intensity: ArrayLike, dt: float | None) -> tuple[np.ndarray, float]:
    t = np.asarray(time, dtype=float).ravel()
    x = np.asarray(intensity, dtype=float).ravel()
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    order = np.argsort(t)
    t, x = t[order], x[order]
    if dt is None:
        dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1] + dt / 2, dt)
    # Gaps count as zero intensity (a stopped recording is not an effort).
    xi = np.interp(grid, t, x)
    gap = np.diff(t) > 5 * dt
    for i in np.flatnonzero(gap):
        xi[(grid > t[i]) & (grid < t[i + 1])] = 0.0
    return xi, dt


def mean_maximal(time: ArrayLike, intensity: ArrayLike,
                 durations: ArrayLike = (5, 10, 30, 60, 120, 180, 300, 420, 600, 900, 1200),
                 dt: float | None = None) -> dict[float, float]:
    """Best mean intensity over each duration (s) of one recording.

    The recording is resampled on a regular grid; rolling means come from a
    cumulative sum, so a match at 10 Hz is processed in milliseconds.
    """
    x, dt = _uniform(time, intensity, dt)
    c = np.concatenate(([0.0], np.cumsum(x)))
    out = {}
    for d in np.asarray(durations, dtype=float):
        n = int(round(d / dt))
        if 1 <= n <= x.size:
            out[float(d)] = float(np.max(c[n:] - c[:-n]) / n)
    return out


def best_of(curves: list[dict[float, float]]) -> dict[float, float]:
    """Combine mean-maximal curves of several recordings (best value per duration)."""
    out: dict[float, float] = {}
    for c in curves:
        for d, v in c.items():
            out[d] = max(out.get(d, -np.inf), v)
    return dict(sorted(out.items()))


def fit_field(curve: dict[float, float], *, min_duration: float = 120, max_duration: float = 1200,
              model: str = "hyperbolic", kind: str = "power") -> CriticalFit:
    """Fit a model to the part of a mean-maximal curve between two durations.

    The defaults (2 to 20 min) keep efforts long enough to be dominated by
    the CP/W' mechanisms and short enough to be maximal in a recording.
    """
    d = np.array([k for k in curve if min_duration <= k <= max_duration], dtype=float)
    if d.size < 3:
        raise ValueError("fewer than 3 durations in the chosen range")
    v = np.array([curve[k] for k in d])
    return fit_critical(d, v, model, kind)


@dataclass
class AllOutResult:
    cp: float
    w_prime: float
    end_window: float
    kind: str = "power"


def all_out_test(time: ArrayLike, intensity: ArrayLike, *, end_window: float = 30.0,
                 kind: str = "power") -> AllOutResult:
    """3-min all-out test (Vanhatalo et al. 2007): CP = mean of the last ``end_window`` s,
    W' = work done above CP over the test. For running, CS and D' (Pettitt et al. 2012)."""
    t = np.asarray(time, dtype=float).ravel()
    x = np.asarray(intensity, dtype=float).ravel()
    order = np.argsort(t)
    t, x = t[order], x[order]
    end = t >= t[-1] - end_window
    if end.sum() < 3:
        raise ValueError("not enough samples in the end window")
    cp = float(np.mean(x[end]))
    above = np.clip(x - cp, 0, None)
    w_prime = float(np.trapezoid(above, t)) if hasattr(np, "trapezoid") else float(np.trapz(above, t))
    return AllOutResult(cp, w_prime, end_window, kind)
