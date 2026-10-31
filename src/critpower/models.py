"""Power-duration (or speed-duration) models and their fits.

The same mathematics serves power (W, CP and W' in J) and running speed
(m/s, critical speed CS and D' in m): only the units change, so the code
talks about an *intensity* and its *capacity above the critical value*.

Models
------
``"hyperbolic"``      P = CP + W'/t, non-linear least squares on P (Hill 1993)
``"linear_work"``     W = CP·t + W', work (or distance) against time (Moritani et al. 1981)
``"linear_inverse"``  P = CP + W'·(1/t), intensity against 1/time (Whipp et al. 1982)
``"three_parameter"`` P = CP + W'/(t + W'/(Pmax − CP)), Morton (1996): a finite Pmax at t = 0
``"omni"``            OmPD, Puchowicz et al. (2020): from a few seconds to hours, with a
                      long-duration decline A·ln(t/1800) beyond 30 min

Every fit returns the parameters, their standard errors, and the standard
error of the estimate (SEE) as a percentage of each parameter, a common way
to judge a fit (methodological choices are reviewed by Muniz-Pumares et al. 2019).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

MODELS = ("hyperbolic", "linear_work", "linear_inverse", "three_parameter", "omni")
T_CPMAX = 1800.0  # s, end of the "CP plateau" in the omni-domain model


# ------------------------------------------------------------------ equations

def hyperbolic(t: ArrayLike, cp: float, w_prime: float) -> np.ndarray:
    """Intensity sustainable for ``t`` seconds, 2-parameter model."""
    return cp + w_prime / np.asarray(t, dtype=float)


def three_parameter(t: ArrayLike, cp: float, w_prime: float, pmax: float) -> np.ndarray:
    """Morton (1996): t = W'/(P − CP) − W'/(Pmax − CP), solved for P."""
    t = np.asarray(t, dtype=float)
    return cp + w_prime / (t + w_prime / (pmax - cp))


def omni(t: ArrayLike, cp: float, w_prime: float, pmax: float, a: float) -> np.ndarray:
    """Omni-domain power-duration model (Puchowicz, Baker & Clarke 2020)."""
    t = np.asarray(t, dtype=float)
    p = w_prime / t * (1.0 - np.exp(-t * (pmax - cp) / w_prime)) + cp
    return np.where(t > T_CPMAX, p - a * np.log(t / T_CPMAX), p)


def time_to_exhaustion(intensity: ArrayLike, cp: float, w_prime: float, pmax: float | None = None) -> np.ndarray:
    """Time (s) an intensity above CP can be held (2- or 3-parameter model)."""
    p = np.asarray(intensity, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = w_prime / (p - cp)
        if pmax is not None:
            t = t - w_prime / (pmax - cp)
    return np.where(p > cp, t, np.inf)


# ------------------------------------------------------------------ fits

@dataclass
class CriticalFit:
    """Result of a power- (or speed-) duration fit.

    ``cp`` is the critical power (W) or speed (m/s); ``w_prime`` the finite
    capacity above it, W' (J) or D' (m). ``pmax`` (3-parameter, omni) is the
    intensity at t → 0 and ``a`` the long-duration decline (omni).
    ``see_pct`` gives the SEE of each parameter in % of its value.
    """

    model: str
    cp: float
    w_prime: float
    pmax: float | None = None
    a: float | None = None
    se: dict[str, float] = field(default_factory=dict)
    duration: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    intensity: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    kind: str = "power"

    @property
    def see_pct(self) -> dict[str, float]:
        vals = {"cp": self.cp, "w_prime": self.w_prime, "pmax": self.pmax, "a": self.a}
        return {k: 100.0 * v / abs(vals[k]) for k, v in self.se.items() if vals.get(k)}

    def predict(self, t: ArrayLike) -> np.ndarray:
        """Intensity sustainable for durations ``t`` (s)."""
        if self.model == "three_parameter":
            return three_parameter(t, self.cp, self.w_prime, self.pmax)
        if self.model == "omni":
            return omni(t, self.cp, self.w_prime, self.pmax, self.a)
        return hyperbolic(t, self.cp, self.w_prime)

    @property
    def residuals(self) -> np.ndarray:
        return self.intensity - self.predict(self.duration)

    @property
    def r2(self) -> float:
        y = self.intensity
        return float(1 - np.sum(self.residuals**2) / np.sum((y - y.mean()) ** 2))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2)))

    def time_to_exhaustion(self, intensity: ArrayLike) -> np.ndarray:
        if self.model in ("hyperbolic", "linear_work", "linear_inverse"):
            return time_to_exhaustion(intensity, self.cp, self.w_prime)
        if self.model == "three_parameter":
            return time_to_exhaustion(intensity, self.cp, self.w_prime, self.pmax)
        # omni: no closed form, invert numerically on a grid
        grid = np.logspace(0, 5, 4000)
        p = self.predict(grid)
        x = np.asarray(intensity, dtype=float)
        return np.interp(-x, -p, grid, left=np.inf, right=np.inf)

    @property
    def units(self) -> tuple[str, str]:
        return ("W", "J") if self.kind == "power" else ("m/s", "m")

    def summary(self) -> str:
        u, c = self.units
        name = ("CP", "W'") if self.kind == "power" else ("CS", "D'")
        lines = [f"Critical {'power' if self.kind == 'power' else 'speed'} ({self.model}, {self.duration.size} efforts)",
                 f"  {name[0]:<4} {self.cp:10.2f} {u}" + (f"   SEE {self.see_pct.get('cp', float('nan')):.1f} %" if self.se else ""),
                 f"  {name[1]:<4} {self.w_prime:10.1f} {c}" + (f"   SEE {self.see_pct.get('w_prime', float('nan')):.1f} %" if self.se else "")]
        if self.pmax is not None:
            lines.append(f"  max  {self.pmax:10.2f} {u}")
        if self.a is not None:
            lines.append(f"  A    {self.a:10.2f} {u}")
        lines.append(f"  R²   {self.r2:10.4f}   RMSE {self.rmse:.3f} {u}")
        return "\n".join(lines)


def _clean(duration, intensity):
    t = np.asarray(duration, dtype=float).ravel()
    p = np.asarray(intensity, dtype=float).ravel()
    if t.shape != p.shape:
        raise ValueError("duration and intensity must have the same length")
    ok = np.isfinite(t) & np.isfinite(p) & (t > 0)
    t, p = t[ok], p[ok]
    order = np.argsort(t)
    return t[order], p[order]


def _ols(x, y):
    """Slope, intercept and their standard errors."""
    n = x.size
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    s2 = np.sum(res**2) / (n - 2) if n > 2 else 0.0
    sxx = np.sum((x - x.mean()) ** 2)
    se_b = np.sqrt(s2 / sxx)
    se_a = np.sqrt(s2 * (1 / n + x.mean() ** 2 / sxx))
    return b, a, se_b, se_a


def fit_critical(duration: ArrayLike, intensity: ArrayLike, model: str = "hyperbolic",
                 kind: str = "power") -> CriticalFit:
    """Fit a power- (or speed-) duration model to maximal efforts.

    Parameters
    ----------
    duration:
        Duration of each maximal effort (s) — time to exhaustion, or the
        length of a time trial.
    intensity:
        Mean power (W) or mean speed (m/s) of each effort.
    model:
        One of :data:`MODELS`.
    kind:
        ``"power"`` or ``"speed"``: only changes names and units.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}: choose among {MODELS}")
    if kind not in ("power", "speed"):
        raise ValueError("kind must be 'power' or 'speed'")
    t, p = _clean(duration, intensity)
    need = {"three_parameter": 4, "omni": 5}.get(model, 3)
    if t.size < need:
        raise ValueError(f"model {model!r} needs at least {need} efforts (got {t.size})")

    if model == "linear_work":
        b, a, se_b, se_a = _ols(t, p * t)
        return CriticalFit(model, b, a, se={"cp": se_b, "w_prime": se_a}, duration=t, intensity=p, kind=kind)
    if model == "linear_inverse":
        b, a, se_b, se_a = _ols(1.0 / t, p)
        return CriticalFit(model, a, b, se={"cp": se_a, "w_prime": se_b}, duration=t, intensity=p, kind=kind)

    # Non-linear models start from the linear work-time estimate.
    b0, a0, _, _ = _ols(t, p * t)
    a0 = max(a0, 1e-3 * b0 * t.max())
    if model == "hyperbolic":
        popt, pcov = curve_fit(hyperbolic, t, p, p0=[b0, a0], bounds=([0, 0], np.inf), maxfev=20000)
        names = ["cp", "w_prime"]
    elif model == "three_parameter":
        popt, pcov = curve_fit(three_parameter, t, p, p0=[b0, a0, p.max() * 1.5],
                               bounds=([0, 0, 0], np.inf), maxfev=20000)
        names = ["cp", "w_prime", "pmax"]
    else:
        popt, pcov = curve_fit(omni, t, p, p0=[b0, a0, p.max() * 1.2, 0.05 * b0],
                               bounds=([0, 0, 0, 0], np.inf), maxfev=50000)
        names = ["cp", "w_prime", "pmax", "a"]
    se = dict(zip(names, np.sqrt(np.clip(np.diag(pcov), 0, None)).tolist()))
    vals = dict(zip(names, popt.tolist()))
    return CriticalFit(model, vals["cp"], vals["w_prime"], vals.get("pmax"), vals.get("a"), se, t, p, kind)


def compare_models(duration: ArrayLike, intensity: ArrayLike, kind: str = "power",
                   models: tuple[str, ...] = ("hyperbolic", "linear_work", "linear_inverse", "three_parameter")
                   ) -> list[CriticalFit]:
    """Fit several models; returned in order of the summed SEE% of CP and W' (best first).

    Models that cannot be fitted (too few efforts) are left out. A model with
    fewer than two degrees of freedom left (e.g. 3 parameters on 4 efforts)
    goes through almost every point: its SEE says nothing, so it is ranked
    after the others.
    """
    fits = []
    for m in models:
        try:
            fits.append(fit_critical(duration, intensity, m, kind))
        except (ValueError, RuntimeError):
            continue
    k = {"three_parameter": 3, "omni": 4}
    return sorted(fits, key=lambda f: (f.duration.size - k.get(f.model, 2) < 2,
                                       f.see_pct.get("cp", np.inf) + f.see_pct.get("w_prime", np.inf)))
