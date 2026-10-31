"""Example data shipped with the package.

* :func:`load_tte_example` — a **synthetic** cycling test: four time-to-exhaustion
  trials and a session, generated from CP = 280 W and W' = 20 kJ with
  measurement noise. It exists to run the examples and the tests; it is not a
  real athlete.
* :func:`load_soccermon_mms` — **real** data: mean-maximal speed curves of one
  player of an elite women's football team, every training session and match
  of 2020-2021 recorded at 10 Hz by GNSS. From SoccerMon (Midoglu et al. 2024,
  *Scientific Data* 11:553; Zenodo 10033832, CC-BY 4.0). The player is the
  pseudonymous identifier of the dataset; only the curves are kept (built by
  ``examples/build_soccermon_example.py``).
"""

from __future__ import annotations

import csv
from importlib import resources

import numpy as np


def load_tte_example() -> dict:
    """Synthetic cycling test: ``duration`` (s) and ``power`` (W) of four trials,
    plus a 40-min interval session (``session_time``, ``session_power``)."""
    rng = np.random.default_rng(7)
    cp, wp = 280.0, 20000.0
    duration = np.array([150.0, 270.0, 480.0, 840.0])
    power = cp + wp / duration + rng.normal(0, 3, duration.size)
    t = np.arange(0, 2400.0, 1.0)
    p = np.full(t.size, 190.0)
    for k in range(5):  # 5 × 2 min at 350 W, 3 min easy between them, after a warm-up
        s = 600 + k * 300
        p[(t >= s) & (t < s + 120)] = 350.0
    p = p + rng.normal(0, 8, t.size)
    return {"duration": duration.round(0), "power": power.round(1), "session_time": t,
            "session_power": p.round(1), "truth": {"cp": cp, "w_prime": wp}}


def load_soccermon_mms(max_speed: float = 9.0) -> dict[str, dict[float, float]]:
    """Mean-maximal speed (m/s) per duration (s), one curve per recorded day.

    A handful of recordings carry GNSS faults (speeds pinned at the 10 m/s
    ceiling for minutes); days whose best 1-s speed exceeds ``max_speed`` are
    left out (pass ``max_speed=None`` to keep everything).
    """
    out: dict[str, dict[float, float]] = {}
    with resources.files(__package__).joinpath("data/soccermon_mms.csv").open() as f:
        for row in csv.DictReader(f):
            day = out.setdefault(row["date"], {})
            d, v = float(row["duration_s"]), float(row["speed_ms"])
            day[d] = max(day.get(d, 0.0), v)  # two recordings on one day: keep the best
    if max_speed is not None:
        out = {k: c for k, c in out.items() if c.get(1.0, 0.0) <= max_speed}
    return out


def soccermon_session_curve() -> dict[float, float]:
    """The best mean-maximal speed curve over all of the player's sessions."""
    from .field import best_of
    return best_of(list(load_soccermon_mms().values()))
