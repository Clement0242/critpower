"""Build the example PDF reports shown on the project page (docs/)."""

from pathlib import Path

import critpower as cp
from critpower.report import critical_report

DOCS = Path(__file__).resolve().parents[1] / "docs"

# 1. Laboratory: four time trials and an interval session (synthetic athlete).
d = cp.load_tte_example()
models = cp.compare_models(d["duration"], d["power"])
fit = models[0]
bal = cp.w_prime_balance(d["session_time"], d["session_power"], fit.cp, fit.w_prime)
for lang in ("fr", "en"):
    critical_report(fit, DOCS / f"example_power_{lang}.pdf", compare=models, session=bal,
                    session_intensity=d["session_power"], body_mass=72.0, lang=lang,
                    athlete="Athlète fictif" if lang == "fr" else "Synthetic athlete",
                    test="4 contre-la-montre, ergomètre" if lang == "fr" else "4 time trials, ergometer",
                    test_date="2026-10-12")

# 2. Field: critical speed from a season of GNSS recordings (SoccerMon, real data).
curve = cp.soccermon_session_curve()
window = [k for k in curve if 120 <= k <= 1200]
models = cp.compare_models(window, [curve[k] for k in window], kind="speed",
                           models=("hyperbolic", "linear_work", "linear_inverse"))
for lang in ("fr", "en"):
    critical_report(models[0], DOCS / f"example_speed_{lang}.pdf", compare=models, curve=curve, lang=lang,
                    athlete="SoccerMon TeamA-f20565cc",
                    test=("276 séances GNSS 10 Hz, 2020-2021" if lang == "fr" else "276 GNSS sessions at 10 Hz, 2020-2021"),
                    test_date="2021-11-13")
print("ok")
