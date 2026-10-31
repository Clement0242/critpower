"""critpower — critical power and critical speed, from lab tests or field data.

    import critpower as cp
    fit = cp.fit_critical([180, 300, 600, 900], [402, 351, 315, 302], model="hyperbolic")
    print(fit.summary())

The same code serves power (CP, W') and running speed (CS, D'): pass
``kind="speed"``. Field data go through :func:`mean_maximal` and
:func:`fit_field`; :func:`w_prime_balance` follows W' through a session; and
``critpower.report.critical_report`` writes a designed PDF report.
"""

from .datasets import load_soccermon_mms, load_tte_example, soccermon_session_curve
from .field import AllOutResult, all_out_test, best_of, fit_field, mean_maximal
from .models import (MODELS, CriticalFit, compare_models, fit_critical, hyperbolic, omni,
                     three_parameter, time_to_exhaustion)
from .wbal import WBalance, w_prime_balance

__version__ = "0.1.0"

__all__ = [
    "MODELS", "CriticalFit", "fit_critical", "compare_models", "hyperbolic", "three_parameter", "omni",
    "time_to_exhaustion", "mean_maximal", "best_of", "fit_field", "all_out_test", "AllOutResult",
    "w_prime_balance", "WBalance", "load_tte_example", "load_soccermon_mms", "soccermon_session_curve",
    "__version__",
]
