import numpy as np
import pytest

import critpower as cp

T = np.array([150.0, 270.0, 480.0, 840.0])


def test_exact_recovery_all_models():
    p = cp.hyperbolic(T, 280, 20000)
    for m in ("hyperbolic", "linear_work", "linear_inverse"):
        f = cp.fit_critical(T, p, m)
        assert f.cp == pytest.approx(280, rel=1e-6)
        assert f.w_prime == pytest.approx(20000, rel=1e-5)


def test_three_parameter_recovery():
    t = np.array([60.0, 120, 240, 480, 900, 1200])
    p = cp.three_parameter(t, 250, 18000, 900)
    f = cp.fit_critical(t, p, "three_parameter")
    assert (f.cp, f.w_prime, f.pmax) == pytest.approx((250, 18000, 900), rel=1e-4)


def test_omni_recovery():
    t = np.array([5.0, 30, 60, 180, 300, 600, 1200, 2400, 3600])
    p = cp.omni(t, 260, 20000, 1100, 25)
    f = cp.fit_critical(t, p, "omni")
    assert f.cp == pytest.approx(260, rel=1e-3)
    assert f.a == pytest.approx(25, rel=1e-2)


def test_hyperbolic_equals_linear_inverse():
    d = cp.load_tte_example()
    a = cp.fit_critical(d["duration"], d["power"], "hyperbolic")
    b = cp.fit_critical(d["duration"], d["power"], "linear_inverse")
    assert a.cp == pytest.approx(b.cp, rel=1e-6) and a.w_prime == pytest.approx(b.w_prime, rel=1e-5)
    assert a.see_pct["cp"] == pytest.approx(b.see_pct["cp"], rel=1e-3)


def test_compare_ranks_overfitted_model_last():
    d = cp.load_tte_example()
    fits = cp.compare_models(d["duration"], d["power"])
    assert fits[-1].model == "three_parameter"  # 3 parameters on 4 efforts
    assert abs(fits[0].cp - 280) < 5


def test_time_to_exhaustion_inverts_predict():
    f = cp.fit_critical(T, cp.hyperbolic(T, 280, 20000))
    assert f.time_to_exhaustion([f.predict([300])[0]])[0] == pytest.approx(300, rel=1e-6)
    assert np.isinf(f.time_to_exhaustion([270])[0])


def test_speed_units_and_validation():
    f = cp.fit_critical(T, cp.hyperbolic(T, 4.2, 180), kind="speed")
    assert f.units == ("m/s", "m")
    with pytest.raises(ValueError):
        cp.fit_critical(T[:2], [1, 2])
    with pytest.raises(ValueError):
        cp.fit_critical(T, T, model="nope")


def test_mean_maximal():
    t = np.arange(0, 600, 1.0)
    x = np.full(t.size, 100.0)
    x[100:160] = 400.0  # one minute hard
    c = cp.mean_maximal(t, x, [1, 60, 120, 1000])
    assert c[1.0] == 400 and c[60.0] == pytest.approx(400) and c[120.0] == pytest.approx(250)
    assert 1000.0 not in c  # longer than the recording


def test_mean_maximal_gap_is_zero():
    t = np.concatenate([np.arange(0, 100, 1.0), np.arange(200, 300, 1.0)])
    c = cp.mean_maximal(t, np.full(t.size, 5.0), [150])
    assert c[150.0] < 5.0


def test_all_out_test():
    t = np.arange(0, 180, 1.0)
    p = 300 + 500 * np.exp(-t / 25)  # decays to CP 300 W
    r = cp.all_out_test(t, p)
    assert r.cp == pytest.approx(300, abs=1)
    assert r.w_prime == pytest.approx(500 * 25, rel=0.05)


def test_wbal_differential_depletion_and_ceiling():
    t = np.arange(0, 300, 1.0)
    p = np.where(t < 100, 380.0, 200.0)
    b = cp.w_prime_balance(t, p, 280, 20000)
    assert b.balance[99] == pytest.approx(20000 - 99 * 100)  # first sample has dt = 0
    assert b.balance.max() <= 20000 and b.balance[-1] > b.balance[100]


def test_wbal_integral_tau():
    t = np.arange(0, 200, 1.0)
    b = cp.w_prime_balance(t, np.where(t < 50, 400.0, 180.0), 280, 20000, model="integral")
    assert b.tau == pytest.approx(546 * np.exp(-0.01 * 100) + 316, rel=1e-6)
    with pytest.raises(ValueError):
        cp.w_prime_balance(t, t, 280, 20000, model="x")


def test_soccermon_example():
    days = cp.load_soccermon_mms()
    assert len(days) > 250
    curve = cp.soccermon_session_curve()
    assert 6 < curve[1.0] < 9
    f = cp.fit_field(curve, kind="speed")
    assert 1.5 < f.cp < 3.5


def test_report(tmp_path):
    pytest.importorskip("weasyprint")
    from critpower.report import critical_report
    d = cp.load_tte_example()
    fits = cp.compare_models(d["duration"], d["power"])
    bal = cp.w_prime_balance(d["session_time"], d["session_power"], fits[0].cp, fits[0].w_prime)
    for lang in ("fr", "en"):
        out = critical_report(fits[0], tmp_path / f"r_{lang}.pdf", compare=fits, session=bal,
                              session_intensity=d["session_power"], body_mass=70, lang=lang)
        assert out.read_bytes()[:4] == b"%PDF"
    curve = cp.soccermon_session_curve()
    out = critical_report(cp.fit_field(curve, kind="speed"), tmp_path / "s.pdf", curve=curve)
    assert out.stat().st_size > 10000
