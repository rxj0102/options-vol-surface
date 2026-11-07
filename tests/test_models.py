import numpy as np
import pytest

from hedging.delta_hedge import simulate_delta_hedge
from integration.garch_integration import _mock_garch_forecasts
from pricing import bsm
from pricing.heston import heston_price

S, K, T, r, sig = 100.0, 105.0, 0.75, 0.04, 0.22


# ------------------------------------------------------------------ BSM
def test_put_call_parity():
    c = bsm.bsm_price(S, K, T, r, sig, "call")
    p = bsm.bsm_price(S, K, T, r, sig, "put")
    assert c - p == pytest.approx(S - K * np.exp(-r * T))


def test_greeks_match_finite_differences():
    h = 1e-4
    up = bsm.bsm_price(S + h, K, T, r, sig)
    dn = bsm.bsm_price(S - h, K, T, r, sig)
    mid = bsm.bsm_price(S, K, T, r, sig)
    assert bsm.delta(S, K, T, r, sig) == pytest.approx((up - dn) / (2 * h), rel=1e-5)
    assert bsm.gamma(S, K, T, r, sig) == pytest.approx((up - 2 * mid + dn) / h**2, rel=1e-3)
    dv = (bsm.bsm_price(S, K, T, r, sig + 1e-5) - bsm.bsm_price(S, K, T, r, sig - 1e-5)) / 2e-5
    assert bsm.vega(S, K, T, r, sig) == pytest.approx(dv / 100, rel=1e-5)
    dt = (bsm.bsm_price(S, K, T - 1e-5, r, sig) - bsm.bsm_price(S, K, T + 1e-5, r, sig)) / 2e-5
    assert bsm.theta(S, K, T, r, sig) == pytest.approx(dt / 365, rel=1e-4)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_implied_vol_round_trip(opt):
    price = bsm.bsm_price(S, K, T, r, sig, opt)
    assert bsm.implied_vol(price, S, K, T, r, opt) == pytest.approx(sig, abs=1e-6)


# --------------------------------------------------------------- Heston
HP = dict(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7)


def test_heston_reduces_to_bsm_with_tiny_vol_of_vol():
    for T_ in (0.25, 1.0, 5.0):
        h = heston_price(S, S, T_, r, 0.04, 2.0, 0.04, 0.01, 0.0)
        assert h == pytest.approx(bsm.bsm_price(S, S, T_, r, 0.2), abs=2e-3)


def test_heston_long_maturity_is_finite_and_increasing():
    prices = [heston_price(S, S, T_, r, 0.04, 0.5, 0.04, 1.0, -0.9) for T_ in (2, 5, 10, 20)]
    assert all(np.isfinite(prices)) and all(np.diff(prices) > 0)
    assert prices[-1] < S  # a call is worth less than the underlying


def test_heston_matches_monte_carlo_for_two_year_itm_call():
    """Regression: the 1993 formulation was off by ~1.5 here."""
    rng = np.random.default_rng(0)
    n, steps, T_, K_ = 100_000, 200, 2.0, 90.0
    v0, kappa, th, sv, rho = 0.09, 1.0, 0.04, 0.4, -0.8
    dt = T_ / steps
    x = np.full(n, np.log(S))
    v = np.full(n, v0)
    for _ in range(steps):
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        vp = np.maximum(v, 0)
        x += (r - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += kappa * (th - vp) * dt + sv * np.sqrt(vp * dt) * z2
    pay = np.exp(-r * T_) * np.maximum(np.exp(x) - K_, 0)
    exact = heston_price(S, K_, T_, r, v0, kappa, th, sv, rho)
    assert abs(exact - pay.mean()) < 4 * pay.std() / np.sqrt(n) + 0.05


def test_heston_put_call_parity_and_skew():
    c = heston_price(S, K, T, r, **HP)
    p = heston_price(S, K, T, r, **HP, option_type="put")
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-3)
    # negative spot/vol correlation -> downside skew
    ivs = [bsm.implied_vol(heston_price(S, k, T, r, **HP), S, k, T, r) for k in (85, 100, 115)]
    assert ivs[0] > ivs[1] > ivs[2]


# -------------------------------------------------------------- hedging
def test_delta_hedge_pnl_is_centred_and_attribution_closes():
    res = simulate_delta_hedge(n_paths=400, n_steps=60)
    se = res["total"].std() / np.sqrt(400)
    assert abs(res["total"].mean()) < 4 * se + 0.05
    # a hedged book is far tighter than the option premium
    assert res["total"].std() < 0.2 * res["C0"]
    assert abs(res["residual"].mean()) < 0.3
    assert res["residual"].std() < 0.5 * res["total"].std() + 0.1


def test_hedge_error_shrinks_with_rebalancing_frequency():
    coarse = simulate_delta_hedge(n_paths=400, n_steps=10, seed=1)["total"].std()
    fine = simulate_delta_hedge(n_paths=400, n_steps=120, seed=1)["total"].std()
    assert fine < 0.6 * coarse


# --------------------------------------------------------- GARCH mock
def test_mock_garch_vol_stays_near_target_level():
    df = _mock_garch_forecasts()
    assert 0.12 < df["garch_vol"].mean() < 0.25
