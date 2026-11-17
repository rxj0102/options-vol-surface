import numpy as np
import pandas as pd
import pytest

from pricing.heston import heston_price as heston_quad
from surface.chain import black76, black76_iv, discount_factor, parity_forward, smile_stats, surface
from surface.fit import fit_heston, fit_svi, heston_call, svi_density_g, svi_w

TRUE = dict(v0=0.02, kappa=3.0, theta=0.035, sigma_v=0.9, rho=-0.75)


def heston_chain(S=740.0, r=0.04, q=0.012, days=(21, 49, 112, 210), rel=0.002):
    rows = []
    for d in days:
        T = d / 365
        F, D = S * np.exp((r - q) * T), np.exp(-r * T)
        for K in np.arange(600.0, 840.0, 10.0):
            c = heston_call(F, K, T, D, *TRUE.values())[0]
            for kind, px in (("call", c), ("put", c - D * (F - K))):
                if px > 0.05:
                    rows.append(dict(expiry=f"E{d:03d}", type=kind, strike=K, T=T,
                                     bid=px * (1 - rel), ask=px * (1 + rel)))
    return pd.DataFrame(rows), S, r, q


@pytest.mark.parametrize("T", [14 / 365, 0.1, 0.5, 1.5])
def test_vectorized_heston_matches_quadrature(T):
    for K in (600.0, 700.0, 740.0, 800.0, 880.0):
        q = heston_quad(740.0, K, T, 0.0, *TRUE.values())
        v = heston_call(740.0, K, T, 1.0, *TRUE.values())[0]
        assert v == pytest.approx(q, abs=1e-4)


def test_black76_iv_round_trip():
    for kind in ("call", "put"):
        for K in (600.0, 740.0, 900.0):
            p = black76(745.0, K, 0.3, 0.99, 0.22, kind)
            assert black76_iv(p, 745.0, K, 0.3, 0.99, kind) == pytest.approx(0.22, abs=1e-8)


def test_parity_recovers_the_forward_given_the_discount_factor():
    S, r, q, T = 740.0, 0.04, 0.012, 0.25
    F, D = S * np.exp((r - q) * T), np.exp(-r * T)
    rows = []
    for K in np.arange(700.0, 780.0, 5.0):
        for kind in ("call", "put"):
            rows.append(dict(strike=K, type=kind, mid=black76(F, K, T, D, 0.2, kind)))
    F_hat, n = parity_forward(pd.DataFrame(rows), D)
    assert F_hat == pytest.approx(F, rel=1e-12)
    assert n == 5


def test_parity_forward_ignores_early_exercise_premium_away_from_the_money():
    # American puts are worth more than European ones, and the gap grows with
    # the strike. Adding such a premium deep in the money must not move the
    # forward, which is read off the strikes where C and P are closest.
    S, r, T = 740.0, 0.04, 0.5
    F, D = S * np.exp(r * T), np.exp(-r * T)
    rows = []
    for K in np.arange(600.0, 900.0, 5.0):
        eep = 0.02 * max(K - F - 30, 0.0)
        rows.append(dict(strike=K, type="call", mid=black76(F, K, T, D, 0.2, "call")))
        rows.append(dict(strike=K, type="put", mid=black76(F, K, T, D, 0.2, "put") + eep))
    F_hat, _ = parity_forward(pd.DataFrame(rows), D)
    assert F_hat == pytest.approx(F, rel=1e-9)


def test_discount_factor_interpolates_the_curve():
    rates = {"0.25": 0.04, "1.0": 0.05}
    assert discount_factor(0.625, rates) == pytest.approx(np.exp(-0.045 * 0.625))
    assert discount_factor(2.0, rates) == pytest.approx(np.exp(-0.05 * 2.0))


def test_heston_fit_recovers_known_parameters():
    ch, S, r, _ = heston_chain()
    iv, fwd = surface(ch, S, {'1.0': r})
    p = fit_heston(iv)
    for k, v in TRUE.items():
        assert p[k] == pytest.approx(v, rel=0.03), k


def test_quotes_without_both_sides_are_dropped():
    ch, S, r, _ = heston_chain(days=(49,))
    full, _ = surface(ch, S, {"1.0": r})
    ch.loc[ch.index[::3], "bid"] = 0.0
    iv, _ = surface(ch, S, {"1.0": r})
    assert len(iv) < len(full)


def test_flat_smile_has_zero_risk_reversal_and_butterfly():
    F, T, D = 740.0, 0.25, 0.99
    rows = []
    for K in np.arange(500.0, 1000.0, 5.0):
        kind = "put" if K < F else "call"
        rows.append(dict(expiry="E", type=kind, strike=K, T=T,
                         bid=black76(F, K, T, D, 0.2, kind) * 0.999,
                         ask=black76(F, K, T, D, 0.2, kind) * 1.001))
        other = "call" if kind == "put" else "put"
        rows.append(dict(expiry="E", type=other, strike=K, T=T,
                         bid=black76(F, K, T, D, 0.2, other) * 0.999,
                         ask=black76(F, K, T, D, 0.2, other) * 1.001))
    iv, _ = surface(pd.DataFrame(rows), F, {'1.0': -np.log(D) / T})
    s = smile_stats(iv).iloc[0]
    assert s["atm"] == pytest.approx(0.2, abs=1e-4)
    assert abs(s["rr25"]) < 1e-4 and abs(s["bf25"]) < 1e-4


def test_svi_fit_recovers_a_known_slice():
    x = (0.01, 0.12, -0.6, 0.02, 0.15)
    k = np.linspace(-0.4, 0.3, 60)
    w = svi_w(k, *x)
    assert np.abs(svi_w(k, *fit_svi(k, w)) - w).max() < 1e-7


def test_svi_butterfly_check_flags_the_known_counterexample():
    # Axel Vogt's raw SVI slice (Gatheral and Jacquier 2014) has butterfly
    # arbitrage; a typical equity slice does not.
    k = np.linspace(-1.5, 1.5, 2001)
    assert svi_density_g(k, -0.0410, 0.1331, 0.3060, 0.3586, 0.4153).min() < 0
    assert svi_density_g(k, 0.01, 0.12, -0.6, 0.02, 0.15).min() > 0


def test_svi_density_of_a_flat_slice_is_lognormal():
    from scipy.stats import norm
    from surface.fit import density_summary, svi_density
    w = 0.04  # flat slice: b = 0, total variance a
    k = np.linspace(-1, 1, 11)
    lognormal = norm.pdf((k + w / 2) / np.sqrt(w)) / np.sqrt(w)
    assert svi_density(k, w, 0.0, 0.0, 0.0, 0.1) == pytest.approx(lognormal, rel=1e-10)
    s = density_summary(dict(a=w, b=0.0, rho=0.0, m=0.0, sigma=0.1, k_min=-0.5, k_max=0.5))
    assert s["mass"] == pytest.approx(1, abs=1e-6)
    assert s["mean_over_forward"] == pytest.approx(1, abs=1e-6)
    assert s["p_below_90"] == pytest.approx(s["p_below_90_atm_lognormal"], abs=1e-6)


def test_svi_density_of_a_skewed_slice_prices_the_forward():
    from surface.fit import density_summary
    s = density_summary(dict(a=0.01, b=0.12, rho=-0.6, m=0.02, sigma=0.15, k_min=-0.4, k_max=0.3))
    assert s["mass"] == pytest.approx(1, abs=1e-4)
    assert s["mean_over_forward"] == pytest.approx(1, abs=1e-4)
    assert s["p_below_80"] > s["p_below_80_atm_lognormal"]  # negative skew fattens the left tail
