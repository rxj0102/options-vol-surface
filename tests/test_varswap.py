import numpy as np
import pandas as pd

from surface.chain import black76
from surface.varswap import _strip, slice_variance, svi_variance


def flat_chain(sigma=0.2, T=30 / 365, F=100.0, D=0.996, step=0.5):
    K = np.arange(40.0, 200.0 + step, step)
    rows = []
    for kind in ("call", "put"):
        p = black76(F, K, T, D, sigma, kind)
        for k, v in zip(K, p):
            rows.append(dict(expiry="x", type=kind, strike=k, bid=max(v - 0.005, 0.0), ask=v + 0.005))
    return pd.DataFrame(rows), F, D, T


def test_flat_smile_recovers_its_vol():
    chain, F, D, T = flat_chain()
    v = slice_variance(chain, F, D, T)
    assert abs(np.sqrt(v["var"]) - 0.2) < 0.002


def test_svi_integral_on_a_flat_smile():
    T, sigma = 30 / 365, 0.2
    flat = dict(a=sigma ** 2 * T, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    assert abs(np.sqrt(svi_variance(flat, T, k_lo=-1.5, k_hi=1.0)) - sigma) < 0.002


def test_strip_stops_after_two_zero_bids():
    side = pd.DataFrame(dict(strike=[100, 99, 98, 97, 96, 95], bid=[1.0, 0.5, 0.0, 0.2, 0.0, 0.0],
                             ask=[1.1, 0.6, 0.1, 0.3, 0.1, 0.1], mid=[1.05, 0.55, 0.05, 0.25, 0.05, 0.05]))
    kept = _strip(side, ascending=False)
    assert list(kept["strike"]) == [100, 99, 97]
