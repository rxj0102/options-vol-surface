import numpy as np
import pandas as pd
import pytest

from integration.vrp import (H, blocks, garch_filter, garch_forecasts,
                             garch_horizon_variance, nw_se, ols_nw)


def fake_history(n=2600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1990-01-02", periods=n)
    omega, a, b = 2e-6, 0.08, 0.9
    s2, r = 1e-4, np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + a * r[t] ** 2 + b * s2
    spx = 300 * np.exp(np.cumsum(r))
    vix = 100 * np.sqrt(252 * 1e-4) * np.ones(n)
    df = pd.DataFrame({"vix": vix, "spx": spx}, index=idx)
    df["r"] = np.log(df["spx"]).diff()
    return df.dropna()


def test_filter_matches_the_recursion():
    r = np.random.default_rng(1).normal(0, 0.01, 200)
    s = np.empty(201)
    s[0] = 3e-4
    for t in range(200):
        s[t + 1] = 1e-6 + 0.07 * r[t] ** 2 + 0.91 * s[t]
    assert np.allclose(garch_filter(r, 1e-6, 0.07, 0.91, 3e-4), s, rtol=0, atol=1e-15)


def test_horizon_variance_matches_iterated_expectation():
    omega, a, b, s = 2e-6, 0.08, 0.9, 4e-4
    e, path = s, []
    for _ in range(H):
        path.append(e)
        e = omega + (a + b) * e
    assert garch_horizon_variance(s, omega, a, b) == pytest.approx(np.mean(path), rel=1e-12)


def test_forecasts_do_not_use_later_returns():
    df = fake_history()
    g = garch_forecasts(df)
    t = int(np.argmax(g.notna().to_numpy())) + 30  # a day inside the first forecast year
    shocked = df.copy()
    shocked.iloc[t + 1:, shocked.columns.get_loc("r")] *= 3.0
    g2 = garch_forecasts(shocked)
    assert g2.iloc[t] == pytest.approx(g.iloc[t], rel=1e-12)
    assert g2.iloc[t + 1] != pytest.approx(g.iloc[t + 1], rel=1e-6)


def test_realized_variance_starts_the_day_after_the_forecast():
    df = fake_history()
    g = garch_forecasts(df)
    b = blocks(df, g)
    t0 = df.index.get_loc(b.index[0])
    r = df["r"].to_numpy()
    assert b["realized"].iloc[0] == pytest.approx(252 * np.mean(r[t0 + 1:t0 + 1 + H] ** 2))
    assert b["trailing"].iloc[0] == pytest.approx(252 * np.mean(r[t0 - H + 1:t0 + 1] ** 2))
    assert (np.diff([df.index.get_loc(d) for d in b.index]) == H).all()


def test_newey_west_reduces_to_iid_without_lags():
    x = np.random.default_rng(2).normal(size=500)
    assert nw_se(x, lags=0) == pytest.approx(x.std() / np.sqrt(500))


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 2))
    y = 0.5 + X @ [1.5, -2.0] + 0.01 * rng.normal(size=400)
    beta, se, r2 = ols_nw(y, X)
    assert beta == pytest.approx([0.5, 1.5, -2.0], abs=0.01)
    assert r2 > 0.99
