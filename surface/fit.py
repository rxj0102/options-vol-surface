"""Fit the surface two ways: Heston across all expiries, raw SVI per expiry.

Heston is one set of five parameters for the whole surface, so its errors show
what the model cannot do. SVI (Gatheral 2004) fits each expiry with its own
five parameters; it is the usual practitioner smile and the benchmark for how
well a smooth curve can fit these quotes at all. Both are checked for static
arbitrage: butterfly (density >= 0 within each expiry) and calendar (total
variance non-decreasing in maturity at fixed log-moneyness).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pricing.heston import heston_char_func
from surface.chain import black76, black76_iv, black76_vega

# Gauss-Legendre nodes on log-spaced segments of (0, 600]; checked against
# scipy quad in tests/test_surface.py.
_EDGES = np.array([0.0, 0.5, 2, 6, 15, 35, 80, 170, 350, 600])
_x, _w = np.polynomial.legendre.leggauss(40)
_PHI = np.concatenate([0.5 * (b - a) * _x + 0.5 * (b + a) for a, b in zip(_EDGES[:-1], _EDGES[1:])])
_W = np.concatenate([0.5 * (b - a) * _w for a, b in zip(_EDGES[:-1], _EDGES[1:])])

PARAMS = ("v0", "kappa", "theta", "sigma_v", "rho")
LOWER = np.array([1e-4, 0.05, 1e-4, 0.05, -0.999])
UPPER = np.array([1.0, 20.0, 1.0, 5.0, 0.5])


def heston_call(F, K, T, D, v0, kappa, theta, sigma_v, rho):
    """Call prices on forward F for an array of strikes, one maturity."""
    K = np.atleast_1d(np.asarray(K, float))
    lnK = np.log(K)
    out = []
    for j in (1, 2):
        cf = heston_char_func(_PHI, F, 1.0, T, 0.0, v0, kappa, theta, sigma_v, rho, j)
        integrand = np.real(np.exp(-1j * np.outer(lnK, _PHI)) * cf / (1j * _PHI))
        out.append(0.5 + integrand @ _W / np.pi)
    P1, P2 = out
    return D * (F * P1 - K * P2)


def heston_price(row_F, K, T, D, kind, p):
    c = heston_call(row_F, K, T, D, *p)
    return np.where(kind == "call", c, c - D * (row_F - K))


def _heston_residuals(x, groups):
    res = []
    for F, T, D, K, kind, mid, vega in groups:
        res.append((heston_price(F, K, T, D, kind, x) - mid) / vega)
    return np.concatenate(res)


def fit_heston(iv: pd.DataFrame, starts=None):
    """Least squares on vega-scaled price errors, which approximate vol errors."""
    groups = []
    for _, s in iv.groupby("expiry"):
        F, T, D = s["F"].iloc[0], s["T"].iloc[0], s["D"].iloc[0]
        groups.append((F, T, D, s["K"].to_numpy(), s["type"].to_numpy(), s["mid"].to_numpy(),
                       np.maximum(black76_vega(F, s["K"].to_numpy(), T, D, s["iv"].to_numpy()), 1e-3)))
    if starts is None:
        atm_var = float(np.median(iv.loc[iv["k"].abs() < 0.03, "iv"]) ** 2) if (iv["k"].abs() < 0.03).any() else 0.04
        starts = [[atm_var, 2.0, atm_var, 0.5, -0.7], [atm_var, 5.0, 1.5 * atm_var, 1.0, -0.8],
                  [atm_var, 1.0, atm_var, 0.3, -0.5], [atm_var, 10.0, atm_var, 2.0, -0.9]]
    best = None
    for x0 in starts:
        r = least_squares(_heston_residuals, np.clip(x0, LOWER, UPPER), bounds=(LOWER, UPPER),
                          args=(groups,), x_scale="jac", max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    p = dict(zip(PARAMS, best.x))
    p["converged"] = bool(best.success)
    p["feller_ratio"] = 2 * p["kappa"] * p["theta"] / p["sigma_v"] ** 2
    return p


def heston_ivs(iv: pd.DataFrame, p: dict) -> np.ndarray:
    x = [p[k] for k in PARAMS]
    out = np.empty(len(iv))
    for i, (F, K, T, D, kind) in enumerate(iv[["F", "K", "T", "D", "type"]].itertuples(index=False)):
        price = heston_price(F, np.array([K]), T, D, np.array([kind]), x)[0]
        out[i] = black76_iv(price, F, K, T, D, kind)
    return out


# --------------------------------------------------------------------------
# SVI
# --------------------------------------------------------------------------

def svi_w(k, a, b, rho, m, sigma):
    """Raw SVI total implied variance w(k) = sigma_imp^2 T."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi(k, w):
    """Fit raw SVI to total variance with b >= 0, |rho| < 1, sigma > 0 and
    Roger Lee's moment bound b (1 + |rho|) <= 2. Without the bound the wings
    can grow faster than any density allows, and the implied distribution of
    S_T has no mean. b is parameterized as 2 u / (1 + |rho|), u in [0, 1].
    """
    k, w = np.asarray(k), np.asarray(w)

    def unpack(x):
        a, u, rho, m, sigma = x
        return a, 2 * u / (1 + abs(rho)), rho, m, sigma

    best = None
    for m0 in (0.0, -0.05, 0.05):
        for s0 in (0.05, 0.2):
            x0 = [max(w.min() * 0.5, 1e-6), 0.05, -0.5, m0, s0]
            r = least_squares(lambda x: svi_w(k, *unpack(x)) - w, x0,
                              bounds=([-1.0, 0.0, -0.999, -1.0, 1e-4], [1.0, 1.0, 0.999, 1.0, 2.0]))
            if best is None or r.cost < best.cost:
                best = r
    return np.array(unpack(best.x))


def svi_density_g(k, a, b, rho, m, sigma):
    """Gatheral's g(k); the slice is free of butterfly arbitrage where g >= 0."""
    x = k - m
    r = np.sqrt(x ** 2 + sigma ** 2)
    w = a + b * (rho * x + r)
    w1 = b * (rho + x / r)
    w2 = b * sigma ** 2 / r ** 3
    return (1 - k * w1 / (2 * w)) ** 2 - w1 ** 2 / 4 * (1 / w + 0.25) + w2 / 2


def fit_svi_surface(iv: pd.DataFrame):
    """SVI per expiry, fitted IVs, and static arbitrage checks on a k grid."""
    params, fitted = {}, pd.Series(np.nan, index=iv.index)
    for exp, s in iv.groupby("expiry"):
        T = s["T"].iloc[0]
        x = fit_svi(s["k"], s["iv"] ** 2 * T)
        params[exp] = dict(T=T, a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4],
                           k_min=s["k"].min(), k_max=s["k"].max())
        fitted.loc[s.index] = np.sqrt(np.maximum(svi_w(s["k"], *x), 0) / T)

    # Butterfly: g(k) on each slice's own quoted range.
    butterfly = {}
    for exp, p in params.items():
        kg = np.linspace(p["k_min"], p["k_max"], 400)
        butterfly[exp] = float(svi_density_g(kg, p["a"], p["b"], p["rho"], p["m"], p["sigma"]).min())

    # Calendar: on the k range all slices share, w must rise with T.
    exps = sorted(params, key=lambda e: params[e]["T"])
    lo = max(params[e]["k_min"] for e in exps)
    hi = min(params[e]["k_max"] for e in exps)
    calendar = []
    if hi > lo:
        kg = np.linspace(lo, hi, 200)
        W = np.array([svi_w(kg, *[params[e][c] for c in ("a", "b", "rho", "m", "sigma")]) for e in exps])
        for i in range(1, len(exps)):
            calendar.append(dict(expiry=exps[i], min_increase=float((W[i] - W[i - 1]).min())))
    return params, fitted, butterfly, calendar, (lo, hi)


# --------------------------------------------------------------------------
# Risk-neutral density from an SVI slice
# --------------------------------------------------------------------------

def svi_density(k, a, b, rho, m, sigma):
    """Risk-neutral density of k = log(S_T / F) implied by an SVI slice.

    p(k) = g(k) / sqrt(2 pi w(k)) * exp(-d2(k)^2 / 2), with
    d2 = -k / sqrt(w) - sqrt(w) / 2 (Gatheral and Jacquier 2014, eq. 2.2).
    It is the second strike derivative of undiscounted call prices, written
    in log-moneyness.
    """
    w = svi_w(k, a, b, rho, m, sigma)
    sw = np.sqrt(w)
    d2 = -k / sw - sw / 2
    return svi_density_g(k, a, b, rho, m, sigma) / np.sqrt(2 * np.pi * w) * np.exp(-d2 ** 2 / 2)


def density_summary(p, k_lo=-8.0, k_hi=4.0, n=60001):
    """Mass, mean of S_T / F and left tail probabilities for one SVI slice."""
    from scipy.stats import norm
    k = np.linspace(k_lo, k_hi, n)
    x = (p["a"], p["b"], p["rho"], p["m"], p["sigma"])
    q = svi_density(k, *x)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(k))])
    mass = cdf[-1]

    def prob_below(level):
        return float(np.interp(np.log(level), k, cdf))

    in_range = float(np.interp(p["k_max"], k, cdf) - np.interp(p["k_min"], k, cdf))
    atm_w = float(svi_w(0.0, *x))
    out = dict(mass=float(mass), mean_over_forward=float(np.trapezoid(np.exp(k) * q, k)),
               mass_in_quoted_range=in_range,
               wing_slope_left=float(p["b"] * (1 - p["rho"])), wing_slope_right=float(p["b"] * (1 + p["rho"])))
    for lvl in (0.9, 0.8):
        out[f"p_below_{int(100 * lvl)}"] = prob_below(lvl)
        # Same probability under a lognormal with the ATM total variance.
        out[f"p_below_{int(100 * lvl)}_atm_lognormal"] = float(norm.cdf((np.log(lvl) + atm_w / 2) / np.sqrt(atm_w)))
    return out
