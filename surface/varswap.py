"""Model-free 30 day variance from the SPY chain, computed the way Cboe computes VIX.

Cboe's VIX white paper, applied to SPY instead of SPX options:

    sigma^2 = (2 / T) * sum_i (dK_i / K_i^2) * e^{RT} * Q(K_i)  -  (1 / T) * (F / K0 - 1)^2

- F is the forward from put-call parity (surface.chain.parity_forward) and
  e^{RT} = 1 / D from the Treasury curve recorded with the snapshot.
- K0 is the first listed strike at or below F.
- Q(K) is the bid-ask mid of the out-of-the-money option at K: puts below K0,
  calls above K0, the average of put and call at K0. Moving away from K0,
  zero-bid quotes are skipped and the strip stops after two consecutive
  zero bids.
- dK_i is half the distance between the strikes on either side of K_i (the
  full distance to the one neighbour at the ends).

The two expiries closest to 30 days on either side are combined by linear
interpolation in total variance, and the result is annualised:

    VIX-style = 100 * sqrt( (T1 s1^2 w1 + T2 s2^2 w2) * 365 / 30 ),
    w1 = (T2 - T30) / (T2 - T1), w2 = 1 - w1.

Differences from VIX that the comparison has to allow for: SPY is a tenth of
the index with its own dividends, SPY options are American (out of the
money ones carry little early exercise premium), the strikes are a dollar
apart near the money, and the snapshot is one intraday moment while the
only free VIX values are the day's open, high, low and close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from surface.chain import black76

TARGET_YEARS = 30 / 365


def _strip(side: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    """Walk away from K0: drop zero bids, stop after two consecutive zero bids."""
    side = side.sort_values("strike", ascending=ascending)
    keep, zeros = [], 0
    for _, q in side.iterrows():
        if q["bid"] <= 0:
            zeros += 1
            if zeros == 2:
                break
            continue
        zeros = 0
        keep.append(q)
    return pd.DataFrame(keep)


def slice_variance(slice_: pd.DataFrame, F: float, D: float, T: float) -> dict:
    """Cboe model-free variance for one expiry from its raw quotes."""
    q = slice_[(slice_["ask"] > 0) & (slice_["ask"] >= slice_["bid"])].copy()
    q["mid"] = 0.5 * (q["bid"] + q["ask"])
    strikes = np.sort(q["strike"].unique())
    below = strikes[strikes <= F]
    if len(below) == 0:
        return dict(var=np.nan)
    K0 = float(below.max())

    puts = _strip(q[(q["type"] == "put") & (q["strike"] < K0)], ascending=False)
    calls = _strip(q[(q["type"] == "call") & (q["strike"] > K0)], ascending=True)
    at = q[q["strike"] == K0].set_index("type")["mid"]
    if not {"put", "call"} <= set(at.index):
        return dict(var=np.nan)

    Q = pd.concat([
        puts[["strike", "mid"]] if len(puts) else pd.DataFrame(columns=["strike", "mid"]),
        pd.DataFrame({"strike": [K0], "mid": [0.5 * (at["put"] + at["call"])]}),
        calls[["strike", "mid"]] if len(calls) else pd.DataFrame(columns=["strike", "mid"]),
    ]).sort_values("strike")
    K = Q["strike"].to_numpy(float)
    P = Q["mid"].to_numpy(float)
    dK = np.empty_like(K)
    dK[1:-1] = 0.5 * (K[2:] - K[:-2])
    dK[0], dK[-1] = K[1] - K[0], K[-1] - K[-2]

    var = (2.0 / T) * np.sum(dK / K ** 2 * P / D) - (F / K0 - 1.0) ** 2 / T
    return dict(var=float(var), K0=K0, n_strikes=int(len(K)),
                k_lo=float(np.log(K[0] / F)), k_hi=float(np.log(K[-1] / F)))


def svi_variance(params: dict, T: float, F: float = 1.0, D: float = 1.0,
                 k_lo: float = -3.0, k_hi: float = 1.5, n: int = 20001) -> float:
    """The same integral with prices from a fitted SVI slice on a fine strike
    grid between k_lo and k_hi. Called with the listed strip's own range, it is
    a model-based counterpart to slice_variance: the two differ only in
    whether prices are quotes or the fitted smile. Extending the range far
    into the wings is not meaningful, because the fitted wing slope is capped
    near Lee's bound and the integral then grows with the range."""
    from surface.fit import svi_w
    k = np.linspace(k_lo, k_hi, n)
    w = np.maximum(svi_w(k, params["a"], params["b"], params["rho"], params["m"], params["sigma"]), 1e-12)
    vol = np.sqrt(w / T)
    K = F * np.exp(k)
    price = np.where(K < F, black76(F, K, T, D, vol, "put"), black76(F, K, T, D, vol, "call"))
    integrand = price / D / K ** 2
    return float((2.0 / T) * np.trapezoid(integrand, K))


def thirty_day(chain: pd.DataFrame, fwd: pd.DataFrame, svi: dict | None = None) -> dict:
    """Interpolate the two expiries that bracket 30 days to a 30 day vol (in %)."""
    f = fwd.sort_values("T")
    below = f[f["T"] <= TARGET_YEARS]
    above = f[f["T"] > TARGET_YEARS]
    if below.empty or above.empty:
        return {}
    near, nxt = below.iloc[-1], above.iloc[0]
    out = {}
    legs = []
    for r in (near, nxt):
        s = chain[chain["expiry"] == r["expiry"]]
        v = slice_variance(s, r["F"], r["D"], r["T"])
        v.update(expiry=r["expiry"], T=float(r["T"]), F=float(r["F"]))
        if svi is not None and r["expiry"] in svi and np.isfinite(v["var"]):
            # Same log-moneyness range as the listed strip, so the two differ
            # only in prices, not in how far into the wings they reach.
            v["svi_var"] = svi_variance(svi[r["expiry"]], r["T"], r["F"], r["D"],
                                        k_lo=v["k_lo"], k_hi=v["k_hi"])
        legs.append(v)
    T1, T2 = legs[0]["T"], legs[1]["T"]
    w1 = (T2 - TARGET_YEARS) / (T2 - T1)

    def combine(key):
        tv = w1 * T1 * legs[0][key] + (1 - w1) * T2 * legs[1][key]
        return float(100 * np.sqrt(tv / TARGET_YEARS))

    out["vol"] = combine("var")
    if all("svi_var" in g for g in legs):
        out["svi_vol"] = combine("svi_var")
    out["legs"] = legs
    out["weight_near"] = float(w1)
    return out
