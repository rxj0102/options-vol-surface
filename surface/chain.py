"""From a raw option chain snapshot to an implied volatility surface.

Steps, per expiry
1. Mid quotes: (bid + ask) / 2, kept only when both sides are quoted, the
   spread is under MAX_REL_SPREAD of the mid and the mid is at least MIN_MID.
2. Discount factor D from the Treasury curve recorded with the snapshot, and
   forward F from put-call parity, F = K + (C - P) / D, at the strikes where
   call and put prices are closest. No dividend yield is assumed.
3. Black-76 implied volatility of out-of-the-money quotes: puts below F,
   calls at or above F, kept between the 2 delta put and the 2 delta call.

SPY options are American. Out-of-the-money puts and calls carry little
early exercise premium, which is why only they are used, but parity holds
only approximately, so F is an estimate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
MAX_REL_SPREAD = 0.5
MIN_MID = 0.05          # dollars; cheaper quotes are mostly tick size
DELTA_RANGE = (0.02, 0.98)  # Black call delta of quotes kept (2 delta wings)
N_PARITY = 5           # strikes with the smallest |C - P| used for the forward
EXPIRY_HOUR_UTC = 21   # 16:00 New York close, standard time


def load_snapshot(meta_path=ROOT / "data" / "spy_chain.json"):
    meta = json.loads(Path(meta_path).read_text())
    chain = pd.read_csv(Path(meta_path).parent / meta["file"])
    pulled = pd.Timestamp(meta["pulled_utc"], tz="UTC")
    exp = pd.to_datetime(chain["expiry"]).dt.tz_localize("UTC") + pd.Timedelta(hours=EXPIRY_HOUR_UTC)
    chain["T"] = (exp - pulled).dt.total_seconds() / (365.0 * 86400)
    return chain, meta


def black76(F, K, T, D, sigma, kind="call"):
    sigma = np.maximum(sigma, 1e-12)
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s ** 2) / s
    d2 = d1 - s
    if kind == "call":
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_vega(F, K, T, D, sigma):
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s ** 2) / s
    return D * F * norm.pdf(d1) * np.sqrt(T)


def black76_delta(F, K, T, sigma, kind="call"):
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s ** 2) / s
    return norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1


def black76_iv(price, F, K, T, D, kind="call"):
    intrinsic = D * max(F - K, 0.0) if kind == "call" else D * max(K - F, 0.0)
    upper = D * F if kind == "call" else D * K
    if not (intrinsic < price < upper):
        return np.nan
    f = lambda v: black76(F, K, T, D, v, kind) - price
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-10)
    except ValueError:
        return np.nan


def mids(chain: pd.DataFrame) -> pd.DataFrame:
    c = chain.copy()
    ok = (c["bid"] > 0) & (c["ask"] > c["bid"])
    c["mid"] = np.where(ok, 0.5 * (c["bid"] + c["ask"]), np.nan)
    c["rel_spread"] = (c["ask"] - c["bid"]) / c["mid"]
    return c[c["mid"].notna() & (c["rel_spread"] < MAX_REL_SPREAD) & (c["mid"] >= MIN_MID)]


def discount_factor(T: float, rates: dict) -> float:
    """exp(-y T), y linearly interpolated in maturity on the Treasury curve
    {years: yield} and held flat beyond its ends."""
    mats = sorted(float(k) for k in rates)
    ys = [rates[k] for k in sorted(rates, key=float)]
    return float(np.exp(-np.interp(T, mats, ys) * T))


def parity_forward(slice_: pd.DataFrame, D: float, n: int = N_PARITY):
    """Forward from put-call parity, F = K + (C - P) / D, at the strikes where
    the call and put are closest in price (the Cboe VIX method), median over
    the n closest. Near the money the early exercise premium of American puts
    is small; a regression of C - P on K across strikes is not, and gives a
    negative implied rate on SPY because the premium grows with the strike.
    """
    p = slice_.pivot_table(index="strike", columns="type", values="mid").dropna()
    if len(p) < 3:
        return np.nan, 0
    diff = p["call"] - p["put"]
    near = diff.abs().nsmallest(min(n, len(p))).index
    F = np.median(near.to_numpy(float) + diff.loc[near].to_numpy() / D)
    return float(F), len(near)


def surface(chain: pd.DataFrame, spot: float, rates: dict, min_T=14 / 365) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-the-money implied vols for every usable expiry, plus forwards.

    rates: Treasury curve {years: yield} used for discounting.
    """
    m = mids(chain)
    rows, fwd = [], []
    for exp, s in m.groupby("expiry"):
        T = float(s["T"].iloc[0])
        if T < min_T:
            continue
        D = discount_factor(T, rates)
        F, n = parity_forward(s, D)
        if not np.isfinite(F):
            continue
        fwd.append(dict(expiry=exp, T=T, F=F, D=D, n_parity=n, carry=np.log(F / spot) / T))
        otm = s[((s["type"] == "put") & (s["strike"] < F)) | ((s["type"] == "call") & (s["strike"] >= F))]
        for _, q in otm.iterrows():
            iv = black76_iv(q["mid"], F, q["strike"], T, D, q["type"])
            if np.isfinite(iv) and DELTA_RANGE[0] <= black76_delta(F, q["strike"], T, iv, "call") <= DELTA_RANGE[1]:
                rows.append(dict(expiry=exp, T=T, F=F, D=D, K=q["strike"], type=q["type"],
                                 mid=q["mid"], rel_spread=q["rel_spread"], iv=iv,
                                 k=np.log(q["strike"] / F),
                                 call_delta=black76_delta(F, q["strike"], T, iv, "call")))
    return pd.DataFrame(rows), pd.DataFrame(fwd)


def smile_stats(iv: pd.DataFrame) -> pd.DataFrame:
    """ATM vol, 25 delta risk reversal and butterfly per expiry.

    Vols are interpolated in call delta. Risk reversal = vol(25d call) -
    vol(25d put); butterfly = average of the two minus ATM. The 25 delta put
    is the 75 call delta point.
    """
    out = []
    for exp, s in iv.groupby("expiry"):
        s = s.sort_values("call_delta")
        d, v = s["call_delta"].to_numpy(), s["iv"].to_numpy()
        if d.min() > 0.25 or d.max() < 0.75:
            continue
        atm = float(np.interp(0.0, s["k"].sort_values().to_numpy(),
                              s.sort_values("k")["iv"].to_numpy()))
        c25, p25 = float(np.interp(0.25, d, v)), float(np.interp(0.75, d, v))
        out.append(dict(expiry=exp, T=float(s["T"].iloc[0]), atm=atm, rr25=c25 - p25,
                        bf25=0.5 * (c25 + p25) - atm, n=len(s)))
    return pd.DataFrame(out)
