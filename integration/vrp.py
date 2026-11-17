"""Does VIX overstate the volatility that follows, and does it still forecast it?

Data: daily VIX and S&P 500 closes from 1990 (data/vix_spx.csv.gz).

Design
- Non-overlapping 21 trading day blocks. On the first day t of a block the
  forecasts are fixed using data up to the close of t; realized variance is
  the annualized sum of squared log returns from t+1 to t+21.
- Implied variance: (VIX_t / 100)^2. VIX is a 30 calendar day variance swap
  rate on SPX options, which is about 21 trading days.
- GARCH(1,1) with Gaussian quasi likelihood, refit at the start of every year
  on all returns before it (expanding window, first fit uses 1990 to 1994), so
  no forecast uses data after its date. The 21 day forecast is the analytic
  average of the conditional variance path.
- Trailing realized variance over the previous 21 days as the naive forecast.

Outputs
- The variance risk premium: implied minus realized, in variance and in
  volatility points, and the monthly profit of selling variance at VIX^2.
- Forecast accuracy: MSE and QLIKE of each forecast, Diebold-Mariano tests,
  Mincer-Zarnowitz regressions and an encompassing regression.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
H = 21          # forecast horizon in trading days
ANN = 252
BURN_IN_END = 1994  # first GARCH fit uses returns up to the end of this year


def load(path=ROOT / "data" / "vix_spx.csv.gz") -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df["r"] = np.log(df["spx"]).diff()
    return df.dropna()


# --------------------------------------------------------------------------
# GARCH(1,1)
# --------------------------------------------------------------------------

def garch_filter(r: np.ndarray, omega: float, alpha: float, beta: float, s0: float | None = None) -> np.ndarray:
    """Conditional variances: s2[t] = Var(r[t] | r[:t]) for t = 0..n.

    s2[0] = s0 (the sample variance of r when not given); s2[n] is the one
    step ahead forecast after the last return. The recursion
    s2[t+1] = omega + alpha r[t]^2 + beta s2[t] is a first order linear
    filter, so it runs in C through scipy.signal.lfilter.
    """
    s0 = float(r.var()) if s0 is None else float(s0)
    x = omega + alpha * np.asarray(r, float) ** 2
    y, _ = lfilter([1.0], [1.0, -beta], x, zi=[beta * s0])
    return np.concatenate([[s0], y])


def garch_fit(r: np.ndarray) -> tuple[float, float, float]:
    r = r - r.mean()
    var = r.var()

    def nll(p):
        a, b = p
        if a <= 0 or b <= 0 or a + b >= 0.9999:
            return 1e10
        omega = var * (1 - a - b)  # variance targeting
        s2 = garch_filter(r, omega, a, b)[:-1]
        return 0.5 * np.sum(np.log(s2) + r ** 2 / s2)

    best = None
    for a0, b0 in ((0.08, 0.90), (0.05, 0.93), (0.12, 0.85)):
        res = minimize(nll, [a0, b0], method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    a, b = best.x
    return var * (1 - a - b), a, b


def garch_horizon_variance(s2_next: float, omega: float, alpha: float, beta: float, h: int = H) -> float:
    """Average expected daily variance over the next h days, given s2 for day 1."""
    p = alpha + beta
    vl = omega / (1 - p)
    k = np.arange(h)
    return float(np.mean(vl + p ** k * (s2_next - vl)))


def garch_forecasts(df: pd.DataFrame) -> pd.Series:
    """Annualized 21 day GARCH variance forecast made at each close, refit yearly."""
    r = df["r"].to_numpy()
    years = df.index.year.to_numpy()
    out = np.full(len(df), np.nan)
    params = {}
    for y in range(BURN_IN_END + 1, years.max() + 1):
        train = r[years < y]
        omega, a, b = garch_fit(train)
        params[y] = (omega, a, b)
        mu = train.mean()
        idx = np.where(years == y)[0]
        if len(idx) == 0:
            continue
        # Filter through everything up to the end of year y with the frozen
        # parameters; s2[t+1] is the forecast for day t+1 made at close t.
        s2 = garch_filter(r[: idx[-1] + 1] - mu, omega, a, b, s0=train.var())
        for t in idx:
            out[t] = ANN * garch_horizon_variance(s2[t + 1], omega, a, b)
    s = pd.Series(out, index=df.index, name="garch")
    s.attrs["params"] = params
    return s


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------

def blocks(df: pd.DataFrame, garch: pd.Series) -> pd.DataFrame:
    """One row per non-overlapping block, forecasts at its first day t."""
    r = df["r"].to_numpy()
    r2 = r ** 2
    start = int(np.argmax(garch.notna().to_numpy()))
    rows = []
    for t in range(start, len(df) - H, H):
        rows.append(dict(
            date=df.index[t],
            vix=df["vix"].iloc[t],
            implied=(df["vix"].iloc[t] / 100) ** 2,
            garch=garch.iloc[t],
            trailing=ANN * r2[t - H + 1: t + 1].mean(),
            realized=ANN * r2[t + 1: t + 1 + H].mean(),
        ))
    b = pd.DataFrame(rows).set_index("date")
    b["realized_vol"] = np.sqrt(b["realized"])
    return b


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def nw_se(x: np.ndarray, lags: int = 3) -> float:
    """Newey-West standard error of the mean."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    s = x @ x / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return float(np.sqrt(s / n))


def ols_nw(y: np.ndarray, X: np.ndarray, lags: int = 3):
    """OLS coefficients with Newey-West standard errors."""
    X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        G = Xu[k:].T @ Xu[:-k]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - u.var() / y.var()
    return beta, se, r2


def qlike(rv, f):
    return rv / f - np.log(rv / f) - 1


def diebold_mariano(l1: np.ndarray, l2: np.ndarray, lags: int = 3):
    d = l1 - l2
    t = d.mean() / nw_se(d, lags)
    return float(t), float(2 * stats.norm.sf(abs(t)))


def study(df: pd.DataFrame | None = None) -> dict:
    df = load() if df is None else df
    g = garch_forecasts(df)
    b = blocks(df, g)
    n = len(b)

    vrp_var = b["implied"] - b["realized"]
    vrp_vol = 100 * (np.sqrt(b["implied"]) - b["realized_vol"])
    pnl = 100 ** 2 * vrp_var  # variance points per block, short one unit of variance at VIX^2
    worst = pnl.nsmallest(3)

    res = dict(
        n_blocks=n,
        first_block=str(b.index[0].date()),
        last_block=str(b.index[-1].date()),
        mean_vix=float(b["vix"].mean()),
        mean_realized_vol=float(100 * b["realized_vol"].mean()),
        vrp_vol_points_mean=float(vrp_vol.mean()),
        vrp_vol_points_t=float(vrp_vol.mean() / nw_se(vrp_vol)),
        vrp_vol_points_median=float(vrp_vol.median()),
        share_implied_above_realized=float((vrp_var > 0).mean()),
        short_var_mean=float(pnl.mean()),
        short_var_t=float(pnl.mean() / nw_se(pnl)),
        short_var_sharpe_annual=float(pnl.mean() / pnl.std() * np.sqrt(ANN / H)),
        short_var_skew=float(stats.skew(pnl)),
        short_var_share_of_total_lost_in_worst3=float(-worst.sum() / pnl.sum()),
        worst_blocks=[dict(date=str(d.date()), pnl=float(v), vix=float(b.loc[d, "vix"]),
                           realized_vol=float(100 * b.loc[d, "realized_vol"])) for d, v in worst.items()],
    )

    loss = {}
    for k in ("implied", "garch", "trailing"):
        loss[k] = dict(mse=float(np.mean((b["realized"] - b[k]) ** 2) * 1e4),
                       qlike=float(np.mean(qlike(b["realized"], b[k]))),
                       mean_forecast_vol=float(100 * np.sqrt(b[k]).mean()))
    res["loss"] = loss
    dm = {}
    for a, c in (("implied", "garch"), ("implied", "trailing"), ("garch", "trailing")):
        for lname, f in (("mse", lambda x: (b["realized"] - b[x]) ** 2), ("qlike", lambda x: qlike(b["realized"], b[x]))):
            t, p = diebold_mariano(f(a).to_numpy(), f(c).to_numpy())
            dm[f"{a}_vs_{c}_{lname}"] = dict(t=t, p=p)
    res["diebold_mariano"] = dm

    mz = {}
    y = b["realized"].to_numpy()
    for k in ("implied", "garch", "trailing"):
        beta, se, r2 = ols_nw(y, b[k].to_numpy())
        mz[k] = dict(intercept=float(beta[0]), slope=float(beta[1]), slope_se=float(se[1]),
                     t_slope_eq_1=float((beta[1] - 1) / se[1]), r2=float(r2))
    beta, se, r2 = ols_nw(y, b[["implied", "garch"]].to_numpy())
    mz["encompassing"] = dict(b_implied=float(beta[1]), t_implied=float(beta[1] / se[1]),
                              b_garch=float(beta[2]), t_garch=float(beta[2] / se[2]), r2=float(r2))
    # Same regressions in logs, which are not dominated by the crash months.
    ly = np.log(y)
    for k in ("implied", "garch"):
        beta, se, r2 = ols_nw(ly, np.log(b[k].to_numpy()))
        mz[f"log_{k}"] = dict(slope=float(beta[1]), slope_se=float(se[1]), r2=float(r2))
    beta, se, r2 = ols_nw(ly, np.log(b[["implied", "garch"]].to_numpy()))
    mz["log_encompassing"] = dict(b_implied=float(beta[1]), t_implied=float(beta[1] / se[1]),
                                  b_garch=float(beta[2]), t_garch=float(beta[2] / se[2]), r2=float(r2))
    res["mincer_zarnowitz"] = mz
    p = g.attrs["params"][max(g.attrs["params"])]
    res["garch_last_fit"] = dict(alpha=float(p[1]), beta=float(p[2]),
                                 long_run_vol=float(100 * np.sqrt(ANN * p[0] / (1 - p[1] - p[2]))))
    return res, b
