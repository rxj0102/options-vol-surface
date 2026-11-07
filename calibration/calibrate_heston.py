"""Calibrate Heston model to SPY market options data."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pricing.bsm import implied_vol, bsm_price
from pricing.heston import heston_price
from calibration.market_data import fetch_spy_options


def compute_market_iv(options_data, r=0.05):
    """Compute market IV for calls/puts, return clean DataFrame."""
    rows = []
    for exp, calls, puts, S, T in options_data:
        for _, row in calls.iterrows():
            K = row["strike"]
            mid = (row.get("bid", 0) + row.get("ask", 0)) / 2.0
            if mid <= 0:
                mid = row["lastPrice"]
            iv = implied_vol(mid, S, K, T, r, "call")
            if not np.isnan(iv) and 0.02 < iv < 2.0:
                rows.append(dict(expiry=exp, T=T, K=K, S=S, iv=iv,
                                  moneyness=K/S, option_type="call", mid=mid))
    return pd.DataFrame(rows)


def heston_rmse(params, iv_data, r=0.05):
    v0, kappa, theta, sigma_v, rho = params
    if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0 or abs(rho) >= 1:
        return 1e6
    # Feller condition soft penalty
    feller = 2 * kappa * theta - sigma_v**2
    penalty = max(0, -feller) * 10

    errors = []
    for _, row in iv_data.iterrows():
        S, K, T = row["S"], row["K"], row["T"]
        try:
            hp = heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho)
            h_iv = implied_vol(hp, S, K, T, r, "call")
            if not np.isnan(h_iv):
                errors.append((h_iv - row["iv"])**2)
        except Exception:
            pass

    if not errors:
        return 1e6
    return np.sqrt(np.mean(errors)) + penalty


def calibrate(r=0.05, n_expiries=3):
    data = fetch_spy_options(n_expiries)
    iv_df = compute_market_iv(data, r)
    print(f"[calibrate] {len(iv_df)} options with valid market IV.")

    # Subsample for speed: at most 60 options
    if len(iv_df) > 60:
        iv_df = iv_df.sample(60, random_state=42)

    x0 = [0.04, 2.0, 0.04, 0.3, -0.7]
    bounds = [(1e-4, 0.5), (0.1, 10.0), (1e-4, 0.5), (0.05, 1.0), (-0.99, 0.0)]

    print("[calibrate] Running L-BFGS-B optimization...")
    res = minimize(heston_rmse, x0, args=(iv_df, r),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-8})

    v0, kappa, theta, sigma_v, rho = res.x
    rmse = res.fun
    params = dict(v0=v0, kappa=kappa, theta=theta, sigma_v=sigma_v, rho=rho)

    print(f"\n{'='*50}")
    print("Heston Calibration Results")
    print(f"{'='*50}")
    for k, v in params.items():
        print(f"  {k:8s}: {v:.6f}")
    print(f"  RMSE    : {rmse:.6f}")
    print(f"  Converged: {res.success}")

    results_df = pd.DataFrame([{**params, "rmse": rmse, "converged": res.success}])
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/calibration_results.csv", index=False)
    print("[calibrate] Saved outputs/calibration_results.csv")

    return params, rmse, iv_df, data


if __name__ == "__main__":
    calibrate()
