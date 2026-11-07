"""GARCH vol forecast integration: compare realized vol to ATM implied vol."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pricing.bsm import bsm_price, implied_vol


def load_garch_forecasts():
    """Try loading from volatility-forecasting repo; fall back to mock."""
    try:
        # Look for CSV from sibling repo or environment
        candidates = [
            "../volatility-forecasting/outputs/garch_forecasts.csv",
            "volatility-forecasting/outputs/garch_forecasts.csv",
        ]
        for path in candidates:
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=["date"])
                print(f"[integration] Loaded GARCH forecasts from {path}")
                return df
    except Exception as e:
        print(f"[integration] Could not load GARCH forecasts: {e}")

    return _mock_garch_forecasts()


def _mock_garch_forecasts(n_days=252):
    """Simulate realistic GARCH(1,1) forecasts for ~1 year."""
    np.random.seed(0)
    dates = [datetime(2024, 1, 2) + timedelta(days=i) for i in range(n_days)]
    # GARCH(1,1) in *daily* variance units, annualised on output.  The
    # long-run level is set to 18% annual vol via variance targeting
    # (omega = (1 - alpha - beta) * long_run_var); mixing annual-vol levels
    # with a daily-scale omega would make the process decay to ~1% vol.
    alpha, beta = 0.08, 0.90
    long_run_daily = 0.18**2 / 252.0
    omega = (1.0 - alpha - beta) * long_run_daily
    sigma2 = np.zeros(n_days)
    sigma2[0] = long_run_daily
    for t in range(1, n_days):
        eps = np.random.normal(0, np.sqrt(sigma2[t-1]))
        sigma2[t] = omega + alpha * eps**2 + beta * sigma2[t-1]
    garch_vol = np.sqrt(sigma2 * 252.0)
    print("[integration] Using mock GARCH vol forecasts.")
    return pd.DataFrame({"date": dates, "garch_vol": garch_vol})


def _simulate_atm_implied_vol(garch_df, risk_premium_mean=0.02):
    """Simulate ATM implied vol as GARCH + risk premium + noise."""
    np.random.seed(1)
    n = len(garch_df)
    risk_premium = risk_premium_mean + 0.01 * np.random.randn(n)
    # Occasional vol surprises: realized > implied
    surprise_mask = np.random.rand(n) < 0.15
    risk_premium[surprise_mask] -= 0.04
    atm_iv = garch_df["garch_vol"].values + risk_premium
    atm_iv = np.clip(atm_iv, 0.05, 1.0)
    return atm_iv


def compute_garch_option_prices(garch_df, S=530.0, K=530.0, T=30/365, r=0.05):
    """BSM prices using GARCH vol vs market (ATM implied vol) prices."""
    atm_iv = _simulate_atm_implied_vol(garch_df)
    garch_prices = [bsm_price(S, K, T, r, v, "call") for v in garch_df["garch_vol"]]
    market_prices = [bsm_price(S, K, T, r, v, "call") for v in atm_iv]
    garch_df = garch_df.copy()
    garch_df["atm_iv"] = atm_iv
    garch_df["garch_price"] = garch_prices
    garch_df["market_price"] = market_prices
    garch_df["vol_spread"] = atm_iv - garch_df["garch_vol"]  # +: risk premium, -: vol surprise
    return garch_df


def plot_garch_vs_implied(df, output_path="outputs/garch_vs_implied_vol.png"):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    ax = axes[0]
    ax.plot(df["date"], df["garch_vol"] * 100, label="GARCH Forecast Vol", color="navy", linewidth=1.2)
    ax.plot(df["date"], df["atm_iv"] * 100, label="ATM Implied Vol", color="crimson",
            linewidth=1.2, alpha=0.85)

    # shade risk premium vs vol surprise regions
    premium_mask = df["vol_spread"] > 0
    surprise_mask = df["vol_spread"] < 0
    ax.fill_between(df["date"], df["garch_vol"] * 100, df["atm_iv"] * 100,
                    where=premium_mask.values, alpha=0.2, color="gold",
                    label="Risk Premium (Implied > GARCH)")
    ax.fill_between(df["date"], df["garch_vol"] * 100, df["atm_iv"] * 100,
                    where=surprise_mask.values, alpha=0.3, color="blue",
                    label="Vol Surprise (GARCH > Implied)")

    ax.set_ylabel("Volatility (%)")
    ax.set_title("GARCH Forecast Vol vs. ATM Implied Vol")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(df["date"], df["vol_spread"] * 100,
            color=np.where(df["vol_spread"] > 0, "gold", "steelblue"),
            width=1.0, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Vol Spread (ATM IV − GARCH, %)")
    ax2.set_xlabel("Date")
    ax2.set_title("Implied-vs-Realized Vol Spread")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[integration] Saved {output_path}")


def run():
    garch_df = load_garch_forecasts()
    df = compute_garch_option_prices(garch_df)

    risk_prem_pct = (df["vol_spread"] > 0).mean() * 100
    mean_spread = df["vol_spread"].mean() * 100
    print(f"\n[integration] Risk premium periods: {risk_prem_pct:.1f}% of days")
    print(f"[integration] Mean vol spread (IV - GARCH): {mean_spread:.2f}%")

    plot_garch_vs_implied(df)
    return df


if __name__ == "__main__":
    run()
