"""IV surface plots: smile, ATM term structure, skew, BSM vs Heston comparison."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pricing.bsm import implied_vol, bsm_price, delta as bsm_delta
from pricing.heston import heston_iv_surface
from calibration.calibrate_heston import calibrate


def plot_iv_surface(iv_df, output_path="outputs/iv_surface.png"):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    x = iv_df["moneyness"].values
    y = (iv_df["T"] * 365).values
    z = iv_df["iv"].values * 100

    sc = ax.scatter(x, y, z, c=z, cmap="RdYlGn_r", s=20, alpha=0.8)
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Days to Expiry")
    ax.set_zlabel("Implied Vol (%)")
    ax.set_title("SPY Implied Volatility Surface")
    fig.colorbar(sc, ax=ax, shrink=0.5, label="IV (%)")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[surface] Saved {output_path}")


def plot_vol_smile(iv_df, output_path="outputs/vol_smile.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    expiries = sorted(iv_df["expiry"].unique())[:3]
    for ax, exp in zip(axes, expiries):
        sub = iv_df[iv_df["expiry"] == exp].sort_values("moneyness")
        ax.plot(sub["moneyness"], sub["iv"] * 100, "o-", markersize=4)
        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel("Moneyness (K/S)")
        ax.set_ylabel("Implied Vol (%)")
        days = int(sub["T"].mean() * 365)
        ax.set_title(f"Vol Smile — {exp} ({days}d)")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Volatility Smile by Expiry", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[surface] Saved {output_path}")


def plot_atm_term_structure(iv_df, output_path="outputs/atm_term_structure.png"):
    atm = iv_df[(iv_df["moneyness"] > 0.97) & (iv_df["moneyness"] < 1.03)]
    atm_mean = atm.groupby("T")["iv"].mean().reset_index().sort_values("T")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(atm_mean["T"] * 365, atm_mean["iv"] * 100, "o-", color="steelblue")
    ax.set_xlabel("Days to Expiry")
    ax.set_ylabel("ATM Implied Vol (%)")
    ax.set_title("ATM Volatility Term Structure")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[surface] Saved {output_path}")


def compute_put_skew(iv_df, r=0.05):
    """25-delta put IV minus 25-delta call IV per expiry."""
    rows = []
    for exp in iv_df["expiry"].unique():
        sub = iv_df[iv_df["expiry"] == exp].copy()
        S = sub["S"].iloc[0]
        T = sub["T"].iloc[0]
        # find strikes near 25-delta
        sub["call_delta"] = sub.apply(
            lambda row: bsm_delta(row["S"], row["K"], row["T"], r, row["iv"], "call"), axis=1)
        calls_25 = sub[(sub["call_delta"] > 0.20) & (sub["call_delta"] < 0.30)]
        puts_25 = sub[(sub["call_delta"] > 0.70) & (sub["call_delta"] < 0.80)]
        if not calls_25.empty and not puts_25.empty:
            skew = puts_25["iv"].mean() - calls_25["iv"].mean()
            rows.append(dict(expiry=exp, T=T, skew=skew * 100))
    return pd.DataFrame(rows)


def plot_heston_vs_bsm(params, iv_df, r=0.05, output_path="outputs/heston_vs_bsm_surface.png"):
    """Side-by-side 3D: BSM flat surface vs Heston smile surface."""
    S = iv_df["S"].iloc[0]
    moneyness_grid = np.linspace(0.85, 1.15, 20)
    expiries_y = sorted(iv_df["T"].unique())[:3]
    strikes_grid = S * moneyness_grid

    # Heston surface
    heston_surf = heston_iv_surface(S, strikes_grid, expiries_y, r, params)
    # BSM flat surface — use ATM iv as flat vol
    atm_iv = iv_df[(iv_df["moneyness"] > 0.97) & (iv_df["moneyness"] < 1.03)]["iv"].mean()
    bsm_surf = np.full_like(heston_surf, atm_iv)

    days_grid = np.array(expiries_y) * 365
    MG, DG = np.meshgrid(moneyness_grid, days_grid)

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(MG, DG, bsm_surf * 100, cmap="Blues", alpha=0.85)
    ax1.set_title("BSM Flat Vol Surface")
    ax1.set_xlabel("Moneyness")
    ax1.set_ylabel("Days to Expiry")
    ax1.set_zlabel("IV (%)")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(MG, DG, heston_surf * 100, cmap="RdYlGn_r", alpha=0.85)
    ax2.set_title("Heston Implied Vol Surface")
    ax2.set_xlabel("Moneyness")
    ax2.set_ylabel("Days to Expiry")
    ax2.set_zlabel("IV (%)")

    plt.suptitle("BSM vs. Heston: Implied Volatility Surface", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[surface] Saved {output_path}")


if __name__ == "__main__":
    params, rmse, iv_df, raw_data = calibrate()
    plot_iv_surface(iv_df)
    plot_vol_smile(iv_df)
    plot_atm_term_structure(iv_df)
    plot_heston_vs_bsm(params, iv_df)
    skew_df = compute_put_skew(iv_df)
    print("\nPut Skew (25Δ put IV − 25Δ call IV):")
    print(skew_df.to_string(index=False))
