"""Delta-hedging simulation: 1000 GBM paths, 30-day horizon, P&L attribution."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pricing.bsm import bsm_price, delta as bsm_delta, gamma as bsm_gamma, theta as bsm_theta


def simulate_delta_hedge(S0=530.0, K=530.0, T=30/365, r=0.05, sigma=0.20,
                          n_paths=1000, n_steps=30, seed=42):
    """
    Simulate delta-hedged ATM call over n_steps days on n_paths GBM paths.
    Returns dict with arrays of P&L components.
    """
    np.random.seed(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    pnl_gamma = np.zeros(n_paths)
    pnl_theta = np.zeros(n_paths)
    pnl_total = np.zeros(n_paths)

    # Buy ATM call at t=0
    C0 = bsm_price(S0, K, T, r, sigma, "call")

    for path in range(n_paths):
        S = S0
        t = 0.0
        hedge_position = bsm_delta(S, K, T, r, sigma, "call")  # shares held short
        cash = C0 - hedge_position * S  # initial cash from option sale minus hedge cost
        cumulative_gamma_pnl = 0.0
        cumulative_theta_pnl = 0.0

        for step in range(n_steps):
            T_rem = T - t
            if T_rem <= 1e-8:
                break

            dZ = np.random.normal(0, 1)
            dS = S * (r * dt + sigma * sqrt_dt * dZ)
            S_new = S + dS

            # gamma P&L: 0.5 * Γ * dS²
            g = bsm_gamma(S, K, T_rem, r, sigma)
            gm_pnl = 0.5 * g * dS**2
            cumulative_gamma_pnl += gm_pnl

            # theta P&L: Θ * dt (negative for long option)
            th = bsm_theta(S, K, T_rem, r, sigma, "call")
            th_pnl = th * dt * 365  # theta is per calendar day
            cumulative_theta_pnl += th_pnl

            # rebalance hedge
            t += dt
            T_rem_new = T - t
            if T_rem_new > 1e-8:
                new_delta = bsm_delta(S_new, K, T_rem_new, r, sigma, "call")
            else:
                new_delta = 1.0 if S_new > K else 0.0

            # cash flow from rebalancing
            cash -= (new_delta - hedge_position) * S_new
            cash *= np.exp(r * dt)  # accrue interest

            hedge_position = new_delta
            S = S_new

        # Final option payoff
        payoff = max(S - K, 0.0)
        # Final P&L = option payoff + cash + hedge liquidation
        final_pnl = payoff + cash + hedge_position * S - C0 * np.exp(r * T)

        pnl_gamma[path] = cumulative_gamma_pnl
        pnl_theta[path] = cumulative_theta_pnl
        pnl_total[path] = final_pnl

    pnl_residual = pnl_total - pnl_gamma - pnl_theta

    return {
        "total": pnl_total,
        "gamma": pnl_gamma,
        "theta": pnl_theta,
        "residual": pnl_residual,
        "C0": C0,
    }


def plot_pnl_distribution(results, output_path="outputs/delta_hedge_pnl_distribution.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    components = [
        ("total", "Total P&L", "steelblue"),
        ("gamma", "Gamma P&L (0.5·Γ·dS²)", "green"),
        ("theta", "Theta P&L (Θ·dt)", "red"),
        ("residual", "Residual P&L", "orange"),
    ]

    for ax, (key, label, color) in zip(axes.flatten(), components):
        data = results[key]
        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(data), color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean: {np.mean(data):.2f}")
        ax.axvline(np.percentile(data, 5), color="darkred", linestyle=":", linewidth=1.2,
                   label=f"5th pct: {np.percentile(data, 5):.2f}")
        ax.set_title(label)
        ax.set_xlabel("P&L ($)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    C0 = results["C0"]
    fig.suptitle(
        f"Delta-Hedging P&L Attribution (1000 paths, 30 days)\n"
        f"ATM Call premium: ${C0:.2f}",
        fontsize=12
    )
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[hedging] Saved {output_path}")


def print_summary(results):
    print("\n" + "="*50)
    print("Delta-Hedge P&L Attribution Summary")
    print("="*50)
    for key in ["total", "gamma", "theta", "residual"]:
        d = results[key]
        print(f"  {key:10s}: mean={np.mean(d):7.3f}  std={np.std(d):6.3f}  "
              f"5pct={np.percentile(d,5):7.3f}  95pct={np.percentile(d,95):7.3f}")


if __name__ == "__main__":
    results = simulate_delta_hedge()
    print_summary(results)
    plot_pnl_distribution(results)
