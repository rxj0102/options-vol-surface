"""Black-Scholes-Merton pricing, Greeks, and implied volatility solver."""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bsm_price(S, K, T, r, sigma, option_type="call"):
    """European option price via BSM."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return 1.0 if (option_type == "call" and S > K) else 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def gamma(S, K, T, r, sigma):
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """Vega per 1% move in vol."""
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100.0


def theta(S, K, T, r, sigma, option_type="call"):
    """Theta per calendar day."""
    if T <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        return (term1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    return (term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0


def rho(S, K, T, r, sigma, option_type="call"):
    """Rho per 1% move in rates."""
    if T <= 0:
        return 0.0
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


def implied_vol(market_price, S, K, T, r, option_type="call", tol=1e-8):
    """Implied volatility via Brent's method."""
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if market_price <= intrinsic + tol:
        return np.nan

    def objective(sigma):
        return bsm_price(S, K, T, r, sigma, option_type) - market_price

    try:
        return brentq(objective, 1e-6, 10.0, xtol=tol)
    except ValueError:
        return np.nan


def greeks(S, K, T, r, sigma, option_type="call"):
    return {
        "price": bsm_price(S, K, T, r, sigma, option_type),
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "vega": vega(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type),
        "rho": rho(S, K, T, r, sigma, option_type),
    }


if __name__ == "__main__":
    S, K, T, r, sigma = 500.0, 500.0, 0.25, 0.05, 0.20
    g = greeks(S, K, T, r, sigma, "call")
    print("BSM ATM Call:")
    for k, v in g.items():
        print(f"  {k:8s}: {v:.6f}")
    iv = implied_vol(g["price"], S, K, T, r, "call")
    print(f"  IV check: {iv:.6f}  (should ≈ {sigma})")
