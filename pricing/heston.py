"""Heston stochastic volatility model — characteristic function approach."""

import numpy as np
from scipy.integrate import quad


def heston_char_func(phi, S, K, T, r, v0, kappa, theta, sigma_v, rho, j):
    """Heston characteristic function for integration (Heston 1993)."""
    x = np.log(S)
    u = 0.5 if j == 1 else -0.5
    b = kappa - rho * sigma_v if j == 1 else kappa

    d = np.sqrt((rho * sigma_v * 1j * phi - b) ** 2 - sigma_v**2 * (2 * u * 1j * phi - phi**2))
    g = (b - rho * sigma_v * 1j * phi + d) / (b - rho * sigma_v * 1j * phi - d)

    exp_dT = np.exp(d * T)
    C = r * 1j * phi * T + (kappa * theta / sigma_v**2) * (
        (b - rho * sigma_v * 1j * phi + d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((b - rho * sigma_v * 1j * phi + d) / sigma_v**2) * (
        (1 - exp_dT) / (1 - g * exp_dT)
    )
    return np.exp(C + D * v0 + 1j * phi * x)


def _integrand(phi, S, K, T, r, v0, kappa, theta, sigma_v, rho, j):
    cf = heston_char_func(phi, S, K, T, r, v0, kappa, theta, sigma_v, rho, j)
    return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))


def heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, option_type="call"):
    """European option price via Heston semi-analytical formula."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    limit = 200
    P1, _ = quad(_integrand, 1e-9, limit, args=(S, K, T, r, v0, kappa, theta, sigma_v, rho, 1), limit=100)
    P2, _ = quad(_integrand, 1e-9, limit, args=(S, K, T, r, v0, kappa, theta, sigma_v, rho, 2), limit=100)

    P1 = 0.5 + P1 / np.pi
    P2 = 0.5 + P2 / np.pi

    call = S * P1 - K * np.exp(-r * T) * P2
    if option_type == "call":
        return max(call, 0.0)
    # put-call parity
    return max(call - S + K * np.exp(-r * T), 0.0)


def heston_iv_surface(S, strikes, expiries, r, params):
    """Compute Heston implied vol surface for a grid of strikes/expiries."""
    from pricing.bsm import implied_vol
    v0, kappa, theta_h, sigma_v, rho = (
        params["v0"], params["kappa"], params["theta"],
        params["sigma_v"], params["rho"]
    )
    surface = np.zeros((len(expiries), len(strikes)))
    for i, T in enumerate(expiries):
        for j, K in enumerate(strikes):
            price = heston_price(S, K, T, r, v0, kappa, theta_h, sigma_v, rho)
            iv = implied_vol(price, S, K, T, r, "call")
            surface[i, j] = iv if not np.isnan(iv) else np.sqrt(theta_h)
    return surface


if __name__ == "__main__":
    params = dict(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
    S, K, T, r = 500.0, 500.0, 0.25, 0.05
    price = heston_price(S, K, T, r, **params)
    print(f"Heston ATM Call: {price:.4f}")
    from pricing.bsm import implied_vol
    iv = implied_vol(price, S, K, T, r, "call")
    print(f"Heston ATM Implied Vol: {iv:.4f}")
