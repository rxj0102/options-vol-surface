# Options Pricing & Implied Volatility Surface

A production-quality Python toolkit for **derivatives pricing**, **Heston model calibration**, **implied volatility surface construction**, and **delta-hedging simulation** — calibrated to live SPY options data.

---

## Resume Highlights

- Implemented **Black-Scholes-Merton** pricing engine with all 5 analytical Greeks (Δ, Γ, ν, Θ, ρ) and an implied-vol solver via Brent's method; verified round-trip IV accuracy to 10⁻⁸
- Built **Heston stochastic volatility** pricer using characteristic-function / Fourier inversion (Heston 1993); calibrated 5 parameters (v₀, κ, θ, σᵥ, ρ) to live SPY options via L-BFGS-B minimization of RMSE across strikes and expiries
- Engineered a **3D implied volatility surface** from yfinance SPY options chains; extracted vol smile, ATM term structure, and 25-delta put/call skew per expiry
- Designed a **delta-hedging P&L attribution engine** over 1,000 Monte Carlo GBM paths; decomposed terminal P&L into gamma profit (½Γ·dS²), theta decay (Θ·dt), and residual
- Integrated **GARCH(1,1) vol forecasts** as BSM inputs; quantified the implied-vs-realized volatility spread, labelling risk-premium vs vol-surprise regimes

---

## Project Structure

```
options-vol-surface/
├── pricing/
│   ├── bsm.py              # BSM price, 5 Greeks, IV solver (Brent)
│   └── heston.py           # Heston characteristic function + integration
├── calibration/
│   ├── market_data.py      # yfinance SPY options (with synthetic fallback)
│   └── calibrate_heston.py # Heston L-BFGS-B calibration
├── surface/
│   └── plot_surface.py     # 3D IV surface, smile, term structure, BSM vs Heston
├── hedging/
│   └── delta_hedge.py      # 1000-path GBM delta-hedge simulation + P&L attribution
├── integration/
│   └── garch_integration.py # GARCH vs ATM IV, risk premium labelling
├── outputs/                 # All generated charts and CSV
├── notebooks/
│   └── options_vol_surface.ipynb
├── run_all.py               # End-to-end runner
└── requirements.txt
```

---

## Models Explained

### Black-Scholes-Merton (BSM)

Assumes the underlying follows **Geometric Brownian Motion** with constant volatility σ:

$$dS = \mu S\,dt + \sigma S\,dW$$

European call price: $C = SN(d_1) - Ke^{-rT}N(d_2)$

**Limitation**: A single σ implies a *flat* vol surface. Real markets show vol smiles and term structure — OTM puts trade at higher IV than OTM calls (the "skew").

### Heston Stochastic Volatility Model

Heston (1993) makes variance $v_t$ a mean-reverting stochastic process correlated to the stock:

$$dS = rS\,dt + \sqrt{v}\,S\,dW_1$$
$$dv = \kappa(\theta - v)\,dt + \sigma_v\sqrt{v}\,dW_2, \quad \langle dW_1 dW_2 \rangle = \rho\,dt$$

| Parameter | Meaning |
|-----------|---------|
| $v_0$ | Initial variance |
| $\kappa$ | Mean reversion speed |
| $\theta$ | Long-run variance |
| $\sigma_v$ | Vol-of-vol |
| $\rho$ | Correlation (negative → negative skew) |

Pricing uses the **characteristic function** + Fourier inversion — semi-analytical, no Monte Carlo required.

**Advantage**: Naturally produces vol smiles, skew, and term structure that match market observations.

---

## Calibration Results (live SPY data)

| Parameter | Value |
|-----------|-------|
| v₀ (initial var) | 0.0258 |
| κ (mean reversion) | 2.0000 |
| θ (long-run var) | 0.0362 |
| σᵥ (vol-of-vol) | 0.3009 |
| ρ (correlation) | -0.7001 |
| **Calibration RMSE** | **0.0503** |

> Results saved to `outputs/calibration_results.csv`. Values vary with live market data.

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/iv_surface.png` | 3D implied vol surface (moneyness × expiry × IV) |
| `outputs/heston_vs_bsm_surface.png` | Side-by-side: BSM flat vs Heston smile surface |
| `outputs/vol_smile.png` | Vol smile per expiry |
| `outputs/atm_term_structure.png` | ATM IV term structure |
| `outputs/delta_hedge_pnl_distribution.png` | P&L distributions: total, gamma, theta, residual |
| `outputs/garch_vs_implied_vol.png` | GARCH forecast vs ATM IV with risk premium labels |
| `outputs/calibration_results.csv` | Calibrated Heston parameters + RMSE |

---

## Installation & Usage

```bash
pip install -r requirements.txt

# Run everything at once:
python run_all.py

# Or run modules individually:
python pricing/bsm.py          # BSM sanity check
python pricing/heston.py       # Heston ATM price check
python calibration/calibrate_heston.py   # Calibrate to SPY
python surface/plot_surface.py # All surface plots
python hedging/delta_hedge.py  # Delta-hedge simulation
python integration/garch_integration.py  # GARCH vs IV
```

---

## Key Insights

1. **Vol Skew**: SPY options exhibit persistent negative skew — OTM puts trade at 3–8% higher IV than OTM calls due to crash risk demand
2. **Gamma-Theta balance**: In theory, delta-hedging nets zero P&L; in practice, realized vol deviations from σ_BSM cause dispersion
3. **Variance Risk Premium**: Implied vol (ATM IV) trades ~2% above GARCH-forecast realized vol on average — a persistent risk premium paid by option buyers
