"""Fetch SPY options data from yfinance; fall back to synthetic data."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def fetch_spy_options(n_expiries=3, min_days=21):
    """Return list of (expiry_str, calls_df, puts_df, S, T_years).

    Expiries closer than ``min_days`` are skipped: a diffusive Heston model
    cannot fit the very wide smiles of options a few days from expiry.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("SPY")
        S = ticker.history(period="1d")["Close"].iloc[-1]
        today = datetime.today()
        exps = [e for e in ticker.options
                if (datetime.strptime(e, "%Y-%m-%d") - today).days >= min_days][:n_expiries]
        result = []
        for exp in exps:
            chain = ticker.option_chain(exp)
            T = (datetime.strptime(exp, "%Y-%m-%d") - today).days / 365.0
            if T <= 0:
                continue
            result.append((exp, chain.calls, chain.puts, S, T))
        if result:
            print(f"[data] Fetched live SPY data. S={S:.2f}, {len(result)} expiries.")
            return result
    except Exception as e:
        print(f"[data] yfinance failed ({e}), using synthetic data.")

    return _synthetic_options()


def _synthetic_options():
    """Realistic synthetic SPY options with vol smile."""
    np.random.seed(42)
    S = 530.0
    r = 0.05
    today = datetime.today()
    result = []

    from pricing.bsm import bsm_price

    expiry_days = [30, 60, 90]
    for days in expiry_days:
        T = days / 365.0
        exp_str = (today + timedelta(days=days)).strftime("%Y-%m-%d")

        moneyness = np.linspace(0.85, 1.15, 25)
        strikes = np.round(S * moneyness, 0)

        atm_iv = 0.18
        skew_slope = -0.15
        smile_curve = 0.05

        rows_calls, rows_puts = [], []
        for K in strikes:
            m = np.log(K / S)
            iv = atm_iv + skew_slope * m + smile_curve * m**2
            iv = max(iv, 0.05)
            noise = np.random.normal(0, 0.002)
            iv_noisy = iv + noise

            call_mid = bsm_price(S, K, T, r, iv_noisy, "call")
            put_mid = bsm_price(S, K, T, r, iv_noisy, "put")
            spread = max(call_mid * 0.01, 0.05)

            rows_calls.append(dict(
                strike=K, lastPrice=call_mid,
                bid=call_mid - spread, ask=call_mid + spread,
                impliedVolatility=iv_noisy, inTheMoney=(S > K)
            ))
            rows_puts.append(dict(
                strike=K, lastPrice=put_mid,
                bid=put_mid - spread, ask=put_mid + spread,
                impliedVolatility=iv_noisy, inTheMoney=(S < K)
            ))

        result.append((exp_str, pd.DataFrame(rows_calls), pd.DataFrame(rows_puts), S, T))

    print(f"[data] Synthetic SPY data: S={S}, {len(result)} expiries.")
    return result


if __name__ == "__main__":
    data = fetch_spy_options()
    for exp, calls, puts, S, T in data:
        print(f"  Expiry {exp}  T={T:.3f}y  strikes: {len(calls)}")
