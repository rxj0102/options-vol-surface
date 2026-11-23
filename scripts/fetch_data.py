"""Download the inputs and save them under data/.

    python scripts/fetch_data.py            # both
    python scripts/fetch_data.py --chain    # SPY option chain snapshot only
    python scripts/fetch_data.py --history  # VIX and S&P 500 daily closes only
    python scripts/fetch_data.py --vix-day  # VIX open, high, low, close on the snapshot's date

The option chain is a snapshot: free sources serve only the current chain,
so the committed file is the one every result in the README is computed from. A new
snapshot changes the surface results; the VIX study uses history and is
stable apart from new months being added.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MIN_DAYS, MAX_DAYS = 14, 400


CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json"


TREASURY = {"DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0, "DGS2": 2.0}


def treasury_curve(on_or_before: str):
    """Constant maturity Treasury yields (FRED, no key) on the latest date with
    data up to and including the quote session. Returns (date, {years: yield})."""
    import io
    import requests
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(TREASURY)
    df = pd.read_csv(io.StringIO(requests.get(url, timeout=120).text), index_col=0, na_values=".")
    df = df.loc[:on_or_before].dropna()
    row = df.iloc[-1]
    return str(df.index[-1]), {f"{TREASURY[k]:.4f}": round(float(row[k]) / 100, 6) for k in TREASURY}


def fetch_chain():
    """SPY option chain from Cboe's free delayed quote feed.

    Yahoo returns zero bids and asks outside market hours, so the chain comes
    from Cboe, which serves the last quotes of the most recent session. The
    quote time recorded is that session's 16:15 New York option close, and the
    spot is the underlying's official close. During market hours the feed is
    delayed, and the quote time is the underlying's last trade in it.
    """
    import requests
    raw = requests.get(CBOE_URL, timeout=120).json()
    d = raw["data"]
    session = d["last_trade_time"][:10]               # underlying's last trade, New York date
    last = pd.Timestamp(d["last_trade_time"], tz="America/New_York")
    intraday = last.hour < 16
    if intraday:
        # During the session the feed is delayed; quotes and the underlying's
        # current price are as of its last trade.
        quote_utc = last.tz_convert("UTC")
        spot = float(d["current_price"])
    else:
        quote_utc = pd.Timestamp(f"{session} 16:15", tz="America/New_York").tz_convert("UTC")
        spot = float(d["close"])
    rows = []
    for o in d["options"]:
        m = re.fullmatch(r"SPY(\d{6})([CP])(\d{8})", o["option"])
        if not m:                                       # adjusted roots such as B1SPY
            continue
        exp = pd.Timestamp("20" + m.group(1))
        days = (exp.tz_localize("UTC") - quote_utc).days
        if not MIN_DAYS <= days <= MAX_DAYS:
            continue
        rows.append(dict(expiry=exp.strftime("%Y-%m-%d"), type="call" if m.group(2) == "C" else "put",
                         strike=int(m.group(3)) / 1000, bid=o["bid"], ask=o["ask"],
                         lastPrice=o["last_trade_price"], volume=o["volume"],
                         openInterest=o["open_interest"], lastTradeDate=o["last_trade_time"]))
    chain = pd.DataFrame(rows).sort_values(["expiry", "type", "strike"]).reset_index(drop=True)
    DATA.mkdir(exist_ok=True)
    path = DATA / f"spy_chain_{session}.csv.gz"
    chain.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    rates_date, rates = treasury_curve(session)
    meta = {"file": path.name,
            "source": "Cboe delayed quotes, " + ("intraday" if intraday else "last quotes of the session"),
            "spot": spot, "pulled_utc": quote_utc.strftime("%Y-%m-%d %H:%M"),
            "treasury_date": rates_date, "treasury_yields": rates}
    (DATA / "spy_chain.json").write_text(json.dumps(meta, indent=2) + "\n")
    quoted = int(((chain.bid > 0) & (chain.ask > chain.bid)).sum())
    print(f"saved {path.name}: {len(chain)} quotes ({quoted} two-sided), "
          f"{chain.expiry.nunique()} expiries, spot {spot}, quotes as of {quote_utc} UTC")


def fetch_history():
    import yfinance as yf
    out = {}
    for name, t in (("vix", "^VIX"), ("spx", "^GSPC")):
        d = yf.download(t, start="1990-01-01", progress=False, auto_adjust=False)["Close"]
        out[name] = d.iloc[:, 0] if isinstance(d, pd.DataFrame) else d
    df = pd.concat(out, axis=1).dropna()
    df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    df.index.name = "date"
    DATA.mkdir(exist_ok=True)
    df.round(4).to_csv(DATA / "vix_spx.csv.gz", compression={"method": "gzip", "mtime": 0})
    print(f"saved vix_spx.csv.gz: {len(df)} days, {df.index[0]} to {df.index[-1]}")


VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def fetch_vix_day():
    """Record VIX's open, high, low and close on the snapshot's session date in
    data/spy_chain.json, from Cboe's free daily history. Run it after that
    session's close. Free sources have no intraday VIX at the snapshot's
    minute, so the day's range is what the model-free SPY number is checked
    against."""
    import io
    import requests
    meta_path = DATA / "spy_chain.json"
    meta = json.loads(meta_path.read_text())
    session = pd.Timestamp(meta["pulled_utc"], tz="UTC").tz_convert("America/New_York").strftime("%Y-%m-%d")
    df = pd.read_csv(io.StringIO(requests.get(VIX_HISTORY_URL, timeout=120).text))
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.strftime("%Y-%m-%d")
    row = df[df["DATE"] == session]
    if row.empty:
        sys.exit(f"no VIX row for {session} yet; run this after that session closes")
    r = row.iloc[0]
    meta["vix_day"] = dict(date=session, source="Cboe VIX_History.csv",
                           **{k.lower(): float(r[k]) for k in ("OPEN", "HIGH", "LOW", "CLOSE")})
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"VIX on {session}: {meta['vix_day']}")


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if not args or "--chain" in args:
        fetch_chain()
    if not args or "--history" in args:
        fetch_history()
    if "--vix-day" in args:
        fetch_vix_day()
