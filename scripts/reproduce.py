"""Run both studies on the committed data and write results/metrics.json and the figures.

    python scripts/reproduce.py

Inputs: data/vix_spx.csv.gz and the SPY chain snapshot named in
data/spy_chain.json (see scripts/fetch_data.py). Numbers are rounded to six
significant figures so reruns compare exactly.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integration.vrp import study as vrp_study          # noqa: E402
from surface.chain import load_snapshot, mids, smile_stats, surface  # noqa: E402
from surface.fit import density_summary, fit_heston, fit_svi_surface, heston_ivs, svi_density  # noqa: E402

OUT = ROOT / "results"


def rnd(x):
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.6g}")
    return x


def surface_study():
    chain, meta = load_snapshot()
    iv, fwd = surface(chain, meta["spot"], meta["treasury_yields"])
    p = fit_heston(iv)
    iv["heston"] = heston_ivs(iv, p)
    svi, iv["svi"], butterfly, calendar, krange = fit_svi_surface(iv)
    smile = smile_stats(iv)

    def rmse(a, b):
        d = (a - b).dropna()
        return float(100 * np.sqrt((d ** 2).mean()))

    per = []
    for exp, s in iv.groupby("expiry"):
        per.append(dict(expiry=exp, T=float(s["T"].iloc[0]), n=len(s),
                        k_min=float(s["k"].min()), k_max=float(s["k"].max()),
                        heston_rmse=rmse(s["heston"], s["iv"]), svi_rmse=rmse(s["svi"], s["iv"])))
    for r in per:
        d = density_summary(svi[r["expiry"]])
        # The mean of S_T / F is 1 in theory once Lee's bound holds, but with a
        # wing slope near 2 the integral converges too slowly for a finite grid
        # to show it, so it is not reported.
        d.pop("mean_over_forward")
        r["density"] = d
    # Model-free check of the density: a tight put spread around 90% of the
    # forward prices P(S_T < K) directly, (P(K2) - P(K1)) / ((K2 - K1) D).
    m = mids(chain)
    fwd_i = fwd.set_index("expiry")
    for r in per:
        F, D = fwd_i.loc[r["expiry"], "F"], fwd_i.loc[r["expiry"], "D"]
        puts = m[(m.expiry == r["expiry"]) & (m.type == "put")].set_index("strike")["mid"].sort_index()
        ks = puts.index.to_numpy(float)
        i = int(np.argmin(np.abs(ks - 0.9 * F)))
        if 0 < i < len(ks) - 1:
            k1, k2 = ks[i - 1], ks[i + 1]
            sp = svi[r["expiry"]]
            kg = np.linspace(-8.0, np.log(0.5 * (k1 + k2) / F), 40001)
            q = svi_density(kg, sp["a"], sp["b"], sp["rho"], sp["m"], sp["sigma"])
            r["put_spread_check"] = dict(k1=float(k1), k2=float(k2),
                                         market=float((puts[k2] - puts[k1]) / ((k2 - k1) * D)),
                                         svi=float(np.trapezoid(q, kg)))
    sm = smile.set_index("expiry")
    for r in per:
        if r["expiry"] in sm.index:
            r.update(atm=float(sm.loc[r["expiry"], "atm"]), rr25=float(sm.loc[r["expiry"], "rr25"]),
                     bf25=float(sm.loc[r["expiry"], "bf25"]))
    short = iv["T"] < 0.15
    res = dict(
        snapshot=meta["file"], spot=meta["spot"], pulled_utc=meta["pulled_utc"],
        n_quotes_raw=int(len(chain)), n_quotes_used=int(len(iv)), n_expiries=int(iv["expiry"].nunique()),
        forwards=fwd.drop(columns=["n_parity"]).to_dict("records"),
        heston={k: v for k, v in p.items()},
        heston_rmse=rmse(iv["heston"], iv["iv"]),
        heston_rmse_short=rmse(iv.loc[short, "heston"], iv.loc[short, "iv"]),
        heston_rmse_long=rmse(iv.loc[~short, "heston"], iv.loc[~short, "iv"]),
        svi_rmse=rmse(iv["svi"], iv["iv"]),
        svi_butterfly_min_g=float(min(butterfly.values())),
        svi_butterfly_violations=int(sum(v < 0 for v in butterfly.values())),
        svi_calendar_violations=int(sum(c["min_increase"] < 0 for c in calendar)),
        svi_calendar_k_range=list(krange),
        density_max_mass_error=float(max(abs(r["density"]["mass"] - 1) for r in per)),
        put_spread_max_gap=float(max(abs(r["put_spread_check"]["market"] - r["put_spread_check"]["svi"])
                                     for r in per if "put_spread_check" in r)),
        put_spread_n=int(sum("put_spread_check" in r for r in per)),
        svi_wing_slope_max=float(max(max(r["density"]["wing_slope_left"], r["density"]["wing_slope_right"]) for r in per)),
        per_expiry=per,
    )
    return res, iv, svi


def figures(iv, svi, blocks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True)

    exps = sorted(iv["expiry"].unique(), key=lambda e: iv.loc[iv.expiry == e, "T"].iloc[0])
    pick = [exps[i] for i in np.linspace(0, len(exps) - 1, min(4, len(exps))).astype(int)]
    fig, axes = plt.subplots(1, len(pick), figsize=(4 * len(pick), 3.4), sharey=True)
    for ax, e in zip(np.atleast_1d(axes), pick):
        s = iv[iv.expiry == e].sort_values("k")
        ax.plot(s["k"], 100 * s["iv"], "o", ms=3, color="0.4", label="market (mid)")
        ax.plot(s["k"], 100 * s["svi"], "-", lw=1.6, color="#2a6fb0", label="SVI")
        ax.plot(s["k"], 100 * s["heston"], "--", lw=1.6, color="#c0504d", label="Heston")
        ax.set_title(f"{e} ({365 * s['T'].iloc[0]:.0f} days)", fontsize=10)
        ax.set_xlabel("log(K / F)")
    np.atleast_1d(axes)[0].set_ylabel("implied vol (%)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "smiles.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax[0].plot(blocks.index, blocks["vix"], lw=1, color="#c0504d", label="VIX at block start")
    ax[0].plot(blocks.index, 100 * blocks["realized_vol"], lw=1, color="0.3", label="realized vol, next 21 days")
    ax[0].set_ylabel("vol (%)")
    ax[0].legend(frameon=False, fontsize=8)
    pnl = 1e4 * (blocks["implied"] - blocks["realized"])
    ax[1].plot(blocks.index, pnl.cumsum(), lw=1.2, color="#2a6fb0")
    ax[1].set_ylabel("cumulative variance points")
    ax[1].set_title("Selling one unit of 21 day variance at VIX squared each block", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "vix_vs_realized.png", dpi=130)
    plt.close(fig)


def main():
    vrp, blocks = vrp_study()
    surf, iv, svi = surface_study()
    metrics = rnd(dict(vrp=vrp, surface=surf))
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    iv.round(6).to_csv(OUT / "surface_quotes.csv", index=False)
    figures(iv, svi, blocks)
    print(f"wrote results/metrics.json ({surf['n_quotes_used']} quotes, {vrp['n_blocks']} VIX blocks)")


if __name__ == "__main__":
    main()
