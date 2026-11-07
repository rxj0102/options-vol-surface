"""Run all modules and generate all outputs."""
import os, sys

os.makedirs("outputs", exist_ok=True)
sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("=" * 60)
    print("Step 1/4: Calibrate Heston model")
    print("=" * 60)
    from calibration.calibrate_heston import calibrate
    params, rmse, iv_df, raw_data = calibrate()

    print("\n" + "=" * 60)
    print("Step 2/4: Plot IV surface and smile")
    print("=" * 60)
    from surface.plot_surface import (plot_iv_surface, plot_vol_smile,
                                       plot_atm_term_structure, plot_heston_vs_bsm)
    plot_iv_surface(iv_df)
    plot_vol_smile(iv_df)
    plot_atm_term_structure(iv_df)
    plot_heston_vs_bsm(params, iv_df)

    print("\n" + "=" * 60)
    print("Step 3/4: Delta-hedge simulation")
    print("=" * 60)
    from hedging.delta_hedge import simulate_delta_hedge, plot_pnl_distribution, print_summary
    results = simulate_delta_hedge()
    print_summary(results)
    plot_pnl_distribution(results)

    print("\n" + "=" * 60)
    print("Step 4/4: GARCH vs Implied Vol")
    print("=" * 60)
    from integration.garch_integration import run
    run()

    print("\n" + "=" * 60)
    print("All outputs generated:")
    for f in sorted(os.listdir("outputs")):
        print(f"  outputs/{f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
