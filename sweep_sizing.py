"""
sweep_sizing.py
===============
Sensitivity of the results to storage sizing, across all four network tariff
archetypes.

Energy capacity is held at one day of average community consumption; only the
duration E/P is varied, so the power rating ranges from five times to one times
the community's coincident peak load. The sweep is run for every archetype
because the four provide structurally different signals on storage power:

    T0  none
    T1  indirect, through the volumetric charge on the extra throughput that a
        higher power rating cycles
    T2  direct and linear, through the monthly capacity charge
    T3  direct and stepped, through the connection-capacity band

Model 4 shares the physical dispatch of Model 3 and is settled on the same
solution, so no additional optimization is needed for it.

    python sweep_sizing.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from config import PATHS, DEFAULTS, TARIFF_GRID
from energy_prices import load_energy_price
from models.model3 import run_model3_physical, build_model3_settlement
from models.model4 import build_model4_settlement
from tariffs import ARCHETYPES
from utils import (
    read_inputs_filled, build_con_gen_matrices, build_loc_price_grid, ensure_output_dir,
)

EP_HOURS = (2.0, 4.0, 6.0, 8.0, 10.0)
MODES = ("flat", "tou", "dynamic")
DESIGNS = ("A", "B", "C")
# As in main.py, Settings B and C are evaluated under dynamic pricing only.
MARKET_ACCESS_MODES = ("dynamic",)


def designs_for(mode: str):
    return [d for d in DESIGNS if d == "A" or mode in MARKET_ACCESS_MODES]


def _m0_baseline(mode: str, tariff: str) -> pd.Series:
    """Per-household Model 0 cost including the network charge."""
    p = PATHS.OUTPUT_DIR / f"model0_kpis_{mode}_{tariff}.csv"
    k = pd.read_csv(p, index_col=0)
    k.index = k.index.astype(str)
    return k["cost"]


def _cv(sav: np.ndarray) -> float:
    sav = sav[np.isfinite(sav)]
    if sav.size < 2 or np.isclose(sav.mean(), 0):
        return np.nan
    return float(sav.std(ddof=0) / abs(sav.mean()))


def _best_over_piloc(settle_fn, base: pd.Series, plocs) -> dict:
    """Evaluate the settlement across sharing prices and return the CV-minimizing one."""
    best = None
    for ploc in plocs:
        st = settle_fn(float(ploc))
        cost = st["cost"]
        sav = ((base - cost) / base.abs()).to_numpy(dtype=float)
        cv = _cv(sav)
        rec = dict(pi_loc=float(ploc), cv=cv,
                   total_cost=float(cost.sum()),
                   savings_pct=float((base.sum() - cost.sum()) / base.sum() * 100))
        if best is None or (np.isfinite(cv) and cv < best["cv"]):
            best = rec
    return best


def run() -> pd.DataFrame:
    out = ensure_output_dir(PATHS.OUTPUT_DIR)
    df, unit_ids = read_inputs_filled(PATHS.INPUTS_FILLED)
    con, gen = build_con_gen_matrices(df, unit_ids)
    load = con.sum(axis=1)
    E = float(con.sum().sum() / 365.0)

    rows = []
    t0 = time.time()
    n = len(TARIFF_GRID) * sum(len(designs_for(m)) for m in MODES) * len(EP_HOURS)
    k = 0

    for tariff in TARIFF_GRID:
        tar = ARCHETYPES[tariff]
        for mode in MODES:
            pi = load_energy_price(mode=mode, index=con.index, load=load)
            plocs = build_loc_price_grid(DEFAULTS.PI_EXP, pi, DEFAULTS.LOC_STEP)
            base = _m0_baseline(mode, tariff)

            for design in designs_for(mode):
                for ep in EP_HOURS:
                    k += 1
                    P = E / ep
                    kpi, phys = run_model3_physical(
                        con=con, gen=gen, pi_imp_t=pi, design=design,
                        pi_exp=DEFAULTS.PI_EXP, ces_power_hours=ep, tariff=tar,
                        delta=DEFAULTS.DELTA_LEG,
                    )
                    comm = phys["community"]

                    b3 = _best_over_piloc(
                        lambda pl: build_model3_settlement(kpi, phys, pl, tariff=tar,
                                                           delta=DEFAULTS.DELTA_LEG),
                        base, plocs)
                    b4 = _best_over_piloc(
                        lambda pl: build_model4_settlement(kpi, phys, pl, tau=DEFAULTS.PI_TAU,
                                                           tariff=tar,
                                                           delta=DEFAULTS.DELTA_LEG)[0],
                        base, plocs)

                    rows.append(dict(
                        tariff=tariff, mode=mode, design=design,
                        ep_hours=ep, P_kw=P, E_kwh=E,
                        m3_savings_pct=b3["savings_pct"], m3_cost=b3["total_cost"],
                        m3_pi_loc=b3["pi_loc"], m3_cv=b3["cv"],
                        m4_savings_pct=b4["savings_pct"], m4_cost=b4["total_cost"],
                        peak_import_kw=float(comm["x_imp"].max()),
                        peak_export_kw=float(comm["x_exp"].max()),
                        grid_import_kwh=float(comm["x_imp"].sum()),
                        cycles=float(comm["x_ch"].sum() / E),
                        ces_peak_ch_kw=float(comm["x_ch"].max()),
                    ))
                    print(f"  [{k:3d}/{n}] {tariff} {mode:>7} {design} E/P={ep:>4.1f}h "
                          f"P={P:5.1f}kW  M3 {b3['savings_pct']:6.1f}%  M4 {b4['savings_pct']:6.1f}%")

    res = pd.DataFrame(rows)
    res.to_csv(out / "sweep_sizing.csv", index=False, float_format="%.4f")
    print(f"\ndone in {time.time()-t0:.0f}s -> outputs/sweep_sizing.csv  ({len(res)} rows)")
    return res


if __name__ == "__main__":
    run()
