"""
rolling_horizon.py
==================
Does perfect foresight flatter the storage, and by how much?

The main analysis solves the year as a single optimization, which gives the
operator perfect knowledge of prices for 8760 hours. A real operator does not
have that. In a day-ahead market prices are not forecast but published: the gate
closes at 12:00 on D-1, so at any moment between 12 and 36 hours of prices are
known exactly, and nothing beyond.

This script re-solves the dispatch as a sequence of overlapping windows,
committing the first 24 hours of each and carrying the state of charge forward,
and evaluates the committed trajectory on the full annual community cost: energy
purchases net of export revenue, the households' network charges net of the
statutory LEG reduction settled exactly on the realised flows, and the storage
connection charge net of its refund. The dispatch-stage objective alone is also
reported for reference. Two window lengths are reported:

  24 h   commit everything solved. No look-ahead at all, so the battery has no
         reason to hold charge past midnight and empties every night. This is a
         lower bound on what an operator could achieve.
  48 h   solve two days, commit one. This is what the day-ahead gate closure
         actually provides, and is the realistic case.

The contracted connection capacity is held at the value chosen by the annual
solution, because a connection is contracted once for the year and is not a
daily decision.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "models"))

from config import DEFAULTS, PATHS
from energy_prices import load_energy_price
from network_settlement import marginal_leg_rate, settle_network
from tariffs import ARCHETYPES, volumetric_rate_series
from utils import read_inputs_filled, build_con_gen_matrices
from models.model3 import (CommunityStorageParams, solve_community_storage_gurobi,
                           run_model3_physical, _decompose_community_flows,
                           _allocate_proportionally)

COMMIT = 24


def annual_cost(x_imp, x_exp, x_ch, x_dis, pi_imp, pi_exp, tariff, index, leg_rate, contracted_kw):
    """Objective of the annual problem, evaluated on a realised trajectory."""
    cost = float((pi_imp * x_imp).sum() - (pi_exp * x_exp).sum())
    if tariff.name != "T0":
        c = volumetric_rate_series(index, tariff).to_numpy()
        cost += float((c * x_ch).sum()) - float(tariff.refund_rate * x_dis.sum())
        cost += float(tariff.cap_chf_kw_year * (contracted_kw or 0.0))
    return cost


def community_cost(xs, P_sur, D_res, d_res, pi_imp, pi_exp, tariff, index, contracted_kw, delta):
    """Full annual community cost of a realised trajectory, settled as in Model 3:
    energy, household network charges net of the exact LEG reduction, and the
    storage connection charge net of its refund (fixed and metering charges
    included, as in the reported community cost)."""
    energy = float((pi_imp * xs["x_imp"]).sum() - (pi_exp * xs["x_exp"]).sum())
    if tariff.name == "T0":
        return energy
    comm = pd.DataFrame({"P_sur": P_sur, "D_res": D_res, "x_ch": xs["x_ch"],
                         "x_dis": xs["x_dis"], "x_imp": xs["x_imp"], "x_exp": xs["x_exp"]},
                        index=index)
    dcp = _decompose_community_flows(comm)
    buy_loc = _allocate_proportionally(d_res, dcp["x_pv_loc"])
    dis_alloc = _allocate_proportionally(d_res, dcp["x_dis_load"])
    net = settle_network(x_draw=d_res, e_leg_direct=buy_loc.sum(axis=0),
                         x_dis_load_alloc=dis_alloc, tariff=tariff, delta=delta,
                         ces_ch=comm["x_ch"], ces_dis=comm["x_dis"],
                         ces_contracted_kw=contracted_kw,
                         x_ch_local_total=float(dcp["x_ch_pv"].sum()))
    return energy + net.total


def run_rolling(P_sur, D_res, pi_imp, pi_exp, params, index, tariff, leg_rate,
                contracted_kw, window):
    """Solve in overlapping windows, committing COMMIT hours of each."""
    T = len(P_sur)
    xs = {k: np.zeros(T) for k in ("x_ch", "x_dis", "x_imp", "x_exp")}
    soc = float(params.soc0)
    t = 0
    while t < T:
        hi = min(t + window, T)
        sub = CommunityStorageParams(
            E=params.E, P=params.P, eta_ch=params.eta_ch, eta_dis=params.eta_dis,
            soc0=soc, design=params.design, cyclic=False,
        )
        sol = solve_community_storage_gurobi(
            P_sur_t=P_sur[t:hi], D_res_t=D_res[t:hi],
            pi_imp_t=pi_imp[t:hi], pi_exp_t=pi_exp[t:hi],
            params=sub, tariff=tariff, time_index=index[t:hi],
            leg_rate=leg_rate, contracted_kw=contracted_kw,
        )
        n = min(COMMIT, hi - t)
        for k in xs:
            xs[k][t:t + n] = sol[k][:n]
        soc = float(sol["soc"][n])
        t += n
    return xs


def main() -> None:
    df, unit_ids = read_inputs_filled(PATHS.INPUTS_FILLED)
    con, gen = build_con_gen_matrices(df, unit_ids)
    index = con.index
    x_self = pd.DataFrame(np.minimum(con.values, gen.values), index=index, columns=con.columns)
    d_res = con - x_self
    P_sur = (gen - x_self).sum(axis=1).to_numpy(float)
    D_res = d_res.sum(axis=1).to_numpy(float)

    pi_imp_s = load_energy_price(mode="dynamic", index=index, load=con.sum(axis=1))
    pi_imp = pi_imp_s.to_numpy(float)

    rows = []
    for tname in ("T0", "T1", "T2", "T3"):
        tariff = ARCHETYPES[tname]
        leg_rate = marginal_leg_rate(d_res, tariff, DEFAULTS.DELTA_LEG)
        for design in ("A", "B", "C"):
            # annual solution: gives the benchmark and the contracted connection
            kpi, phys = run_model3_physical(
                con=con, gen=gen, pi_imp_t=pi_imp_s, design=design,
                pi_exp=DEFAULTS.PI_EXP, ces_power_hours=DEFAULTS.CES_POWER_HOURS,
                tariff=tariff, delta=DEFAULTS.DELTA_LEG,
            )
            comm = phys["community"]
            ckw = phys.get("contracted_kw")
            pi_exp = comm["pi_exp"].to_numpy(float)
            E = float(phys["battery_E_kwh"]); P = float(phys["battery_P_kw"])
            params = CommunityStorageParams(
                E=E, P=P, eta_ch=DEFAULTS.ETA_CH, eta_dis=DEFAULTS.ETA_DIS,
                soc0=DEFAULTS.SOC0_FRAC * E, design=design, cyclic=True,
            )
            xs_pf = {k: comm[k].to_numpy(float) for k in ("x_ch", "x_dis", "x_imp", "x_exp")}
            pf = annual_cost(xs_pf["x_imp"], xs_pf["x_exp"], xs_pf["x_ch"], xs_pf["x_dis"],
                             pi_imp, pi_exp, tariff, index, leg_rate, ckw)
            pf_full = community_cost(xs_pf, P_sur, D_res, d_res, pi_imp, pi_exp, tariff,
                                     index, ckw, DEFAULTS.DELTA_LEG)

            for window in (24, 48):
                t1 = time.time()
                xs = run_rolling(P_sur, D_res, pi_imp, pi_exp, params, index,
                                 tariff, leg_rate, ckw, window)
                rc = annual_cost(xs["x_imp"], xs["x_exp"], xs["x_ch"], xs["x_dis"],
                                 pi_imp, pi_exp, tariff, index, leg_rate, ckw)
                rc_full = community_cost(xs, P_sur, D_res, d_res, pi_imp, pi_exp, tariff,
                                         index, ckw, DEFAULTS.DELTA_LEG)
                rows.append(dict(
                    tariff=tname, design=design, window=window,
                    # full annual community cost (reported in S8.7)
                    perfect_foresight=pf_full, rolling=rc_full,
                    penalty_chf=rc_full - pf_full,
                    penalty_pct=100.0 * (rc_full - pf_full) / abs(pf_full) if pf_full else np.nan,
                    # dispatch-stage objective only, for reference
                    perfect_foresight_dispatch=pf, rolling_dispatch=rc,
                    penalty_pct_dispatch=100.0 * (rc - pf) / abs(pf) if pf else np.nan,
                    cycles_pf=float(comm["x_ch"].sum() / E),
                    cycles_roll=float(xs["x_ch"].sum() / E),
                    seconds=time.time() - t1,
                ))
                print(f"  {tname} M3{design} window={window}h: "
                      f"PF {pf_full:8.1f}  rolling {rc_full:8.1f}  "
                      f"penalty {rc_full-pf_full:+7.1f} CHF ({100*(rc_full-pf_full)/abs(pf_full):+5.2f}%)  "
                      f"[{time.time()-t1:.0f}s]")

    out = pd.DataFrame(rows)
    out.to_csv(PATHS.OUTPUT_DIR / "rolling_horizon.csv", index=False, float_format="%.4f")
    print(f"\n-> outputs/rolling_horizon.csv  ({len(out)} rows)")


if __name__ == "__main__":
    main()
