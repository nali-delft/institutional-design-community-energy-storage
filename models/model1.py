"""
models/model1.py
================
Model 1 - Energy Community (EC) only, no storage.

Households self-consume first; residual PV surplus and residual demand are
shared locally at the internal price pi_loc (proportional buy/sell allocation).
Any leftover surplus is exported at pi_exp and leftover demand imported at
pi_imp. pi_loc enters only ex-post, so the physical flows are solved once and
pi_loc is then scanned over a grid for settlement.

Reads : input_data/inputs_filled.csv; retail prices via energy_prices.py
Writes: outputs/model1_{summary,flows}_<mode>_<tariff>.csv,
        outputs/model1_kpis_<mode>_<tariff>_piLoc_*.csv
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path("..").resolve()))

from config import PATHS, DEFAULTS, PriceMode
from tariffs import ARCHETYPES
from network_settlement import settle_network
from energy_prices import load_energy_price
from utils import (
    read_inputs_filled,
    build_con_gen_matrices,
    build_loc_price_grid,
    gini,
    ensure_output_dir,
    save_kpi_and_summary,
)


# =========================================================
# Model 1 core
# =========================================================
def run_model1_ec_only(
    con: pd.DataFrame,
    gen: pd.DataFrame,
    pi_imp_t: pd.Series,
    pi_exp: float = DEFAULTS.PI_EXP,
    pi_loc: float = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Model 1 (Energy Community only; no storage):
      - household self-consumption first
      - residual PV surplus + residual demand are shared locally at price pi_loc
      - any remaining surplus is exported at pi_exp, remaining demand imported at pi_imp_t
      - proportional allocation of local exchange (buy/sell shares)

    Cost per household:
      sum_t [ pi_imp_t*x_imp + pi_loc*x_buy - pi_loc*x_sell - pi_exp*x_exp ]
    """
    if pi_loc is None:
        raise ValueError("pi_loc must be provided for Model 1.")

    # Align time and columns
    gen, con = gen.align(con, join="inner", axis=0)
    gen, con = gen.align(con, join="inner", axis=1)

    # Align price
    pi_imp_t = pi_imp_t.reindex(con.index).astype(float)
    if pi_imp_t.isna().any():
        raise ValueError("pi_imp_t has NaN after reindex; check time alignment with input index.")

    # Sanity checks
    if con.isna().any().any() or gen.isna().any().any():
        raise ValueError("con/gen contains NaN. Fill missing data first.")
    if (con < 0).any().any() or (gen < 0).any().any():
        raise ValueError("con/gen contains negative values.")

    # Self-consumption
    x_self = np.minimum(con, gen)

    # Residual surplus & demand
    p_sur = gen - x_self
    d_res = con - x_self

    # Community totals
    P_sur = p_sur.sum(axis=1)  # total surplus each hour
    D_res = d_res.sum(axis=1)  # total residual demand each hour

    # Total local exchange each hour
    x_loc_t = np.minimum(P_sur, D_res)

    # Proportional allocation (safe division)
    share_buy = d_res.div(D_res.replace(0.0, np.nan), axis=0).fillna(0.0)
    share_sell = p_sur.div(P_sur.replace(0.0, np.nan), axis=0).fillna(0.0)

    x_buy = share_buy.mul(x_loc_t, axis=0)
    x_sell = share_sell.mul(x_loc_t, axis=0)

    # Residual grid exchanges
    x_imp = (d_res - x_buy)
    x_exp = (p_sur - x_sell)

    # Numerical clipping
    x_buy = x_buy.clip(lower=0.0)
    x_sell = x_sell.clip(lower=0.0)
    x_imp = x_imp.clip(lower=0.0)
    x_exp = x_exp.clip(lower=0.0)

    # Cost time series and annual cost
    cost_ts = (
        x_imp.mul(pi_imp_t, axis=0)
        + x_buy * float(pi_loc)
        - x_sell * float(pi_loc)
        - x_exp * float(pi_exp)
    )
    cost = cost_ts.sum(axis=0)

    # KPIs per household
    annual_load = con.sum(axis=0)
    annual_pv = gen.sum(axis=0)
    self_energy = x_self.sum(axis=0)

    kpi = pd.DataFrame(index=con.columns)
    kpi["annual_load"] = annual_load
    kpi["annual_pv"] = annual_pv
    kpi["self_consumption"] = self_energy
    kpi["import"] = x_imp.sum(axis=0)
    kpi["export"] = x_exp.sum(axis=0)
    kpi["local_buy"] = x_buy.sum(axis=0)
    kpi["local_sell"] = x_sell.sum(axis=0)
    kpi["cost"] = cost

    # kpi["self_consumption_ratio"] = self_energy / annual_pv.replace(0, np.nan)
    kpi["self_consumption_ratio"] = ((kpi["self_consumption"] + kpi["local_sell"]) / kpi["annual_pv"].replace(0, np.nan))
    kpi["self_sufficiency_ratio"] = ((kpi["self_consumption"] + kpi["local_sell"]) / kpi["annual_load"].replace(0, np.nan))
    
    kpi["has_pv"] = (annual_pv > 1e-12).astype(int)

    flows = {
        "x_self": x_self,
        "p_sur": p_sur,
        "d_res": d_res,
        "x_buy": x_buy,
        "x_sell": x_sell,
        "x_imp": x_imp,
        "x_exp": x_exp,
        "cost_ts": cost_ts,
        "community": pd.DataFrame({"P_sur": P_sur, "D_res": D_res, "x_loc": x_loc_t}),
        "pi_imp_t": pi_imp_t,
        "pi_loc": float(pi_loc),
        "pi_exp": float(pi_exp),
    }
    return kpi, flows


# =========================================================
# Unified runner (same style as Model 0)
# =========================================================
def run_model1_for_price_mode(
    mode: PriceMode,
    pi_exp: float = DEFAULTS.PI_EXP,
    loc_step: float = DEFAULTS.LOC_STEP,
    save_flows: bool = True,
    tariff: str = DEFAULTS.TARIFF,
    delta: float = DEFAULTS.DELTA_LEG,
) -> pd.DataFrame:
    """
    Runner for Model 1:
      - physical flows do NOT depend on pi_loc
      - ex-post settlement (costs) does depend on pi_loc
    """
    out = ensure_output_dir(PATHS.OUTPUT_DIR)

    # Inputs
    df, unit_ids = read_inputs_filled(PATHS.INPUTS_FILLED)
    con, gen = build_con_gen_matrices(df, unit_ids)

    tar = ARCHETYPES[tariff]
    suffix = f"{mode}_{tar.name}"

    # Energy component only, CHF excl. VAT (Supplementary Information S8.3)
    pi_imp_t = load_energy_price(mode=mode, index=con.index, load=con.sum(axis=1))

    # Candidate local prices
    loc_prices = build_loc_price_grid(pi_exp=pi_exp, pi_imp_t=pi_imp_t, step=loc_step)

    # -------------------------------------------------
    # Step 1: run once to get physical flows
    # -------------------------------------------------
    ref_ploc = float(loc_prices[0])
    _, ref_flows = run_model1_ec_only(
        con=con,
        gen=gen,
        pi_imp_t=pi_imp_t,
        pi_exp=pi_exp,
        pi_loc=ref_ploc,
    )

    if save_flows:
        flows_wide = pd.concat(
            [
                ref_flows["x_self"].add_prefix("x_self__"),
                ref_flows["p_sur"].add_prefix("p_sur__"),
                ref_flows["d_res"].add_prefix("d_res__"),
                ref_flows["x_buy"].add_prefix("x_buy__"),
                ref_flows["x_sell"].add_prefix("x_sell__"),
                ref_flows["x_imp"].add_prefix("x_imp__"),
                ref_flows["x_exp"].add_prefix("x_exp__"),
            ],
            axis=1,
        )

        flows_wide["total_load"] = con.sum(axis=1)
        flows_wide["total_pv"] = gen.sum(axis=1)
        flows_wide["total_x_self"] = ref_flows["x_self"].sum(axis=1)
        flows_wide["total_p_sur"] = ref_flows["p_sur"].sum(axis=1)
        flows_wide["total_d_res"] = ref_flows["d_res"].sum(axis=1)
        flows_wide["total_x_buy"] = ref_flows["x_buy"].sum(axis=1)
        flows_wide["total_x_sell"] = ref_flows["x_sell"].sum(axis=1)
        flows_wide["total_x_imp"] = ref_flows["x_imp"].sum(axis=1)
        flows_wide["total_x_exp"] = ref_flows["x_exp"].sum(axis=1)
        flows_wide["pi_imp"] = ref_flows["pi_imp_t"]

        flows_wide.to_csv(
            out / f"model1_flows_{suffix}.csv",
            float_format="%.6f"
        )

    # -------------------------------------------------
    # Step 2: scan pi_loc only for ex-post settlement
    # -------------------------------------------------
    summary_rows = []

    for ploc in loc_prices:
        kpi, flows = run_model1_ec_only(
            con=con,
            gen=gen,
            pi_imp_t=pi_imp_t,
            pi_exp=pi_exp,
            pi_loc=float(ploc),
        )

        # Network charges and the LEG reduction. Model 1 is a community without
        # storage, so all eligible energy is direct local sharing.
        net_household = pd.Series(0.0, index=kpi.index)
        net = None
        if tar.name != "T0":
            net = settle_network(
                x_draw=flows["d_res"],
                e_leg_direct=flows["x_buy"].sum(axis=0),
                x_dis_load_alloc=None,
                tariff=tar,
                delta=delta,
            )
            net_household = net.household_cost.reindex(kpi.index).fillna(0.0)
        kpi["network_cost"] = net_household
        kpi["cost"] = kpi["cost"] + net_household
        kpi["tariff"] = tar.name
        kpi["delta_leg"] = float(delta) if tar.name != "T0" else 0.0

        costs = kpi["cost"].to_numpy(dtype=float)
        std_cost = float(np.std(costs))

        summary_rows.append({
            "model": "model1",
            "tariff": tar.name,
            "delta_leg": float(delta) if tar.name != "T0" else 0.0,
            "total_network_cost": float(net_household.sum()),
            "total_network_reduction": float(net.household_reduction.sum()) if net is not None else 0.0,
            "mode": mode,
            "pi_exp": float(pi_exp),
            "pi_loc": float(ploc),
            "total_cost": float(kpi["cost"].sum()),
            "total_local_exchange": float(flows["x_buy"].to_numpy().sum()),
            "total_import": float(kpi["import"].sum()),
            "total_export": float(kpi["export"].sum()),
            "self_consumption_ratio_agg": float(
                (kpi["self_consumption"].sum() + kpi["local_sell"].sum()) / kpi["annual_pv"].sum()
            ) if kpi["annual_pv"].sum() > 0 else np.nan,
            "self_sufficiency_ratio_agg": float(
                (kpi["self_consumption"].sum() + kpi["local_buy"].sum()) / kpi["annual_load"].sum()
            ) if kpi["annual_load"].sum() > 0 else np.nan,
        })

        # Optional: save KPI per pi_loc
        kpi.to_csv(
            out / f"model1_kpis_{suffix}_piLoc_{ploc:.2f}.csv",
            float_format="%.6f"
        )

    summary = pd.DataFrame(summary_rows).sort_values(["mode", "pi_loc"])
    summary.to_csv(out / f"model1_summary_{suffix}.csv", index=False)


    print(f"[OK] model1 finished for mode={mode} -> {out}")
    return summary

# =========================================================
# Debug
# =========================================================
if __name__ == "__main__":
    for m in ["flat", "tou", "dynamic"]:
        run_model1_for_price_mode(
            mode=m,
            pi_exp=DEFAULTS.PI_EXP,
            loc_step=DEFAULTS.LOC_STEP,
            save_flows=True,
        )