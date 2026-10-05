"""
models/model0.py
================
Model 0 - Reference case (status quo: no community, no storage).

Each household acts independently: it self-consumes its own PV, imports its
residual demand from the grid at the retail price pi_imp, and exports its
residual PV surplus at pi_exp. Provides the per-household cost baseline that
Models 1-4 are benchmarked against.

Reads : input_data/inputs_filled.csv; retail prices via energy_prices.py
Writes: outputs/model0_{summary,kpis,flows}_<mode>_<tariff>.csv
"""
import numpy as np
import pandas as pd
from typing import Tuple
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
    ensure_output_dir,
    save_kpi_and_summary,
)


# =========================================================
# Model 0 core
# =========================================================
def run_model0(
    con: pd.DataFrame,
    gen: pd.DataFrame,
    pi_imp_t: pd.Series,
    pi_exp: float = DEFAULTS.PI_EXP,
) -> Tuple[pd.DataFrame, dict]:
    """
    Model 0 (reference case):
      - households act independently (no sharing, no storage)
      - self-consume PV, then import/export with the grid
    """
    # Align time and columns
    gen, con = gen.align(con, join="inner", axis=0)
    gen, con = gen.align(con, join="inner", axis=1)

    # Align price
    pi_imp_t = pi_imp_t.reindex(con.index).astype(float)
    if pi_imp_t.isna().any():
        raise ValueError("pi_imp_t has NaN after reindex; check time alignment with input index.")

    # Basic sanity
    if con.isna().any().any() or gen.isna().any().any():
        raise ValueError("con/gen contains NaN. Fill missing data first.")
    if (con < 0).any().any() or (gen < 0).any().any():
        raise ValueError("con/gen contains negative values.")

    # Flows
    x_self = np.minimum(con, gen)
    x_imp = con - x_self
    x_exp = gen - x_self

    # Costs (CHF)
    cost_ts = x_imp.mul(pi_imp_t, axis=0) - x_exp * float(pi_exp)
    cost = cost_ts.sum(axis=0)

    # KPIs
    annual_load = con.sum(axis=0)
    annual_pv = gen.sum(axis=0)
    self_energy = x_self.sum(axis=0)

    kpi = pd.DataFrame(index=con.columns)
    kpi["annual_load"] = annual_load
    kpi["annual_pv"] = annual_pv
    kpi["import"] = x_imp.sum(axis=0)
    kpi["export"] = x_exp.sum(axis=0)
    kpi["self_consumption"] = self_energy
    kpi["cost"] = cost
    kpi["self_consumption_ratio"] = self_energy / annual_pv.replace(0, np.nan)
    kpi["self_sufficiency_ratio"] = self_energy / annual_load.replace(0, np.nan)

    # Aggregated physical flows for analysis / plotting
    physical = pd.DataFrame(index=con.index)
    physical["total_load"] = con.sum(axis=1)
    physical["total_pv"] = gen.sum(axis=1)
    physical["x_self"] = x_self.sum(axis=1)
    physical["x_imp"] = x_imp.sum(axis=1)
    physical["x_exp"] = x_exp.sum(axis=1)
    physical["pi_imp"] = pi_imp_t
    physical["cost_ts"] = cost_ts.sum(axis=1)

    flows = {
        "x_self": x_self,
        "x_imp": x_imp,
        "x_exp": x_exp,
        "cost_ts": cost_ts,
        "pi_imp_t": pi_imp_t,
        "physical": physical,
    }
    return kpi, flows


# =========================================================
# Unified runner
# =========================================================
def run_model0_for_price_mode(
    mode: PriceMode,
    pi_exp: float = DEFAULTS.PI_EXP,
    save_flows: bool = True,
    tariff: str = DEFAULTS.TARIFF,
) -> pd.DataFrame:
    out = ensure_output_dir(PATHS.OUTPUT_DIR)

    df, unit_ids = read_inputs_filled(PATHS.INPUTS_FILLED)
    con, gen = build_con_gen_matrices(df, unit_ids)

    tar = ARCHETYPES[tariff]
    suffix = f"{mode}_{tar.name}"
    pi_imp_t = load_energy_price(mode=mode, index=con.index, load=con.sum(axis=1))

    kpi, flows = run_model0(con=con, gen=gen, pi_imp_t=pi_imp_t, pi_exp=pi_exp)

    # Network charges. Model 0 has no community, so no LEG reduction applies
    # (delta = 0): the reduction is conditional on membership.
    net_household = pd.Series(0.0, index=kpi.index)
    if tar.name != "T0":
        net = settle_network(
            x_draw=flows["x_imp"],
            e_leg_direct=pd.Series(0.0, index=kpi.index),
            x_dis_load_alloc=None,
            tariff=tar,
            delta=0.0,
        )
        net_household = net.household_cost.reindex(kpi.index).fillna(0.0)
    kpi["network_cost"] = net_household
    kpi["cost"] = kpi["cost"] + net_household
    kpi["tariff"] = tar.name

    summary = {
        "model": "model0",
        "mode": mode,
        "pi_exp": float(pi_exp),
        "total_cost": float(kpi["cost"].sum()),
        "total_network_cost": float(net_household.sum()),
        "tariff": tar.name,
        "total_import": float(kpi["import"].sum()),
        "total_export": float(kpi["export"].sum()),
        "self_consumption_ratio_agg": float(
            kpi["self_consumption"].sum() / kpi["annual_pv"].sum()
        ) if kpi["annual_pv"].sum() > 0 else np.nan,
        "self_sufficiency_ratio_agg": float(
            kpi["self_consumption"].sum() / kpi["annual_load"].sum()
        ) if kpi["annual_load"].sum() > 0 else np.nan,
    }

    # Per-household KPIs including the network charge. The flows file carries
    # only the energy-cost timeseries, so downstream analysis must read this
    # file for the baseline against which Models 1-4 are compared.
    kpi.to_csv(out / f"model0_kpis_{suffix}.csv", float_format="%.6f")

    save_kpi_and_summary(
        model_name="model0",
        mode=suffix,
        kpi=kpi,
        summary_dict=summary,
        output_dir=out,
    )

    if save_flows:
    # Combine all household-level flows into one wide dataframe
        flows_wide = pd.concat(
            [
                flows["x_self"].add_prefix("x_self__"),
                flows["x_imp"].add_prefix("x_imp__"),
                flows["x_exp"].add_prefix("x_exp__"),
                flows["cost_ts"].add_prefix("cost__"),
            ],
            axis=1,
        )

        # Optional: add community-level aggregates
        flows_wide["total_load"] = con.sum(axis=1)
        flows_wide["total_pv"] = gen.sum(axis=1)
        flows_wide["total_x_self"] = flows["x_self"].sum(axis=1)
        flows_wide["total_x_imp"] = flows["x_imp"].sum(axis=1)
        flows_wide["total_x_exp"] = flows["x_exp"].sum(axis=1)
        flows_wide["total_cost"] = flows["cost_ts"].sum(axis=1)
        flows_wide["pi_imp"] = flows["pi_imp_t"]

        flows_wide.to_csv(
            out / f"model0_flows_{suffix}.csv",
            float_format="%.6f"
        )

    print(f"[OK] model0 finished for mode={mode} -> {out}")
    return kpi


# =========================================================
# Debug
# =========================================================
if __name__ == "__main__":
    for m in ["flat", "tou", "dynamic"]:
        run_model0_for_price_mode(mode=m, pi_exp=DEFAULTS.PI_EXP, save_flows=True)