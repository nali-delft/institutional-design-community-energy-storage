"""
models/model2.py
================
Model 2 - Household behind-the-meter (BTM) storage + community sharing.

Stage 1 (MILP per household, solved with Gurobi): each household dispatches its
own BTM battery - charge only from own PV, discharge only to own load, no
simultaneous charge/discharge or grid import/export - to minimize its
pre-sharing expenditure. Stage 2: residual surplus/demand are shared
proportionally within the community. The physical dispatch depends on
pi_imp/pi_exp but NOT on pi_loc, which is applied only in ex-post settlement.

Reads : input_data/inputs_filled.csv; retail prices via energy_prices.py
Writes: outputs/model2_*_<mode>_<tariff>.csv
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

sys.path.append(str(Path("..").resolve()))

from config import PATHS, DEFAULTS, PriceMode
from tariffs import ARCHETYPES, NetworkTariff
from tariff_milp import add_network_tariff
from tariffs import HOUSEHOLD_BAND_KW
from network_settlement import settle_network
from energy_prices import load_energy_price
from utils import (
    read_inputs_filled,
    build_con_gen_matrices,
    build_loc_price_grid,
    ensure_output_dir,
    save_kpi_and_summary,
)


# =========================================================
# Stage 1 (MILP): Household behind-the-meter storage dispatch
# =========================================================
@dataclass(frozen=True)
class BTMStorageParams:
    pi_exp: float
    E: float
    P: float
    eta_ch: float
    eta_dis: float
    soc0: float
    cyclic: bool = True


def solve_household_btm_storage_gurobi(
    load: np.ndarray,            # (T,)
    pv: np.ndarray,              # (T,)
    pi_imp_t: np.ndarray,        # (T,)
    params: BTMStorageParams,
    solver_verbose: bool = False,
    tariff: NetworkTariff | None = None,
    time_index: pd.DatetimeIndex | None = None,
) -> Dict[str, np.ndarray]:
    """
    Stage 1 MILP for one household with behind-the-meter battery.

    Logic:
      - battery can only charge from on-site PV
      - battery can only discharge to serve own load
      - no simultaneous charge/discharge
      - no simultaneous grid import/export
      - objective: minimize own pre-sharing electricity expenditure
    """
    load = np.asarray(load, dtype=float)
    pv = np.asarray(pv, dtype=float)
    pi_imp_t = np.asarray(pi_imp_t, dtype=float)

    T = int(load.size)
    if pv.size != T or pi_imp_t.size != T:
        raise ValueError("load, pv, pi_imp_t must have same length T.")
    if T == 0:
        raise ValueError("Empty time series.")
    if np.any(load < -1e-12) or np.any(pv < -1e-12):
        raise ValueError("load and pv must be non-negative.")

    E = float(params.E)
    P = float(params.P)
    eta_ch = float(params.eta_ch)
    eta_dis = float(params.eta_dis)
    pi_exp = float(params.pi_exp)

    if E < 0 or P < 0:
        raise ValueError("E and P must be non-negative.")
    if not (0 < eta_ch <= 1.0 and 0 < eta_dis <= 1.0):
        raise ValueError("eta_ch and eta_dis must be in (0, 1].")

    soc0 = float(np.clip(params.soc0, 0.0, E))

    # Trivial no-battery case
    if E <= 1e-12 or P <= 1e-12:
        ch = np.zeros(T)
        dis = np.zeros(T)
        imp = np.maximum(load - pv, 0.0)
        exp = np.maximum(pv - load, 0.0)
        soc = np.full(T + 1, soc0)
        return {"ch": ch, "dis": dis, "imp": imp, "exp": exp, "soc": soc}
    
    M_grid = float(max(np.max(load), np.max(pv), P))

    model = gp.Model("BTMStorageMILP")
    model.Params.OutputFlag = 1 if solver_verbose else 0
    model.Params.WorkLimit = 250.0

    ch = model.addVars(T, lb=0.0, ub=P, vtype=GRB.CONTINUOUS, name="ch")
    dis = model.addVars(T, lb=0.0, ub=P, vtype=GRB.CONTINUOUS, name="dis")
    imp = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="imp")
    exp = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="exp")
    soc = model.addVars(T + 1, lb=0.0, ub=E, vtype=GRB.CONTINUOUS, name="soc")

    y_ch = model.addVars(T, vtype=GRB.BINARY, name="y_ch")
    y_dis = model.addVars(T, vtype=GRB.BINARY, name="y_dis")
    y_grid = model.addVars(T, vtype=GRB.BINARY, name="y_grid")  # 1=import, 0=export

    model.addConstr(soc[0] == soc0, name="soc0")
    if params.cyclic:
        model.addConstr(soc[T] == soc0, name="soc_terminal")

    for t in range(T):
        model.addConstr(
            soc[t + 1] == soc[t] + eta_ch * ch[t] - dis[t] / eta_dis,
            name=f"soc_dyn[{t}]",
        )

        model.addConstr(ch[t] <= P * y_ch[t], name=f"ch_mode[{t}]")
        model.addConstr(dis[t] <= P * y_dis[t], name=f"dis_mode[{t}]")
        model.addConstr(y_ch[t] + y_dis[t] <= 1, name=f"bat_mode[{t}]")

        model.addConstr(imp[t] <= M_grid * y_grid[t], name=f"imp_mode[{t}]")
        model.addConstr(exp[t] <= M_grid * (1 - y_grid[t]), name=f"exp_mode[{t}]")

        model.addConstr(imp[t] <= M_grid * (1 - y_ch[t]), name=f"no_imp_when_ch[{t}]")
        model.addConstr(exp[t] <= M_grid * (1 - y_dis[t]), name=f"no_exp_when_dis[{t}]")

        model.addConstr(
            pv[t] + dis[t] + imp[t] == load[t] + ch[t] + exp[t],
            name=f"balance[{t}]",
        )

    obj = gp.quicksum(pi_imp_t[t] * imp[t] - pi_exp * exp[t] for t in range(T))

    if tariff is not None and tariff.name != "T0":
        if time_index is None:
            raise ValueError("time_index is required when a network tariff is applied.")
        # Each household is its own connection point; the battery sits behind
        # that meter, so the charge applies to the household's net import.
        # The battery sits behind this household's meter, so the household's own
        # net import is both the metered withdrawal and the decision variable.
        # Clause 7 does not apply: it addresses a storage facility serving end
        # consumers over the distribution grid, not storage behind a consumer's
        # own connection.
        net_term, _ = add_network_tariff(
            model, imp, time_index, tariff,
            x_inject=exp, fixed_band_kw=HOUSEHOLD_BAND_KW,
        )
        obj += net_term

    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    # A capacity charge assessed on the storage's own charging admits very many
    # schedules with identical monthly peaks, so the incumbent converges within
    # seconds while the bound crawls: on the hardest instance the objective moves
    # by 0.39 CHF out of 3492 (0.011%) between 75 s and 150 s of solving. We
    # therefore cap the search with WorkLimit, Gurobi's DETERMINISTIC analogue of
    # a time limit, so that results remain reproducible across machines, and
    # accept the incumbent with its achieved gap recorded. See S8.4.
    _ok = (GRB.OPTIMAL, GRB.WORK_LIMIT, GRB.TIME_LIMIT, GRB.SUBOPTIMAL)
    if model.Status not in _ok or model.SolCount == 0:
        raise RuntimeError(f"Gurobi returned no usable solution. Status={model.Status}")
    if model.Status != GRB.OPTIMAL:
        print(f"      [work limit reached: accepting incumbent, MIP gap "
              f"{model.MIPGap*100:.3f}%]")

    ch_arr = np.array([ch[t].X for t in range(T)], dtype=float)
    dis_arr = np.array([dis[t].X for t in range(T)], dtype=float)
    imp_arr = np.array([imp[t].X for t in range(T)], dtype=float)
    exp_arr = np.array([exp[t].X for t in range(T)], dtype=float)
    soc_arr = np.array([soc[t].X for t in range(T + 1)], dtype=float)

    return {
        "ch": np.clip(ch_arr, 0.0, None),
        "dis": np.clip(dis_arr, 0.0, None),
        "imp": np.clip(imp_arr, 0.0, None),
        "exp": np.clip(exp_arr, 0.0, None),
        "soc": np.clip(soc_arr, 0.0, E),
    }


# =========================================================
# Stage 2: Community sharing (physical allocation, independent of pi_loc)
# =========================================================
def community_sharing_proportional(
    surplus: pd.DataFrame,
    demand: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    P_sur = surplus.sum(axis=1)
    D_res = demand.sum(axis=1)
    x_loc = np.minimum(P_sur, D_res)

    share_buy = demand.div(D_res.replace(0.0, np.nan), axis=0).fillna(0.0)
    share_sell = surplus.div(P_sur.replace(0.0, np.nan), axis=0).fillna(0.0)

    x_buy = share_buy.mul(x_loc, axis=0).clip(lower=0.0)
    x_sell = share_sell.mul(x_loc, axis=0).clip(lower=0.0)

    community = pd.DataFrame({"P_sur": P_sur, "D_res": D_res, "x_loc": x_loc})
    return x_buy, x_sell, community


# =========================================================
# Physical model (independent of pi_loc)
# =========================================================
def run_model2_physical(
    con: pd.DataFrame,
    gen: pd.DataFrame,
    pi_imp_t: pd.Series,
    pi_exp: float = DEFAULTS.PI_EXP,
    eta_ch: float = DEFAULTS.ETA_CH,
    eta_dis: float = DEFAULTS.ETA_DIS,
    ces_power_hours: float = DEFAULTS.CES_POWER_HOURS,
    soc0_frac: float = DEFAULTS.SOC0_FRAC,
    tariff: NetworkTariff | None = None,
    cyclic_soc: bool = True,
    solver_verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Model 2 physical dispatch:
      - optimization depends on pi_imp_t and pi_exp
      - does NOT depend on pi_loc
      - pi_loc will only be used later for ex-post settlement
    """
    if ces_power_hours <= 0:
        raise ValueError("ces_power_hours must be > 0.")

    gen, con = gen.align(con, join="inner", axis=0)
    gen, con = gen.align(con, join="inner", axis=1)

    pi_imp_t = pi_imp_t.reindex(con.index).astype(float)
    if pi_imp_t.isna().any():
        raise ValueError("pi_imp_t has NaN after reindex; check time alignment.")

    if con.isna().any().any() or gen.isna().any().any():
        raise ValueError("con/gen contains NaN.")
    if (con < 0).any().any() or (gen < 0).any().any():
        raise ValueError("con/gen contains negative values.")

    cols = con.columns
    pi_imp_arr = pi_imp_t.to_numpy(dtype=float)

    annual_pv = gen.sum(axis=0).astype(float)
    annual_load = con.sum(axis=0).astype(float)
    has_pv = annual_pv > 1e-12
    has_batt = has_pv.copy()

    # Battery sizing
    E_series = (annual_load / 365.0).astype(float)
    P_series = (E_series / float(ces_power_hours)).astype(float)

    # Stage 1 outputs
    x_ch = pd.DataFrame(0.0, index=con.index, columns=cols)
    x_dis = pd.DataFrame(0.0, index=con.index, columns=cols)
    x_imp1 = pd.DataFrame(0.0, index=con.index, columns=cols)
    x_exp1 = pd.DataFrame(0.0, index=con.index, columns=cols)
    soc = pd.DataFrame(0.0, index=con.index, columns=cols)

    for i in cols:
        load_i = con[i].to_numpy(dtype=float)
        pv_i = gen[i].to_numpy(dtype=float)

        if bool(has_batt[i]) and float(E_series[i]) > 1e-12 and float(P_series[i]) > 1e-12:
            params = BTMStorageParams(
                pi_exp=float(pi_exp),
                E=float(E_series[i]),
                P=float(P_series[i]),
                eta_ch=float(eta_ch),
                eta_dis=float(eta_dis),
                soc0=float(soc0_frac) * float(E_series[i]),
                cyclic=bool(cyclic_soc),
            )
            sol = solve_household_btm_storage_gurobi(
                load=load_i,
                pv=pv_i,
                pi_imp_t=pi_imp_arr,
                params=params,
                solver_verbose=solver_verbose,
                tariff=tariff,
                time_index=con.index,
            )
            x_ch[i] = sol["ch"]
            x_dis[i] = sol["dis"]
            x_imp1[i] = sol["imp"]
            x_exp1[i] = sol["exp"]
            soc[i] = sol["soc"][1:]  # drop initial soc
        else:
            x_imp1[i] = np.maximum(load_i - pv_i, 0.0)
            x_exp1[i] = np.maximum(pv_i - load_i, 0.0)

    # Stage 2 physical sharing
    x_buy, x_sell, community = community_sharing_proportional(surplus=x_exp1, demand=x_imp1)

    x_imp = (x_imp1 - x_buy).clip(lower=0.0)
    x_exp = (x_exp1 - x_sell).clip(lower=0.0)

    direct_self = np.minimum(con, gen).sum(axis=0)

    kpi = pd.DataFrame(index=cols)
    kpi["annual_load"] = annual_load
    kpi["annual_pv"] = annual_pv
    kpi["has_pv"] = has_pv.astype(int)
    kpi["has_batt"] = has_batt.astype(int)
    kpi["battery_E_kwh"] = E_series.where(has_batt, 0.0)
    kpi["battery_P_kw"] = P_series.where(has_batt, 0.0)
    kpi["battery_charge"] = x_ch.sum(axis=0)
    kpi["battery_discharge"] = x_dis.sum(axis=0)
    kpi["direct_self_consumption"] = direct_self
    kpi["self_consumption"] = direct_self + x_ch.sum(axis=0)
    kpi["import_pre_share"] = x_imp1.sum(axis=0)
    kpi["export_pre_share"] = x_exp1.sum(axis=0)
    kpi["import"] = x_imp.sum(axis=0)
    kpi["export"] = x_exp.sum(axis=0)
    kpi["local_buy"] = x_buy.sum(axis=0)
    kpi["local_sell"] = x_sell.sum(axis=0)

    kpi["self_consumption_ratio"] = ((kpi["self_consumption"] + kpi["local_sell"]) / kpi["annual_pv"].replace(0, np.nan))
    kpi["self_sufficiency_ratio"] = ((kpi["self_consumption"] + kpi["local_sell"]) / kpi["annual_load"].replace(0, np.nan))
    
    physical = {
        "x_buy": x_buy,
        "x_sell": x_sell,
        "x_imp": x_imp,
        "x_exp": x_exp,
        "x_imp1": x_imp1,
        "x_exp1": x_exp1,
        "x_ch": x_ch,
        "x_dis": x_dis,
        "soc": soc,
        "community": community,
        "pi_imp_t": pi_imp_t,
        "pi_exp": float(pi_exp),
    }
    return kpi, physical


# =========================================================
# Ex-post settlement (depends on pi_loc only)
# =========================================================
def build_model2_settlement(
    physical_kpi: pd.DataFrame,
    physical: Dict[str, Any],
    pi_loc: float,
    pi_imp_t: pd.Series,
    pi_exp: float,
    tariff: NetworkTariff | None = None,
    delta: float = DEFAULTS.DELTA_LEG,
) -> pd.DataFrame:
    """
    Build ex-post cost allocation for one pi_loc.
    Physical results are fixed; only cost allocation changes with pi_loc.
    """
    x_imp = physical["x_imp"]
    x_exp = physical["x_exp"]
    x_buy = physical["x_buy"]
    x_sell = physical["x_sell"]

    cost_ts = (
        x_imp.mul(pi_imp_t, axis=0)
        + x_buy * float(pi_loc)
        - x_sell * float(pi_loc)
        - x_exp * float(pi_exp)
    )
    cost = cost_ts.sum(axis=0)

    # Network charges and the LEG reduction. Each household is its own
    # connection point with the battery behind that meter, so the charge falls
    # on the household's net import. Eligible LEG energy is direct local
    # sharing only: individually owned storage is not community storage, so the
    # storage cap in StromVV Art. 19h para. 4 does not arise here.
    net_household = pd.Series(0.0, index=physical_kpi.index)
    net = None
    if tariff is not None and tariff.name != "T0":
        net = settle_network(
            x_draw=physical["x_imp1"],
            e_leg_direct=x_buy.sum(axis=0),
            x_dis_load_alloc=None,
            tariff=tariff,
            delta=delta,
        )
        net_household = net.household_cost.reindex(physical_kpi.index).fillna(0.0)

    out = physical_kpi.copy()
    out["network_cost"] = net_household
    out["cost"] = cost + net_household
    out["pi_loc"] = float(pi_loc)
    out["tariff"] = tariff.name if tariff is not None else "T0"
    out["delta_leg"] = float(delta) if tariff is not None else 0.0
    if net is not None:
        out["network_gross"] = net.household_gross.reindex(physical_kpi.index).fillna(0.0)
        out["network_reduction"] = net.household_reduction.reindex(physical_kpi.index).fillna(0.0)
    return out


# =========================================================
# Unified runner
# =========================================================
def run_model2_for_price_mode(
    mode: PriceMode,
    pi_exp: float = DEFAULTS.PI_EXP,
    loc_step: float = DEFAULTS.LOC_STEP,
    cyclic_soc: bool = True,
    solver_verbose: bool = False,
    tariff: str = DEFAULTS.TARIFF,
    delta: float = DEFAULTS.DELTA_LEG,
) -> pd.DataFrame:
    """
    New logic:
      1. solve physical model only once for the chosen price mode
      2. save physical KPI once
      3. evaluate ex-post allocation over all pi_loc
      4. save pi_loc impact in ONE csv
    """
    out = ensure_output_dir(PATHS.OUTPUT_DIR)

    df, unit_ids = read_inputs_filled(PATHS.INPUTS_FILLED)
    con, gen = build_con_gen_matrices(df, unit_ids)

    tar = ARCHETYPES[tariff]
    suffix = f"{mode}_{tar.name}"
    pi_imp_t = load_energy_price(mode=mode, index=con.index, load=con.sum(axis=1))

    # ---------- 1) physical optimization only once ----------
    physical_kpi, physical = run_model2_physical(
        con=con,
        gen=gen,
        pi_imp_t=pi_imp_t,
        pi_exp=pi_exp,
        eta_ch=DEFAULTS.ETA_CH,
        eta_dis=DEFAULTS.ETA_DIS,
        ces_power_hours=DEFAULTS.CES_POWER_HOURS,
        soc0_frac=DEFAULTS.SOC0_FRAC,
        cyclic_soc=cyclic_soc,
        solver_verbose=solver_verbose,
        tariff=tar,
    )

    # Save physical KPI once
    physical_kpi.to_csv(out / f"model2_physical_kpis_{suffix}.csv", float_format="%.6f")

    physical_summary = pd.DataFrame([{
        "model": "model2",
        "mode": mode,
        "pi_exp": float(pi_exp),
        "pi_imp_mean": float(pi_imp_t.mean()),
        "total_batt_charge": float(physical_kpi["battery_charge"].sum()),
        "total_batt_discharge": float(physical_kpi["battery_discharge"].sum()),
        "total_import_pre_share": float(physical_kpi["import_pre_share"].sum()),
        "total_export_pre_share": float(physical_kpi["export_pre_share"].sum()),
        "total_import_final": float(physical_kpi["import"].sum()),
        "total_export_final": float(physical_kpi["export"].sum()),
        "total_local_exchange": float(physical_kpi["local_buy"].sum()),
        "total_self_consumption": float(physical_kpi["self_consumption"].sum()),
    }])
    physical_summary.to_csv(out / f"model2_physical_summary_{suffix}.csv", index=False, float_format="%.6f")
   
    flow_vars = ["x_ch", "x_dis", "x_imp1", "x_exp1", "x_buy", "x_sell", "x_imp", "x_exp", "soc"]

    flows_wide = None
    for var in flow_vars:
        tmp = physical[var].copy()
        tmp["time"] = tmp.index
        tmp = tmp.melt(id_vars="time", var_name="household", value_name=var)

        if flows_wide is None:
            flows_wide = tmp
        else:
            flows_wide = flows_wide.merge(tmp, on=["time", "household"], how="outer")

    flows_wide = flows_wide.sort_values(["household", "time"]).reset_index(drop=True)
    flows_wide.to_csv(out / f"model2_opt_flows_{suffix}.csv", index=False, float_format="%.6f")
    
    # ---------- 2) ex-post allocation across pi_loc ----------
    loc_prices = build_loc_price_grid(pi_exp=pi_exp, pi_imp_t=pi_imp_t, step=loc_step)

    allocation_rows = []
    summary_rows = []

    for ploc in loc_prices:
        settlement_kpi = build_model2_settlement(
            physical_kpi=physical_kpi,
            physical=physical,
            pi_loc=float(ploc),
            pi_imp_t=pi_imp_t,
            pi_exp=float(pi_exp),
            tariff=tar,
            delta=delta,
        )

        costs = settlement_kpi["cost"].to_numpy(dtype=float)

        summary_rows.append({
            "model": "model2",
            "mode": mode,
            "pi_loc": float(ploc),
            "pi_exp": float(pi_exp),
            "pi_imp_mean": float(pi_imp_t.mean()),
            "total_cost": float(settlement_kpi["cost"].sum()),
            "mean_cost": float(settlement_kpi["cost"].mean()),
            "std_cost": float(np.std(costs)),
            "min_cost": float(np.min(costs)),
            "max_cost": float(np.max(costs)),
            "tariff": tar.name,
            "delta_leg": float(delta),
            "total_network_cost": float(settlement_kpi["network_cost"].sum()),
            "total_network_reduction": float(settlement_kpi["network_reduction"].sum())
                if "network_reduction" in settlement_kpi else 0.0,
        })

        tmp = settlement_kpi[[
            "pi_loc",
            "annual_load",
            "annual_pv",
            "has_pv",
            "has_batt",
            "battery_E_kwh",
            "battery_P_kw",
            "battery_charge",
            "battery_discharge",
            "direct_self_consumption",
            "self_consumption",
            "import",
            "export",
            "local_buy",
            "local_sell",
            "cost",
            "self_consumption_ratio",
            "self_sufficiency_ratio",
        ]].copy()
        tmp = tmp.reset_index().rename(columns={"index": "household"})
        allocation_rows.append(tmp)

    allocation_df = pd.concat(allocation_rows, axis=0, ignore_index=True)
    allocation_df.to_csv(out / f"model2_allocation_by_piLoc_{suffix}.csv", index=False, float_format="%.6f")

    summary = pd.DataFrame(summary_rows).sort_values(["mode", "pi_loc"])
    summary.to_csv(out / f"model2_summary_{suffix}.csv", index=False, float_format="%.6f")
    

    print(f"[OK] model2 finished for mode={mode} -> {out}")
    return summary


if __name__ == "__main__":
    for m in ["flat", "tou", "dynamic"]:
        run_model2_for_price_mode(
            mode=m,
            pi_exp=DEFAULTS.PI_EXP,
            loc_step=DEFAULTS.LOC_STEP,
            cyclic_soc=True,
            solver_verbose=False,
        )