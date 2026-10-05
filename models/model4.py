"""
models/model4.py
================
Model 4 - Third-party owned Community Energy Storage (CES).

Same centralized MILP dispatch and market-access designs (A/B/C) as Model 3,
but the storage is owned by an external third party that levies a per-kWh
storage service fee tau (DEFAULTS.PI_TAU, CHF/kWh). The physical dispatch is
independent of pi_loc / tau. The ex-post settlement also carries optional
fixed-fee / profit-share terms (fixed_fee_annual, fee_shares), but these
default to zero and are not used in the paper runs.

Market-access designs (shared with Model 3):
  A  behind-the-meter: no market access (no grid charging; export at pi_exp)
  B  partial, import-only access (grid charging allowed; export at pi_exp)
  C  full market access (export price = import price)

Reads : input_data/inputs_filled.csv; retail prices via energy_prices.py
Writes: outputs/model4_*_<mode>_<design>_<tariff>.csv
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

sys.path.append(str(Path("..").resolve()))

from config import PATHS, DEFAULTS, PriceMode, DesignABC
from tariffs import NetworkTariff, ARCHETYPES
from tariff_milp import add_network_tariff, realised_band
from network_settlement import settle_network, marginal_leg_rate
from energy_prices import load_energy_price
from utils import (
    read_inputs_filled,
    build_con_gen_matrices,
    build_loc_price_grid,
    ensure_output_dir,
    save_kpi_and_summary,
)


# =========================================================
# Third-party CES MILP
# =========================================================
@dataclass(frozen=True)
class ThirdPartyStorageParams:
    E: float
    P: float
    eta_ch: float
    eta_dis: float
    soc0: float
    design: str = "A"   # A, B, C
    cyclic: bool = True


def _normalize_fee_shares(
    households: pd.Index,
    fee_shares: Mapping[str, float] | pd.Series | None,
) -> pd.Series:
    """Normalize fixed-fee shares to sum to 1 over all households."""
    idx = pd.Index(households, dtype=object)

    if fee_shares is None:
        return pd.Series(1.0 / len(idx), index=idx, dtype=float)

    if isinstance(fee_shares, pd.Series):
        a = fee_shares.reindex(idx).fillna(0.0).astype(float)
    else:
        a = pd.Series({str(k): float(v) for k, v in fee_shares.items()}, dtype=float)
        a = a.reindex(idx).fillna(0.0).astype(float)

    if (a < -1e-12).any():
        raise ValueError("fee_shares must be non-negative.")

    total = float(a.sum())
    if total <= 1e-12:
        raise ValueError("fee_shares must sum to a positive value.")

    return a / total


def solve_third_party_storage_gurobi(
    P_sur_t: np.ndarray,
    D_res_t: np.ndarray,
    pi_imp_t: np.ndarray,
    pi_exp_t: np.ndarray,
    params: ThirdPartyStorageParams,
    solver_verbose: bool = False,
    tariff: NetworkTariff | None = None,
    time_index: pd.DatetimeIndex | None = None,
    leg_rate: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Centralized third-party CES dispatch for Model 4.

    Design settings:
      A: behind-the-meter coordination
         - no import while charging
         - no export while discharging
         - export price fixed at PI_EXP
      B: market-enabled procurement
         - charging from grid allowed
         - no export while discharging
         - export price fixed at PI_EXP
      C: fully market-integrated
         - no additional CES-grid interaction constraint
         - export price equals import price
    """
    P_sur_t = np.asarray(P_sur_t, dtype=float)
    D_res_t = np.asarray(D_res_t, dtype=float)
    pi_imp_t = np.asarray(pi_imp_t, dtype=float)
    pi_exp_t = np.asarray(pi_exp_t, dtype=float)

    T = int(P_sur_t.size)
    if D_res_t.size != T or pi_imp_t.size != T or pi_exp_t.size != T:
        raise ValueError("P_sur_t, D_res_t, pi_imp_t, and pi_exp_t must have the same length.")
    if T == 0:
        raise ValueError("Empty time series.")
    if np.any(P_sur_t < -1e-12) or np.any(D_res_t < -1e-12):
        raise ValueError("P_sur_t and D_res_t must be non-negative.")

    E = float(params.E)
    P = float(params.P)
    eta_ch = float(params.eta_ch)
    eta_dis = float(params.eta_dis)
    design = str(params.design).upper()

    if design not in {"A", "B", "C"}:
        raise ValueError("design must be one of: 'A', 'B', 'C'.")
    if E < 0 or P < 0:
        raise ValueError("E and P must be non-negative.")
    if not (0 < eta_ch <= 1.0 and 0 < eta_dis <= 1.0):
        raise ValueError("eta_ch and eta_dis must be in (0, 1].")

    soc0 = float(np.clip(params.soc0, 0.0, E))

    if E <= 1e-12 or P <= 1e-12:
        imp = np.maximum(D_res_t - P_sur_t, 0.0)
        exp = np.maximum(P_sur_t - D_res_t, 0.0)
        ch = np.zeros(T)
        dis = np.zeros(T)
        soc = np.full(T + 1, soc0)
        return {
            "x_ch": ch,
            "x_dis": dis,
            "x_imp": imp,
            "x_exp": exp,
            "soc": soc,
            "obj_val": float(np.dot(pi_imp_t, imp) - np.dot(pi_exp_t, exp)),
        }

    M_grid = float(max(np.max(P_sur_t), np.max(D_res_t), P))
    M_grid = max(M_grid, 1e-6)

    model = gp.Model("ThirdPartyStorageMILP")
    model.Params.OutputFlag = 1 if solver_verbose else 0
    model.Params.MIPGap = 1e-3
    model.Params.WorkLimit = 250.0

    x_ch = model.addVars(T, lb=0.0, ub=P, vtype=GRB.CONTINUOUS, name="x_ch")
    x_dis = model.addVars(T, lb=0.0, ub=P, vtype=GRB.CONTINUOUS, name="x_dis")
    x_imp = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="x_imp")
    x_exp = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="x_exp")
    soc = model.addVars(T + 1, lb=0.0, ub=E, vtype=GRB.CONTINUOUS, name="soc")

    y_ch = model.addVars(T, vtype=GRB.BINARY, name="y_ch")
    y_dis = model.addVars(T, vtype=GRB.BINARY, name="y_dis")
    y_grid = model.addVars(T, vtype=GRB.BINARY, name="y_grid")

    model.addConstr(soc[0] == soc0, name="soc0")
    if params.cyclic:
        model.addConstr(soc[T] == soc0, name="soc_terminal")

    for t in range(T):
        model.addConstr(
            soc[t + 1] == soc[t] + eta_ch * x_ch[t] - x_dis[t] / eta_dis,
            name=f"soc_dyn[{t}]",
        )

        model.addConstr(x_ch[t] <= P * y_ch[t], name=f"ch_mode[{t}]")
        model.addConstr(x_dis[t] <= P * y_dis[t], name=f"dis_mode[{t}]")
        model.addConstr(y_ch[t] + y_dis[t] <= 1, name=f"bat_mode[{t}]")

        model.addConstr(x_imp[t] <= M_grid * y_grid[t], name=f"imp_mode[{t}]")
        model.addConstr(x_exp[t] <= M_grid * (1 - y_grid[t]), name=f"exp_mode[{t}]")

        model.addConstr(
            P_sur_t[t] + x_imp[t] + x_dis[t] == D_res_t[t] + x_ch[t] + x_exp[t],
            name=f"balance[{t}]",
        )

        if design == "A":
            model.addConstr(x_imp[t] <= M_grid * (1 - y_ch[t]), name=f"A_no_imp_when_ch[{t}]")
            model.addConstr(x_exp[t] <= M_grid * (1 - y_dis[t]), name=f"A_no_exp_when_dis[{t}]")
        elif design == "B":
            model.addConstr(x_exp[t] <= M_grid * (1 - y_dis[t]), name=f"B_no_exp_when_dis[{t}]")
        # design C: no additional constraints

    obj = gp.quicksum(pi_imp_t[t] * x_imp[t] - pi_exp_t[t] * x_exp[t] for t in range(T))

    # ---- statutory network-fee reduction, valued in dispatch ---------------
    # Electricity the storage supplies to community load earns a reduction of
    # the network tariff (StromVV Art. 19h), worth `leg_rate` CHF per kWh. The
    # rate is constant because the reduction's base and denominator are both
    # exogenous, so the term is exactly linear. Article 19h paragraph 4 caps the
    # eligible amount at what the storage drew FROM the community, which is a
    # minimum of two linear expressions; since the term enters the objective as
    # a benefit, two inequalities represent it exactly and no binary is needed.
    # The split variables are pure accounting: each is bounded by the physical
    # flow and by the exogenous room available, so neither constrains dispatch.
    if leg_rate > 0.0:
        P_excess = np.maximum(P_sur_t - D_res_t, 0.0)
        D_gap = np.maximum(D_res_t - P_sur_t, 0.0)
        ch_loc = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="leg_ch_local")
        dis_load = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="leg_dis_load")
        for t in range(T):
            model.addConstr(ch_loc[t] <= x_ch[t], name=f"leg_chloc_a[{t}]")
            model.addConstr(ch_loc[t] <= P_excess[t], name=f"leg_chloc_b[{t}]")
            model.addConstr(dis_load[t] <= x_dis[t], name=f"leg_disload_a[{t}]")
            model.addConstr(dis_load[t] <= D_gap[t], name=f"leg_disload_b[{t}]")
        z = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="leg_eligible")
        model.addConstr(z <= gp.quicksum(dis_load[t] for t in range(T)), name="leg_cap_dis")
        model.addConstr(z <= gp.quicksum(ch_loc[t] for t in range(T)), name="leg_cap_ch")
        obj -= leg_rate * z


    net_aux: Dict[str, Any] = {}
    if tariff is not None and tariff.name != "T0":
        if time_index is None:
            raise ValueError("time_index is required when a network tariff is applied.")
        # See model3: the endogenous withdrawal is the storage's own charging.
        net_term, net_aux = add_network_tariff(
            model, x_ch, time_index, tariff, x_refund=x_dis,
            x_inject=x_dis, choose_band=True,
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

    out = {
        "x_ch": np.array([x_ch[t].X for t in range(T)], dtype=float),
        "x_dis": np.array([x_dis[t].X for t in range(T)], dtype=float),
        "x_imp": np.array([x_imp[t].X for t in range(T)], dtype=float),
        "x_exp": np.array([x_exp[t].X for t in range(T)], dtype=float),
        "soc": np.array([soc[t].X for t in range(T + 1)], dtype=float),
        "obj_val": float(model.ObjVal),
    }
    if net_aux:
        rb = realised_band(net_aux)
        if rb is not None:
            out["connection_band"], out["contracted_kw"] = rb
    return out


# =========================================================
# Helpers
# =========================================================
def _allocate_proportionally(base: pd.DataFrame, total: pd.Series) -> pd.DataFrame:
    denom = base.sum(axis=1).replace(0.0, np.nan)
    share = base.div(denom, axis=0).fillna(0.0)
    return share.mul(total, axis=0).clip(lower=0.0)


def _decompose_community_flows(community: pd.DataFrame) -> pd.DataFrame:
    """
    Ex-post decomposition for Model 4, consistent with the paper:

      x_pv_loc  = min(D_res, P_sur)
      D_gap     = D_res - x_pv_loc
      P_excess  = P_sur - x_pv_loc

      x_ch_pv   = min(x_ch, P_excess)
      x_imp_bat = x_ch - x_ch_pv
      x_imp_load = x_imp - x_imp_bat

      x_dis_load = min(x_dis, D_gap)
      x_exp_bat  = x_dis - x_dis_load
      x_exp_pv   = x_exp - x_exp_bat
    """
    out = pd.DataFrame(index=community.index)

    P_sur = community["P_sur"].astype(float)
    D_res = community["D_res"].astype(float)
    x_ch = community["x_ch"].astype(float)
    x_dis = community["x_dis"].astype(float)
    x_imp = community["x_imp"].astype(float)
    x_exp = community["x_exp"].astype(float)

    x_pv_loc = np.minimum(D_res, P_sur)

    D_gap = (D_res - x_pv_loc).clip(lower=0.0)
    P_excess = (P_sur - x_pv_loc).clip(lower=0.0)

    x_ch_pv = np.minimum(x_ch, P_excess)
    x_imp_bat = (x_ch - x_ch_pv).clip(lower=0.0)
    x_imp_load = (x_imp - x_imp_bat).clip(lower=0.0)

    x_dis_load = np.minimum(x_dis, D_gap)
    x_exp_bat = (x_dis - x_dis_load).clip(lower=0.0)
    x_exp_pv = (x_exp - x_exp_bat).clip(lower=0.0)

    out["x_pv_loc"] = x_pv_loc
    out["D_gap"] = D_gap
    out["P_excess"] = P_excess

    out["x_ch_pv"] = x_ch_pv
    out["x_imp_bat"] = x_imp_bat
    out["x_imp_load"] = x_imp_load

    out["x_dis_load"] = x_dis_load
    out["x_exp_bat"] = x_exp_bat
    out["x_exp_pv"] = x_exp_pv

    return out


# =========================================================
# Physical model (independent of pi_loc, tau, fixed fee)
# =========================================================
def run_model4_physical(
    con: pd.DataFrame,
    gen: pd.DataFrame,
    pi_imp_t: pd.Series,
    design: DesignABC = "A",
    pi_exp: float = DEFAULTS.PI_EXP,
    eta_ch: float = DEFAULTS.ETA_CH,
    eta_dis: float = DEFAULTS.ETA_DIS,
    ces_power_hours: float = DEFAULTS.CES_POWER_HOURS,
    soc0_frac: float = DEFAULTS.SOC0_FRAC,
    cyclic_soc: bool = True,
    ces_energy_kwh: float | None = None,
    solver_verbose: bool = False,
    tariff: NetworkTariff | None = None,
    delta: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Physical dispatch of Model 4.

    Important:
      - physical optimization depends on mode/design
      - it does NOT depend on pi_loc, tau, or fixed fee
      - those will only be used later for ex-post settlement
    """
    if ces_power_hours <= 0:
        raise ValueError("ces_power_hours must be > 0.")

    gen, con = gen.align(con, join="inner", axis=0)
    gen, con = gen.align(con, join="inner", axis=1)

    pi_imp_t = pi_imp_t.reindex(con.index).astype(float)
    if pi_imp_t.isna().any():
        raise ValueError("pi_imp_t has NaN after reindex; check time alignment.")

    design = str(design).upper()
    if design not in {"A", "B", "C"}:
        raise ValueError("design must be one of: 'A', 'B', 'C'.")

    if design in {"A", "B"}:
        pi_exp_t = pd.Series(float(pi_exp), index=con.index, name="pi_exp")
    else:
        pi_exp_t = pi_imp_t.copy().rename("pi_exp")

    cols = con.columns
    annual_load = con.sum(axis=0).astype(float)
    annual_pv = gen.sum(axis=0).astype(float)
    has_pv = (annual_pv > 1e-12).astype(int)

    x_self = np.minimum(con, gen)
    p_sur = (gen - x_self).clip(lower=0.0)
    d_res = (con - x_self).clip(lower=0.0)

    P_sur_t = p_sur.sum(axis=1)
    D_res_t = d_res.sum(axis=1)

    if ces_energy_kwh is None:
        E = float(annual_load.sum() / 365.0)
    else:
        E = float(ces_energy_kwh)
    P = float(E / ces_power_hours) if E > 0 else 0.0

    params = ThirdPartyStorageParams(
        E=E,
        P=P,
        eta_ch=float(eta_ch),
        eta_dis=float(eta_dis),
        soc0=float(soc0_frac) * E,
        design=design,
        cyclic=bool(cyclic_soc),
    )

    # Value of the statutory reduction per eligible kWh, from the exogenous
    # household withdrawals. See network_settlement.marginal_leg_rate.
    leg_rate = (marginal_leg_rate(d_res, tariff, float(delta))
                if tariff is not None and delta else 0.0)

    sol = solve_third_party_storage_gurobi(
        P_sur_t=P_sur_t.to_numpy(dtype=float),
        D_res_t=D_res_t.to_numpy(dtype=float),
        pi_imp_t=pi_imp_t.to_numpy(dtype=float),
        pi_exp_t=pi_exp_t.to_numpy(dtype=float),
        params=params,
        solver_verbose=solver_verbose,
        tariff=tariff,
        time_index=con.index,
        leg_rate=leg_rate,
    )

    community = pd.DataFrame(index=con.index)
    community["P_sur"] = P_sur_t
    community["D_res"] = D_res_t
    community["x_ch"] = sol["x_ch"]
    community["x_dis"] = sol["x_dis"]
    community["x_imp"] = sol["x_imp"]
    community["x_exp"] = sol["x_exp"]
    community["x_loc"] = np.minimum(P_sur_t, D_res_t)
    community["soc"] = sol["soc"][:-1]
    community["pi_imp"] = pi_imp_t
    community["pi_exp"] = pi_exp_t

    # ---------------------------------------------------------
    # Ex-post decomposition consistent with the paper
    # ---------------------------------------------------------
    community_decomp = _decompose_community_flows(community)
    community = pd.concat([community, community_decomp], axis=1)

    # Household-level allocations used for settlement
    x_buy_loc_alloc = _allocate_proportionally(d_res, community["x_pv_loc"])
    x_sell_loc_alloc = _allocate_proportionally(p_sur, community["x_pv_loc"])

    x_dis_load_alloc = _allocate_proportionally(d_res, community["x_dis_load"])
    x_imp_load_alloc = _allocate_proportionally(d_res, community["x_imp_load"])

    x_ch_pv_alloc = _allocate_proportionally(p_sur, community["x_ch_pv"])
    x_exp_pv_alloc = _allocate_proportionally(p_sur, community["x_exp_pv"])

    physical_kpi = pd.DataFrame(index=cols)
    physical_kpi["annual_load"] = annual_load
    physical_kpi["annual_pv"] = annual_pv
    physical_kpi["has_pv"] = has_pv
    physical_kpi["direct_self_consumption"] = x_self.sum(axis=0)
    physical_kpi["residual_demand"] = d_res.sum(axis=0)
    physical_kpi["residual_surplus"] = p_sur.sum(axis=0)

    physical_kpi["local_buy_alloc"] = x_buy_loc_alloc.sum(axis=0)
    physical_kpi["local_sell_alloc"] = x_sell_loc_alloc.sum(axis=0)
    physical_kpi["battery_discharge_alloc"] = x_dis_load_alloc.sum(axis=0)
    physical_kpi["grid_import_alloc"] = x_imp_load_alloc.sum(axis=0)
    physical_kpi["battery_charge_alloc"] = x_ch_pv_alloc.sum(axis=0)
    physical_kpi["grid_export_alloc"] = x_exp_pv_alloc.sum(axis=0)

    physical_kpi["self_consumption_ratio"] = (
        (physical_kpi["direct_self_consumption"]
         + physical_kpi["local_sell_alloc"]
         + physical_kpi["battery_charge_alloc"])
        / physical_kpi["annual_pv"].replace(0, np.nan)
    )

    physical_kpi["self_sufficiency_ratio"] = (
        (physical_kpi["direct_self_consumption"]
         + physical_kpi["local_buy_alloc"]
         + physical_kpi["battery_discharge_alloc"])
        / physical_kpi["annual_load"].replace(0, np.nan)
    )

    physical = {
        "x_self": x_self,
        "p_sur": p_sur,
        "d_res": d_res,
        "community": community,
        "soc_full": pd.Series(sol["soc"], name="soc"),
        "pi_imp_t": pi_imp_t,
        "pi_exp_t": pi_exp_t,
        "design": str(design),
        "battery_E_kwh": float(E),
        "battery_P_kw": float(P),
        "contracted_kw": sol.get("contracted_kw"),
        "connection_band": sol.get("connection_band"),
        "obj_val": sol["obj_val"],
        "tariff": tariff,
        # settlement-relevant household allocations
        "x_buy_loc_alloc": x_buy_loc_alloc,
        "x_sell_loc_alloc": x_sell_loc_alloc,
        "x_dis_load_alloc": x_dis_load_alloc,
        "x_imp_load_alloc": x_imp_load_alloc,
        "x_ch_pv_alloc": x_ch_pv_alloc,
        "x_exp_pv_alloc": x_exp_pv_alloc,
    }
    return physical_kpi, physical


# =========================================================
# Ex-post settlement (depends on pi_loc, tau, fixed fee only)
# =========================================================
def build_model4_settlement(
    physical_kpi: pd.DataFrame,
    physical: Dict[str, Any],
    pi_loc: float,
    tau: float = DEFAULTS.PI_TAU,
    fixed_fee_annual: float = 0.0,
    tariff: NetworkTariff | None = None,
    delta: float = DEFAULTS.DELTA_LEG,
    fee_shares: Mapping[str, float] | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Ex-post settlement for one pi_loc, consistent with the updated paper:

      C_i =
        pi_loc * x_buy_loc
        + (pi_loc + tau) * x_dis_load
        + pi_imp * x_imp_load
        - pi_loc * x_sell_loc
        - pi_loc * x_ch_pv
        - pi_exp * x_exp_pv
        + alpha_i * F_CES

      R_3rd =
        (pi_loc + tau) * x_dis_load
        - pi_loc * x_ch_pv
        + pi_exp * x_exp_bat
        - pi_imp * x_imp_bat
        + F_CES
    """
    if tau < -1e-12:
        raise ValueError("tau must be non-negative.")
    if fixed_fee_annual < -1e-12:
        raise ValueError("fixed_fee_annual must be non-negative.")

    cols = physical_kpi.index
    alpha = _normalize_fee_shares(cols, fee_shares)

    community = physical["community"]
    pi_imp_t = physical["pi_imp_t"]
    pi_exp_t = physical["pi_exp_t"]

    x_buy_loc_alloc = physical["x_buy_loc_alloc"]
    x_sell_loc_alloc = physical["x_sell_loc_alloc"]
    x_dis_load_alloc = physical["x_dis_load_alloc"]
    x_imp_load_alloc = physical["x_imp_load_alloc"]
    x_ch_pv_alloc = physical["x_ch_pv_alloc"]
    x_exp_pv_alloc = physical["x_exp_pv_alloc"]

    # 1) household cost components
    C_buy_loc_ts = x_buy_loc_alloc * float(pi_loc)
    C_dis_load_ts = x_dis_load_alloc * float(pi_loc + tau)
    C_imp_load_ts = x_imp_load_alloc.mul(pi_imp_t, axis=0)

    R_sell_loc_ts = x_sell_loc_alloc * float(pi_loc)
    R_ch_pv_ts = x_ch_pv_alloc * float(pi_loc)
    R_exp_pv_ts = x_exp_pv_alloc.mul(pi_exp_t, axis=0)

    energy_cost_ts = (
        C_buy_loc_ts
        + C_dis_load_ts
        + C_imp_load_ts
        - R_sell_loc_ts
        - R_ch_pv_ts
        - R_exp_pv_ts
    )

    # 1b) network charges. Each household is its own connection point. The
    #     storage is a further connection point, but under third-party ownership
    #     its network cost is borne by the OPERATOR, not the community -- the
    #     mirror of Model 3, where it enters the CES settlement account.
    net = None
    net_household = pd.Series(0.0, index=cols)
    ces_network_cost = 0.0
    if tariff is not None and tariff.name != "T0":
        net = settle_network(
            x_draw=physical["d_res"],
            e_leg_direct=x_buy_loc_alloc.sum(axis=0),
            x_dis_load_alloc=x_dis_load_alloc,
            tariff=tariff,
            delta=delta,
            ces_ch=community["x_ch"],
            ces_dis=community["x_dis"],
            ces_contracted_kw=physical.get("contracted_kw"),
            x_ch_local_total=float(community["x_ch_pv"].sum()),
        )
        net_household = net.household_cost.reindex(cols).fillna(0.0)
        ces_network_cost = net.ces_cost

    fixed_fee_alloc = alpha * float(fixed_fee_annual)
    cost = energy_cost_ts.sum(axis=0) + fixed_fee_alloc + net_household

    # 2) third-party revenue
    third_party_discharge_revenue = float(((float(pi_loc) + float(tau)) * community["x_dis_load"]).sum())
    third_party_charge_credit = float((float(pi_loc) * community["x_ch_pv"]).sum())
    third_party_export_revenue = float((community["x_exp_bat"] * pi_exp_t).sum())
    third_party_import_cost = float((community["x_imp_bat"] * pi_imp_t).sum())

    third_party_revenue_total = (
        third_party_discharge_revenue
        - third_party_charge_credit
        + third_party_export_revenue
        - third_party_import_cost
        + float(fixed_fee_annual)
        - ces_network_cost
    )

    market_margin = third_party_export_revenue - third_party_import_cost
    storage_service_revenue = third_party_discharge_revenue - third_party_charge_credit

    out = physical_kpi.copy()
    out["fee_share"] = alpha

    out["local_buy_cost"] = C_buy_loc_ts.sum(axis=0)
    out["battery_supply_cost"] = C_dis_load_ts.sum(axis=0)
    out["import_cost"] = C_imp_load_ts.sum(axis=0)

    out["local_sell_revenue"] = R_sell_loc_ts.sum(axis=0)
    out["battery_charge_credit"] = R_ch_pv_ts.sum(axis=0)
    out["grid_export_revenue"] = R_exp_pv_ts.sum(axis=0)

    out["energy_cost"] = energy_cost_ts.sum(axis=0)
    out["fixed_fee_alloc"] = fixed_fee_alloc
    out["cost"] = cost
    out["pi_loc"] = float(pi_loc)
    out["tau"] = float(tau)
    out["network_cost"] = net_household
    out["tariff"] = tariff.name if tariff is not None else "T0"
    out["delta_leg"] = float(delta) if tariff is not None else 0.0
    if net is not None:
        out["network_gross"] = net.household_gross.reindex(cols).fillna(0.0)
        out["network_reduction"] = net.household_reduction.reindex(cols).fillna(0.0)
        out["leg_eligible_ratio"] = net.eligible_ratio
        out["connection_band"] = physical.get("connection_band")
        out["contracted_kw"] = physical.get("contracted_kw")

    third_party_totals = {
        "third_party_network_cost": float(ces_network_cost),
        "connection_band": physical.get("connection_band"),
        "contracted_kw": physical.get("contracted_kw"),
        "third_party_discharge_revenue_total": third_party_discharge_revenue,
        "third_party_charge_credit_total": third_party_charge_credit,
        "third_party_export_revenue_total": third_party_export_revenue,
        "third_party_import_cost_total": third_party_import_cost,
        "third_party_revenue_total": float(third_party_revenue_total),
        "market_margin": float(market_margin),
        "storage_service_revenue": float(storage_service_revenue),
    }

    return out, third_party_totals


# =========================================================
# Unified runner
# =========================================================
def run_model4_for_price_mode(
    mode: PriceMode,
    design: DesignABC,
    pi_exp: float = DEFAULTS.PI_EXP,
    tau: float = DEFAULTS.PI_TAU,
    tariff: str = DEFAULTS.TARIFF,
    delta: float = DEFAULTS.DELTA_LEG,
    loc_step: float = DEFAULTS.LOC_STEP,
    cyclic_soc: bool = True,
    ces_energy_kwh: float | None = None,
    fixed_fee_annual: float = 0.0,
    fee_shares: Mapping[str, float] | pd.Series | None = None,
    solver_verbose: bool = False,
) -> pd.DataFrame:
    """
    New logic:
      1. solve Model 4 physical optimization only once for (mode, design)
      2. save physical KPI once
      3. evaluate ex-post settlement over all pi_loc
      4. save pi_loc impact in ONE csv
    """
    out = ensure_output_dir(PATHS.OUTPUT_DIR)

    df, unit_ids = read_inputs_filled(PATHS.INPUTS_FILLED)
    con, gen = build_con_gen_matrices(df, unit_ids)

    tar = ARCHETYPES[tariff]
    suffix = f"{mode}_{design}_{tar.name}"

    # Energy component only, in CHF excl. VAT; the network tariff is applied
    # separately through `tar` (Supplementary Information S8.3-S8.4).
    pi_imp_t = load_energy_price(mode=mode, index=con.index, load=con.sum(axis=1))

    # ---------- 1) physical optimization only once ----------
    physical_kpi, physical = run_model4_physical(
        con=con,
        gen=gen,
        pi_imp_t=pi_imp_t,
        design=design,
        pi_exp=pi_exp,
        eta_ch=DEFAULTS.ETA_CH,
        eta_dis=DEFAULTS.ETA_DIS,
        ces_power_hours=DEFAULTS.CES_POWER_HOURS,
        soc0_frac=DEFAULTS.SOC0_FRAC,
        cyclic_soc=cyclic_soc,
        ces_energy_kwh=ces_energy_kwh,
        solver_verbose=solver_verbose,
        tariff=tar,
        delta=float(delta),
    )

    physical["community"].to_csv(
        out / f"model4_opt_community_{suffix}.csv",
        float_format="%.6f"
    )

    physical_kpi.to_csv(
        out / f"model4_physical_kpis_{suffix}.csv",
        float_format="%.6f"
    )

    physical_summary = pd.DataFrame([{
        "model": "model4",
        "mode": mode,
        "design": design,
        "pi_exp_base": float(pi_exp),
        "pi_imp_mean": float(pi_imp_t.mean()),
        "community_battery_E_kwh": float(physical["battery_E_kwh"]),
        "community_battery_P_kw": float(physical["battery_P_kw"]),
        "tariff": tar.name,
        "obj_val": physical["obj_val"],
        "total_direct_self_consumption": float(physical_kpi["direct_self_consumption"].sum()),
        "total_residual_demand": float(physical_kpi["residual_demand"].sum()),
        "total_residual_surplus": float(physical_kpi["residual_surplus"].sum()),
        "total_local_buy_alloc": float(physical_kpi["local_buy_alloc"].sum()),
        "total_local_sell_alloc": float(physical_kpi["local_sell_alloc"].sum()),
        "total_grid_import_alloc": float(physical_kpi["grid_import_alloc"].sum()),
        "total_grid_export_alloc": float(physical_kpi["grid_export_alloc"].sum()),
        "total_battery_charge_alloc": float(physical_kpi["battery_charge_alloc"].sum()),
        "total_battery_discharge_alloc": float(physical_kpi["battery_discharge_alloc"].sum()),
    }])
    physical_summary.to_csv(
        out / f"model4_physical_summary_{suffix}.csv",
        index=False,
        float_format="%.6f"
    )

    flow_vars = [
        "x_buy_loc_alloc",
        "x_sell_loc_alloc",
        "x_dis_load_alloc",
        "x_imp_load_alloc",
        "x_ch_pv_alloc",
        "x_exp_pv_alloc",
    ]

    flows_wide = None
    for var in flow_vars:
        tmp = physical[var].copy()
        # tmp["time"] = tmp.index
        tmp = physical[var].copy().reset_index().rename(columns={physical[var].index.name or "index": "time"})
        tmp = tmp.melt(id_vars="time", var_name="household", value_name=var)

        if flows_wide is None:
            flows_wide = tmp
        else:
            flows_wide = flows_wide.merge(tmp, on=["time", "household"], how="outer")

    flows_wide = flows_wide.sort_values(["time", "household"]).reset_index(drop=True)

    flows_wide.to_csv(
        out / f"model4_opt_alloc_flows_{suffix}.csv",
        index=False,
        float_format="%.6f"
    )

    # ---------- 2) ex-post allocation across pi_loc ----------
    loc_prices = build_loc_price_grid(pi_exp=pi_exp, pi_imp_t=pi_imp_t, step=loc_step)

    allocation_rows = []
    summary_rows = []

    for ploc in loc_prices:
        settlement_kpi, third_party_totals = build_model4_settlement(
            physical_kpi=physical_kpi,
            physical=physical,
            pi_loc=float(ploc),
            tau=float(tau),
            fixed_fee_annual=float(fixed_fee_annual),
            fee_shares=fee_shares,
            tariff=tar,
            delta=delta,
        )

        costs = settlement_kpi["cost"].to_numpy(dtype=float)
        community = physical["community"]

        summary_rows.append({
            "model": "model4",
            "mode": mode,
            "design": design,
            "pi_loc": float(ploc),
            "tau": float(tau),
            "pi_exp_base": float(pi_exp),
            "fixed_fee_annual": float(fixed_fee_annual),
            "pi_imp_mean": float(pi_imp_t.mean()),

            "total_cost": float(settlement_kpi["cost"].sum()),
            "mean_cost": float(settlement_kpi["cost"].mean()),
            "std_cost": float(np.std(costs)),
            "total_energy_cost": float(settlement_kpi["energy_cost"].sum()),
            "total_fixed_fee_alloc": float(settlement_kpi["fixed_fee_alloc"].sum()),

            "total_local_buy_cost": float(settlement_kpi["local_buy_cost"].sum()),
            "total_battery_supply_cost": float(settlement_kpi["battery_supply_cost"].sum()),
            "total_import_cost": float(settlement_kpi["import_cost"].sum()),
            "total_local_sell_revenue": float(settlement_kpi["local_sell_revenue"].sum()),
            "total_battery_charge_credit": float(settlement_kpi["battery_charge_credit"].sum()),
            "total_grid_export_revenue": float(settlement_kpi["grid_export_revenue"].sum()),

            "total_local_buy_alloc": float(settlement_kpi["local_buy_alloc"].sum()),
            "total_local_sell_alloc": float(settlement_kpi["local_sell_alloc"].sum()),
            "total_battery_discharge_alloc": float(settlement_kpi["battery_discharge_alloc"].sum()),
            "total_grid_import_alloc": float(settlement_kpi["grid_import_alloc"].sum()),
            "total_battery_charge_alloc": float(settlement_kpi["battery_charge_alloc"].sum()),
            "total_grid_export_alloc": float(settlement_kpi["grid_export_alloc"].sum()),

            "total_x_pv_loc": float(community["x_pv_loc"].sum()),
            "total_x_imp_bat": float(community["x_imp_bat"].sum()),
            "total_x_imp_load": float(community["x_imp_load"].sum()),
            "total_x_dis_load": float(community["x_dis_load"].sum()),
            "total_x_exp_bat": float(community["x_exp_bat"].sum()),
            "total_x_exp_pv": float(community["x_exp_pv"].sum()),

            "community_battery_E_kwh": float(physical["battery_E_kwh"]),
            "community_battery_P_kw": float(physical["battery_P_kw"]),
            "tariff": tar.name,
            "delta_leg": float(delta),
            "total_network_cost": float(settlement_kpi["network_cost"].sum()),
            "third_party_network_cost": float(third_party_totals.get("third_party_network_cost", 0.0)),

            "third_party_discharge_revenue_total": third_party_totals["third_party_discharge_revenue_total"],
            "third_party_charge_credit_total": third_party_totals["third_party_charge_credit_total"],
            "third_party_export_revenue_total": third_party_totals["third_party_export_revenue_total"],
            "third_party_import_cost_total": third_party_totals["third_party_import_cost_total"],
            "third_party_revenue_total": third_party_totals["third_party_revenue_total"],
            "market_margin": third_party_totals["market_margin"],
            "storage_service_revenue": third_party_totals["storage_service_revenue"],
        })

        tmp = settlement_kpi[[
            "pi_loc",
            "tau",
            "annual_load",
            "annual_pv",
            "has_pv",
            "fee_share",
            "direct_self_consumption",
            "residual_demand",
            "residual_surplus",
            "local_buy_alloc",
            "local_sell_alloc",
            "battery_discharge_alloc",
            "grid_import_alloc",
            "battery_charge_alloc",
            "grid_export_alloc",
            "local_buy_cost",
            "battery_supply_cost",
            "import_cost",
            "local_sell_revenue",
            "battery_charge_credit",
            "grid_export_revenue",
            "energy_cost",
            "fixed_fee_alloc",
            "cost",
            "self_consumption_ratio",
            "self_sufficiency_ratio",
        ]].copy()

        tmp = tmp.reset_index().rename(columns={"index": "household"})
        allocation_rows.append(tmp)

    allocation_df = pd.concat(allocation_rows, axis=0, ignore_index=True)
    allocation_df.to_csv(
        out / f"model4_allocation_by_piLoc_{suffix}.csv",
        index=False,
        float_format="%.6f"
    )

    summary = pd.DataFrame(summary_rows).sort_values(["mode", "design", "pi_loc"])
    summary.to_csv(
        out / f"model4_summary_{suffix}.csv",
        index=False,
        float_format="%.6f"
    )
    print(f"[OK] model4 finished for mode={mode}, design={design} -> {out}")
    return summary


if __name__ == "__main__":
    for d in ["A", "B", "C"]:
        for m in ["flat", "tou", "dynamic"]:
            run_model4_for_price_mode(
                mode=m,
                design=d,
                pi_exp=DEFAULTS.PI_EXP,
                tau=DEFAULTS.PI_TAU,
                loc_step=DEFAULTS.LOC_STEP,
                cyclic_soc=True,
                fixed_fee_annual=0.0,
                fee_shares=None,
                solver_verbose=False,
            )