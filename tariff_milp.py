"""
tariff_milp.py
==============
Adds network tariff terms to a Gurobi dispatch model.

A network charge that varies with operation is an external price signal, so it
belongs in the dispatch objective:

  volumetric   c_net[t] * x_draw[t], summed over the horizon            (T1, T3)
  refund       - r * x_refund[t]                                        (T1, T3)
  capacity     cap_rate * C, where C is the CONTRACTED connection        (T2, T3)
               capacity, chosen from a discrete menu of three-phase fuse
               ratings and enforced as a hard limit on withdrawal AND injection

WHICH FLOW IS THE BASE. Network charges are levied per metering point on the
energy withdrawn there (Glattwerk basic sheet, clause 5), not on the community's
net exchange with the grid. The households' withdrawals are exogenous -- load net
of own on-site PV self-consumption, unaffected by community membership or by a
battery behind a different meter -- so they contribute only a constant to the
objective and are assessed in settlement. The only endogenous withdrawal in a
community-storage configuration is the storage's own charging, and that is what
`x_draw` should be for models 3 and 4. For model 2 the battery sits behind the
household meter, so the household's own net import is both the withdrawal and
the decision variable.

BOTH THE REFUND AND THE LEG REDUCTION ARE IN THE OBJECTIVE. Clause 7 refunds the
network usage charge on all energy the storage feeds back into the distribution
grid. It is deterministic and known when the schedule is set, so an operator
plans against the net of charge and refund; omitting it would make the model
over-state the cost of cycling by charging storage-mediated energy twice.

The LEG reduction of StromVV Art. 19h is added in the model files rather than
here, because it needs the exogenous surplus and residual-demand profiles. It
looks as though it should be bilinear -- it is a share of a base -- but it is
not: the base and the denominator are both exogenous, being fixed by household
load and on-site generation, so the reduction is linear in eligible community
supply at a constant marginal rate. See network_settlement.marginal_leg_rate and
the LEG block in model3/model4. Leaving it to ex-post settlement, as an earlier
version did, made the storage stand idle for a whole year under a flat tariff,
where the reduction is the only intertemporal signal there is.

CONTRACTED CAPACITY, NOT REALISED PEAK. A connection is bought in discrete sizes,
paid for whether or not it is used, and physically limited to its rating. Modeled
that way the capacity charge gives a SIZING signal and no operational one: within
the limit, using the connection is free. This is deliberate. Charging the realised
monthly peak instead would leave the connection unpriced and unbounded, so a 60 kW
asset could be installed, operated at 2.5 kW and billed for 2.5 kW, which no real
connection tariff permits.

For the storage the contracted capacity is a decision: a smaller connection is
cheaper but curtails both charging and discharging. For a household it is fixed at
the standard rating, so the charge is a constant and only the limit enters.

A note on what this implies. Because the refund covers the whole discharge, the
volumetric term nets down to a charge on round-trip losses plus whatever the
HT/NT calendar makes of the timing difference between charging and discharging.
A volumetric network tariff therefore disciplines storage operation only weakly.
A contracted-capacity charge disciplines the rating instead, and leaves operation
within that rating untouched. That contrast is a result, not an artefact.

See Supplementary Information S8.4.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from tariffs import NetworkTariff, volumetric_rate_series, CES_BANDS


def add_network_tariff(
    model: Any,
    x_draw: Mapping[int, Any],
    index: pd.DatetimeIndex,
    tariff: NetworkTariff,
    *,
    x_refund: Mapping[int, Any] | None = None,
    x_inject: Mapping[int, Any] | None = None,
    choose_band: bool = False,
    fixed_band_kw: float | None = None,
    name_prefix: str = "net",
):
    """
    Add network tariff variables and constraints, and return the objective term.

    Parameters
    ----------
    model      gurobipy.Model being built
    x_draw     dict-like of withdrawal variables indexed 0..T-1 (kWh per hour) at
               the connection point whose charge is endogenous
    index      DatetimeIndex aligned with x_draw, used for the HT/NT calendar
    tariff     the archetype to apply
    x_refund   dict-like of injection variables eligible for the clause 7 refund
               (total storage discharge), or None where the provision does not apply
    x_inject   dict-like of injection variables limited by the connection rating;
               a connection is bidirectional at the same rating
    choose_band    let the model select the contracted capacity from the menu
                   (used for the storage, whose connection is sized deliberately)
    fixed_band_kw  contracted capacity held fixed at this rating (used for a
                   household, which holds the standard connection)

    Returns
    -------
    (obj_term, aux) where obj_term is a gurobipy LinExpr to add to the objective
    and aux is a dict of any auxiliary variables created, for later inspection.
    """
    import gurobipy as gp
    from gurobipy import GRB

    T = len(index)
    if len(x_draw) != T:
        raise ValueError("x_draw and index must have the same length")

    obj = gp.LinExpr()
    aux: dict[str, Any] = {}

    if tariff.name == "T0":
        return obj, aux

    # ---- volumetric, net of the clause 7 refund ---------------------------
    if tariff.has_volumetric:
        c = volumetric_rate_series(index, tariff).to_numpy(dtype=float)
        obj += gp.quicksum(c[t] * x_draw[t] for t in range(T))
        if x_refund is not None:
            if len(x_refund) != T:
                raise ValueError("x_refund and index must have the same length")
            # Single consolidated refund rate, not the HT/NT rate of the hour --
            # see NetworkTariff.refund_rate. Refunding at the hourly rate would
            # make charging at low tariff and discharging at high tariff
            # profitable on the network charge alone, an artefact of the calendar.
            r = tariff.refund_rate
            obj -= gp.quicksum(r * x_refund[t] for t in range(T))

    # ---- contracted connection capacity -----------------------------------
    # The capacity is bought, not measured: it is paid for whether or not it is
    # used, and it caps both withdrawal and injection at this connection.
    if choose_band and fixed_band_kw is not None:
        raise ValueError("pass either choose_band or fixed_band_kw, not both")

    # Where the archetype does not price capacity there is nothing to decide and
    # nothing to charge. Adding the binaries anyway would leave them unpriced, so
    # the solver would fix the connection arbitrarily and impose a limit that the
    # archetype never charged for -- an unpriced constraint differing between
    # archetypes, which would confound the comparison. T0 and T1 therefore carry
    # no connection constraint: the connection is assumed adequate for the asset,
    # which x_ch <= P already enforces.
    if choose_band and not tariff.has_capacity:
        choose_band = False

    if choose_band:
        caps = [kw for _, kw in CES_BANDS]
        y = model.addVars(len(caps), vtype=GRB.BINARY, name=f"{name_prefix}_band")
        model.addConstr(gp.quicksum(y[b] for b in range(len(caps))) == 1,
                        name=f"{name_prefix}_band_one")
        C = gp.quicksum(caps[b] * y[b] for b in range(len(caps)))
        for t in range(T):
            model.addConstr(x_draw[t] <= C, name=f"{name_prefix}_cap_draw[{t}]")
            if x_inject is not None:
                model.addConstr(x_inject[t] <= C, name=f"{name_prefix}_cap_inj[{t}]")
        if tariff.cap_chf_kw_year > 0:
            obj += gp.quicksum(tariff.cap_chf_kw_year * caps[b] * y[b]
                               for b in range(len(caps)))
        aux["band"] = y
        aux["band_caps"] = caps

    elif fixed_band_kw is not None:
        for t in range(T):
            model.addConstr(x_draw[t] <= fixed_band_kw,
                            name=f"{name_prefix}_cap_draw[{t}]")
            if x_inject is not None:
                model.addConstr(x_inject[t] <= fixed_band_kw,
                                name=f"{name_prefix}_cap_inj[{t}]")
        # The charge itself is a constant and does not affect the argmin; it is
        # assessed in settlement. Recorded here so the caller can report it.
        aux["contracted_kw"] = float(fixed_band_kw)

    return obj, aux


def realised_band(aux: Mapping[str, Any]) -> tuple[str, float] | None:
    """Contracted capacity chosen by the solver, as (band name, kW)."""
    y = aux.get("band")
    if y is None:
        kw = aux.get("contracted_kw")
        return (None, float(kw)) if kw is not None else None
    for b, (name, kw) in enumerate(CES_BANDS):
        if y[b].X > 0.5:
            return name, kw
    return None
