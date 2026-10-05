"""
network_settlement.py
=====================
Ex-post settlement of network charges, the statutory LEG reduction, and the
refund on energy discharged from storage back into the distribution grid.

Charges are assessed PER CONNECTION POINT -- Glattwerk basic tariff sheet,
clause 5: "Muss die Energie einem Kunden an mehr als einer Stelle abgegeben
werden, so wird die Netznutzung von jeder Messstelle einzeln verrechnet."
The community therefore holds eight metering points: one per household and one
for the storage.

    volumetric   on that connection's own withdrawal
    capacity     on the CONTRACTED capacity of that connection       (T2, T3)
    fixed, meter per connection

The capacity charge is paid on the connection that has been contracted, not on
the power that happened to flow through it: a connection is bought in discrete
sizes, paid for whether or not it is used, and physically limited to its rating.
Every dwelling holds the standard household connection; the storage contracts a
size chosen alongside its dispatch, which is why the contracted capacity is
returned from the dispatch stage and passed in here.

WHAT COUNTS AS A WITHDRAWAL. A household withdraws its load net of its own
on-site PV self-consumption. Electricity received from another member, or from
the community storage, travels over the distribution grid and is metered and
charged at the receiving connection exactly like grid-supplied electricity; the
LEG mechanism does not exempt it, it discounts it. The operator's published
worked example bills the network charge on all 400 kWh withdrawn, of which
110 kWh is LEG-supplied, and then applies the reduction to the whole base.
A household's withdrawal is consequently EXOGENOUS: joining the community, and
the operation of a community battery behind a different meter, leave it
unchanged. Only storage behind the household's own meter alters it.

Two statutory provisions are applied here rather than in dispatch:

  * StromVV Art. 19h -- the LEG reduction. It depends on the realised share of
    locally supplied electricity, which is an outcome of dispatch, so settling it
    ex post keeps the dispatch problem linear. Paragraph 4 caps the eligible
    storage discharge at the amount the storage drew from the community over the
    billing period.

  * Clause 7 -- "Fuer Speicher mit Endverbrauchern, wird auf Antrag das
    Netznutzungsentgelt fuer die aus dem Speicher ins Verteilnetz zurueckgespeiste
    Energiemenge rueckerstattet." The refund covers ALL energy the storage feeds
    back into the distribution grid, whether it is subsequently withdrawn by a
    community member or exported. Its purpose is to prevent the same kilowatt-hour
    being charged twice, at the storage connection and again at the consuming
    connection. The clause states that the statutory components are bundled into a
    separate "Rueckerstattungstarif" whose value is not published; we refund at the
    volumetric Netznutzung rate, which is the natural reading and is documented as
    an assumption in S8.5. Only the volumetric component is refundable -- the
    refund attaches to an Energiemenge.

See Supplementary Information S8.4 and S8.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tariffs import (
    HOUSEHOLD_BAND_KW,
    NetworkCost,
    NetworkTariff,
    leg_reduction,
    volumetric_rate_series,
)


# =========================================================
# Per-connection network cost
# =========================================================
def connection_cost(
    x_draw: pd.Series,
    tariff: NetworkTariff,
    *,
    contracted_kw: float = HOUSEHOLD_BAND_KW,
    n_months: int = 12,
) -> NetworkCost:
    """
    Network cost of one connection point over the year.

    `x_draw` is the hourly energy withdrawn at that metering point (kWh), which
    for a household is its load net of its own on-site PV self-consumption, and
    for the storage is its total charging. `contracted_kw` is the capacity of
    the connection, which the capacity charge is levied on whether or not it is
    used.
    """
    if tariff.name == "T0":
        return NetworkCost()

    vol = 0.0
    if tariff.has_volumetric:
        vol = float((volumetric_rate_series(x_draw.index, tariff).to_numpy()
                     * x_draw.to_numpy()).sum())

    cap = float(tariff.cap_chf_kw_year * contracted_kw) if tariff.has_capacity else 0.0

    return NetworkCost(volumetric=vol, capacity=cap,
                       fixed=tariff.fixed_chf_month * n_months,
                       metering=tariff.meter_chf_month * n_months)


def marginal_leg_rate(x_draw: pd.DataFrame, tariff: NetworkTariff, delta: float) -> float:
    """
    Value to the community of one additional kWh supplied from within the
    community, through the statutory reduction, in CHF/kWh.

        R_i = discountable_i * (E_LEG_i / E_draw_i) * delta

    Both the discountable base and the withdrawal are exogenous -- they are set
    by household load and on-site generation, which no dispatch decision changes
    -- so the reduction is LINEAR in eligible community supply and its marginal
    rate is a constant. That constant is what the dispatch problem needs in order
    to value storage correctly; settling the reduction only ex post leaves the
    operator blind to it, and under a flat tariff, where there is no arbitrage
    spread to compete with it, that omission is the whole signal.

    We use the community-level average rate, aggregating over households. The
    per-household rates differ slightly because the fixed charge is not
    proportional to withdrawal; settlement still applies the exact per-household
    formula, so the approximation affects only the dispatch signal.
    """
    if delta <= 0 or tariff.name == "T0":
        return 0.0
    disc = tot = 0.0
    for h in x_draw.columns:
        c = connection_cost(x_draw[h], tariff, contracted_kw=HOUSEHOLD_BAND_KW)
        disc += c.discountable
        tot += float(x_draw[h].sum())
    return float(disc * delta / tot) if tot > 0 else 0.0


def storage_grid_refund(x_dis: pd.Series, tariff: NetworkTariff) -> float:
    """
    Refund of the network usage charge on energy the storage feeds back into the
    distribution grid (clause 7), granted on request after billing.

    The base is TOTAL discharge, not only the part exported beyond the community:
    the storage has no direct connection to any member, so every kilowatt-hour it
    discharges enters the distribution grid before being withdrawn again at a
    member's meter, where it is charged a second time. Refunding only the exported
    part would leave storage-mediated energy double-charged.

    Only the volumetric component is refundable; the capacity and fixed charges
    are properties of the connection and do not attach to an energy quantity.

    The refund is applied at the single consolidated rate `tariff.refund_rate`,
    not at the HT/NT rate of the discharge hour -- see NetworkTariff.refund_rate.
    """
    if not tariff.has_volumetric:
        return 0.0
    return float(tariff.refund_rate * x_dis.to_numpy().sum())


# =========================================================
# StromVV Art. 19h para. 4 -- eligible storage discharge
# =========================================================
def eligible_discharge_ratio(x_dis_load_total: float, x_ch_local_total: float) -> float:
    """
    Fraction of storage discharge to community load that qualifies for the LEG
    reduction. The storage may not supply more within the community than it drew
    from the community over the billing period; the entitlement lapses for the
    excess. The constraint is assessed on the storage as a whole, so the
    resulting ratio is applied uniformly to each household's allocated share.
    """
    if x_dis_load_total <= 0:
        return 0.0
    return float(min(1.0, x_ch_local_total / x_dis_load_total))


# =========================================================
# Community-level settlement
# =========================================================
@dataclass
class NetworkSettlement:
    """Outcome of the network settlement. All values CHF/year excl. VAT."""
    household_cost: pd.Series          # net of the LEG reduction
    household_gross: pd.Series         # before the reduction
    household_reduction: pd.Series     # the reduction itself, >= 0
    ces_cost: float = 0.0              # storage connection, net of its grid refund
    ces_gross: float = 0.0
    ces_refund: float = 0.0
    eligible_ratio: float = 1.0
    delta_applied: float = 0.0
    detail: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return float(self.household_cost.sum()) + self.ces_cost


def settle_network(
    x_draw: pd.DataFrame,
    e_leg_direct: pd.Series,
    x_dis_load_alloc: pd.DataFrame | None,
    tariff: NetworkTariff,
    delta: float,
    *,
    ces_ch: pd.Series | None = None,
    ces_dis: pd.Series | None = None,
    ces_contracted_kw: float | None = None,
    x_ch_local_total: float = 0.0,
) -> NetworkSettlement:
    """
    Settle network charges for the whole community.

    Parameters
    ----------
    x_draw             hourly energy withdrawn at each household's metering point
                       (kWh): its load net of its own on-site PV self-consumption.
                       Its annual total is also the DENOMINATOR of the LEG share:
                       the reduction is a percentage of the network charge, and
                       that charge is levied on the withdrawal, so the share must
                       be measured against the same quantity. Using total
                       consumption instead would inflate the denominator by a
                       household's own on-site self-consumption -- up to 89% here
                       -- and would under-credit PV owners only, biasing the
                       distributional results. The operator's worked example is on
                       this basis: its 400 kWh denominator is the sum of the four
                       metered lines, not load plus self-consumption.
    e_leg_direct       annual electricity each household received directly from
                       other members, bypassing storage (kWh)
    x_dis_load_alloc   hourly storage discharge allocated to each household, or
                       None where the configuration has no community storage
    tariff             network tariff archetype
    delta              statutory LEG reduction rate (0 to 1)
    ces_ch             hourly total charging of the community storage (its
                       withdrawal at its own connection point)
    ces_dis            hourly total discharge of the community storage (the base
                       of the clause 7 refund)
    ces_contracted_kw  connection capacity contracted for the storage, chosen in
                       the dispatch stage
    x_ch_local_total   annual storage charging drawn from community generation,
                       used to apply StromVV Art. 19h para. 4
    """
    households = list(x_draw.columns)

    # --- eligible LEG energy per household -------------------------------
    if x_dis_load_alloc is not None and not x_dis_load_alloc.empty:
        dis_tot = x_dis_load_alloc.sum(axis=0)
        ratio = eligible_discharge_ratio(float(dis_tot.sum()), float(x_ch_local_total))
        e_leg = e_leg_direct.reindex(households).fillna(0.0) + dis_tot * ratio
    else:
        ratio = 1.0
        e_leg = e_leg_direct.reindex(households).fillna(0.0)

    # --- per-household charges and reduction ------------------------------
    gross, reduction, detail = {}, {}, {}
    for h in households:
        c = connection_cost(x_draw[h], tariff, contracted_kw=HOUSEHOLD_BAND_KW)
        r = leg_reduction(c, float(e_leg[h]), float(x_draw[h].sum()), delta)
        gross[h], reduction[h] = c.total, r
        detail[h] = c

    gross_s = pd.Series(gross, dtype=float)
    red_s = pd.Series(reduction, dtype=float)

    # --- storage connection ------------------------------------------------
    # The storage is a separate metering point. It is charged on what it
    # withdraws (its total charging, whatever the source) and refunded on what it
    # feeds back. No LEG reduction is claimed for it: Art. 19h para. 1 attaches
    # the reduction to a participant's withdrawal of self-generated electricity,
    # and the storage's own withdrawal is an intermediate flow whose final
    # consumption is already discounted at the member's meter. This is the
    # conservative treatment and it is close to costless -- see S8.5.
    ces_gross = ces_refund = 0.0
    if ces_ch is not None and tariff.name != "T0":
        c_ces = connection_cost(ces_ch, tariff,
                                contracted_kw=float(ces_contracted_kw or 0.0))
        ces_gross = c_ces.total
        detail["CES"] = c_ces
        if ces_dis is not None:
            ces_refund = storage_grid_refund(ces_dis, tariff)

    return NetworkSettlement(
        household_cost=gross_s - red_s,
        household_gross=gross_s,
        household_reduction=red_s,
        ces_cost=ces_gross - ces_refund,
        ces_gross=ces_gross,
        ces_refund=ces_refund,
        eligible_ratio=ratio,
        detail=detail,
        delta_applied=float(delta),
    )
