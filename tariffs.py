"""
tariffs.py
==========
Network tariff archetypes, time-of-use calendar, and the statutory network-fee
reduction for local electricity communities (LEG).

All monetary values are in CHF and EXCLUDE VAT (8.1% in Switzerland). VAT is a
uniform proportional markup and therefore cancels in every relative metric
reported by the study (savings %, CV, Gini). Amounts published in EUR are
converted at the 2025 annual average rate, 1 EUR = 0.9370 CHF.

The archetypes span the two dimensions along which a network tariff can carry a
behavioral signal -- energy volume and power -- as a 2x2 factorial:

                   no capacity charge      capacity charge
    no volumetric  (pure fixed charge)     T2  capacity only
    volumetric     T1  volumetric only     T3  volumetric + capacity

    T0  no network charge at all (pure-market benchmark; reproduces the
        originally submitted model)

The empty cell is a purely fixed charge per connection point, as levied on Dutch
small-consumer connections (Stedin). It is not run as a separate archetype: with
no volumetric and no capacity component it adds a constant to every configuration
and produces dispatch identical to T0, which already spans the no-signal case.

All three charging archetypes are calibrated to recover the SAME network revenue
from the baseline (Model 0, no community, no storage): CHF 3,697.6/yr across the
seven case-study connections. Revenue neutrality separates the STRUCTURE of cost
recovery, which is what the archetypes vary, from its LEVEL, which is
jurisdiction-specific. T1 is the anchor and uses the published Glattwerk basic
rates unchanged; T2 and T3 are scaled to match it.

Charges are assessed PER CONNECTION POINT, on the energy withdrawn at that point
-- Glattwerk basic tariff sheet, clause 5: "Muss die Energie einem Kunden an mehr
als einer Stelle abgegeben werden, so wird die Netznutzung von jeder Messstelle
einzeln verrechnet." A household's withdrawal is its load net of its own on-site
PV self-consumption; electricity received from other members or from the
community storage travels over the distribution grid and is charged like any
other withdrawal, then discounted under StromVV Art. 19h.

Sources are documented in ../network-tariff-data.md and the PDFs in
../tariff-sources/. See Supplementary Information S8 of the manuscript.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Archetype = Literal["T0", "T1", "T2", "T3"]

EUR_TO_CHF = 0.9370          # 2025 annual average
VAT_RATE = 0.081             # Swiss VAT, used only to convert published incl.-VAT figures

LOCAL_TZ = "Europe/Zurich"   # HT/NT windows are defined in local time

# Baseline network revenue recovered from the seven case-study connections under
# Model 0, at the published Glattwerk basic rates. All archetypes are calibrated
# to this figure. See calibrate_tariffs.py.
M0_NETWORK_REVENUE_CHF = 3697.6

# Baseline withdrawal split between high and low tariff hours, kWh. Used to
# derive the flat refund rate below.
M0_HT_KWH = 14049.2
M0_NT_KWH = 21262.0


# =========================================================
# Time-of-use calendar (Glattwerk)
# =========================================================
def ht_mask(index: pd.DatetimeIndex) -> pd.Series:
    """
    High-tariff (Hochtarif) hours: Mon-Fri 07:00-20:00 and Sat 07:00-13:00,
    Europe/Zurich local time. All other hours are low tariff (Niedertarif).
    """
    loc = index.tz_convert(LOCAL_TZ) if index.tz is not None else index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    h, dow = loc.hour, loc.dayofweek
    weekday = (dow < 5) & (h >= 7) & (h < 20)
    saturday = (dow == 5) & (h >= 7) & (h < 13)
    return pd.Series(weekday | saturday, index=index, name="ht")


def summer_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Summer season: 1 April - 30 September (local time). Winter otherwise."""
    loc = index.tz_convert(LOCAL_TZ) if index.tz is not None else index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    return pd.Series(loc.month.isin(range(4, 10)), index=index, name="summer")


def month_key(index: pd.DatetimeIndex) -> pd.Series:
    """Billing-month label (local time), used for monthly peak charges."""
    loc = index.tz_convert(LOCAL_TZ) if index.tz is not None else index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    return pd.Series(loc.strftime("%Y-%m"), index=index, name="month")


# =========================================================
# Connection capacity
# =========================================================
# A connection is bought in discrete sizes -- fuse ratings -- not in arbitrary
# kilowatts. We take the SIZES from the Dutch small-consumer schedule (Stedin),
# which is the standard three-phase ladder, but NOT its prices: those are set by
# our own revenue anchor at a single rate per kW (see T2 below). Importing the
# published Dutch amounts would import their category-factor schedule, in which
# the capaciteitstarief is a base of EUR 729.40/yr times a regulated factor
# {0.005, 0.05, 0.4, 2, 3, 4, 5}. The jump from 0.4 at 3x25 A to 2 at 3x35 A is a
# deliberate cross-subsidy of the standard household connection -- linear
# scaling in amperes would put 3x25 A at about 1.43 -- and is a distributional
# policy choice rather than a cost signal. Building it in would make part of the
# contrast between archetypes an artefact of that subsidy.
CONNECTION_BANDS: tuple[tuple[str, float], ...] = (
    ("1x10A", 2.3),
    ("3x25A", 17.3),
    ("3x35A", 24.2),
    ("3x50A", 34.6),
    ("3x63A", 43.6),
    ("3x80A", 55.4),
)

# A community-scale storage installation is connected three-phase. The single-
# phase 1x10 A supply (230 V, 2.3 kW) exists for a garage or a bedsit, not for a
# 121 kWh battery, so it is not in the menu the storage chooses from.
CES_BANDS: tuple[tuple[str, float], ...] = CONNECTION_BANDS[1:]

# Every dwelling holds the standard three-phase household connection. Household
# connections are not re-contracted year by year, and every unit's peak
# withdrawal (1.4 to 7.0 kW) sits well inside this rating.
HOUSEHOLD_BAND_KW = 17.3


# =========================================================
# Archetype definitions
# =========================================================
@dataclass(frozen=True)
class NetworkTariff:
    """
    A network tariff archetype. All rates in CHF excl. VAT.

    vol_ht / vol_nt      volumetric charge on withdrawal, CHF/kWh
    cap_chf_kw_year      capacity charge on the CONTRACTED connection capacity,
                         CHF/kW/year. It is paid whether or not the capacity is
                         used, and the contracted capacity is a hard limit on
                         withdrawal and on injection at that connection.
    fixed_chf_month      fixed charge per connection point, CHF/month (Grundpreis)
    meter_chf_month      metering charge per connection point, CHF/month (Messpreis)
    """
    name: Archetype
    vol_ht: float = 0.0
    vol_nt: float = 0.0
    cap_chf_kw_year: float = 0.0
    fixed_chf_month: float = 0.0
    meter_chf_month: float = 0.0
    label: str = ""

    @property
    def has_volumetric(self) -> bool:
        return self.vol_ht > 0 or self.vol_nt > 0

    @property
    def has_capacity(self) -> bool:
        return self.cap_chf_kw_year > 0

    @property
    def refund_rate(self) -> float:
        """Flat rate at which energy fed from storage into the distribution grid
        is refunded, CHF/kWh (clause 7 of the basic tariff sheet).

        The clause consolidates the refundable components into a single
        *Rueckerstattungstarif* -- "Alle gesetzlich vorgeschriebenen Komponenten
        sind dabei im Rueckerstattungstarif zusammengefasst" -- whose value is not
        published. We therefore refund at ONE rate rather than at the HT/NT rate
        prevailing at the hour of discharge, and set it to the baseline
        load-weighted mean of the volumetric charge so that the refund is
        revenue-neutral against the charge it reverses.

        Refunding at the hourly rate instead would make the network tariff pay the
        battery to cycle: charging in a low-tariff hour and discharging in a
        high-tariff hour would return -0.071 + 0.9 x 0.081 = +0.0019 CHF/kWh under
        T1, a profit created purely by the calendar. A single consolidated rate
        removes that artefact, and is the reading the clause supports.
        """
        if not self.has_volumetric:
            return 0.0
        return (M0_HT_KWH * self.vol_ht + M0_NT_KWH * self.vol_nt) / (M0_HT_KWH + M0_NT_KWH)


# The Grundpreis and Messpreis of the household class are common to all three
# charging archetypes: they are the non-behavioral residual of the bill, and
# holding them fixed is what makes the volumetric/capacity contrast clean.
_FIXED = 7.50    # Grundpreis, CHF/month  (Glattwerk basic, excl. VAT)
_METER = 5.00    # Messpreis,  CHF/month  (Glattwerk basic, excl. VAT)

# T0: no network charge; reproduces the originally submitted model.
T0 = NetworkTariff(name="T0", label="No network charge (benchmark)")

# T1: volumetric only. Published Glattwerk `glattpower basic + Netznutzung Basis`
#     rates, < 50 MWh/yr, grid level 7, used unchanged. Volumetric rates published
#     incl. VAT (8.76 / 7.68 Rp/kWh) and divided by 1.081; the excl.-VAT values are
#     confirmed independently by the operator's published LEG invoice example.
#     This archetype is the revenue anchor for T2 and T3.
T1 = NetworkTariff(
    name="T1",
    vol_ht=0.0810, vol_nt=0.0710,
    fixed_chf_month=_FIXED, meter_chf_month=_METER,
    label="Volumetric only",
)

# T2: capacity only, on CONTRACTED connection capacity.
#
#     Capacity-only recovery is not hypothetical: Dutch small-consumer network
#     tariffs contain no charge per kilowatt-hour at all, and the entire network
#     bill follows from the capacity of the connection (Stedin). We adopt that
#     basis -- contracted capacity, not realised peak.
#
#     The distinction matters. A charge on the realised monthly peak leaves the
#     connection itself unpriced and unbounded, so an operator can install a
#     60 kW asset, use 2.5 kW of it, and pay accordingly. A contracted-capacity
#     charge is what real connection tariffs do: the capacity is bought in
#     discrete sizes, paid for whether or not it is used, and enforced as a hard
#     limit (in Spain by the ICP breaker; in the Netherlands and Switzerland by
#     the fuse). Sizing therefore becomes a genuine decision and operation
#     receives no signal at all beyond the limit -- which is the standard
#     critique of connection-based tariffs and is here a result rather than an
#     assumption.
#
#     Rate set by revenue neutrality: CHF 2,647.6 over the seven households'
#     contracted capacity of 7 x 17.3 = 121.1 kW.
#
#     Cross-check: the Dutch charge for the same 3x25 A connection, net of
#     metering, is CHF 333.5/yr, or 19.28 CHF/kW/yr -- within 13% of the
#     calibrated rate. Above that band the Dutch schedule is roughly three times
#     steeper, so pricing every band at one rate is CONSERVATIVE for the storage
#     connection; the Dutch factor schedule is reported as a sensitivity.
T2 = NetworkTariff(
    name="T2",
    cap_chf_kw_year=21.8629,
    fixed_chf_month=_FIXED, meter_chf_month=_METER,
    label="Capacity only",
)

# T3: volumetric + capacity. Structure taken from `glattpower business +
#     Netznutzung NS` (50-1000 MWh/yr, grid level 7), which at published rates of
#     5.00/4.60 Rp/kWh and 10.50 CHF/kW/month splits its variable recovery
#     44.3% volumetric / 55.7% capacity on the case-study data. That split is
#     preserved and the level rebalanced to the household anchor (k = 0.69808);
#     the capacity half is then recovered from contracted capacity on the same
#     basis as T2, so that "capacity" means the same thing in both cells of the
#     factorial.
#
#         volumetric  0.443 x 2647.6 = 1173.1  ->  3.4904 / 3.2112 Rp/kWh
#         capacity    0.557 x 2647.6 = 1474.5  ->  12.1756 CHF/kW/yr over 121.1 kW
_K_T3 = 0.69808
T3 = NetworkTariff(
    name="T3",
    vol_ht=0.034904, vol_nt=0.032112,
    cap_chf_kw_year=12.1756,
    fixed_chf_month=_FIXED, meter_chf_month=_METER,
    label="Volumetric + capacity",
)

ARCHETYPES: dict[str, NetworkTariff] = {"T0": T0, "T1": T1, "T2": T2, "T3": T3}


# =========================================================
# Cost components
# =========================================================
def volumetric_rate_series(index: pd.DatetimeIndex, tariff: NetworkTariff) -> pd.Series:
    """Hourly volumetric network charge on withdrawal, CHF/kWh."""
    if not tariff.has_volumetric:
        return pd.Series(0.0, index=index, name=f"c_net_{tariff.name}")
    ht = ht_mask(index)
    return pd.Series(np.where(ht, tariff.vol_ht, tariff.vol_nt),
                     index=index, name=f"c_net_{tariff.name}")


@dataclass
class NetworkCost:
    """Annual network cost of one connection point, decomposed. CHF excl. VAT."""
    volumetric: float = 0.0
    capacity: float = 0.0
    fixed: float = 0.0
    metering: float = 0.0

    @property
    def discountable(self) -> float:
        """Base to which the LEG reduction applies: volumetric + Grundtarif +
        Leistungstarif. Metering is excluded, and so are the levies listed in
        StromVV Art. 19h para. 5 (system services, winter reserve, Netzzuschlag,
        charges to public authorities), which are not part of the Netznutzung
        component modeled here. Verified against the operator's worked example
        (see leg_reduction)."""
        return self.volumetric + self.capacity + self.fixed

    @property
    def total(self) -> float:
        return self.volumetric + self.capacity + self.fixed + self.metering


# =========================================================
# LEG reduction of the network tariff (StromVV Art. 19h)
# =========================================================
def leg_reduction(
    cost: NetworkCost,
    e_leg: float,
    e_total: float,
    delta: float,
) -> float:
    """
    Statutory reduction of the network usage tariff for a LEG participant.

        R = (volumetric + capacity + fixed) * (E_LEG / E_total) * delta

    Verified against Glattwerk's published worked example, in which a participant
    withdrawing 400 kWh of which 110 kWh is LEG-supplied is billed the network
    charge on ALL 400 kWh plus the Grundpreis -- 12.15 + 9.94 + 4.05 + 4.26 + 22.50
    = CHF 53.00 -- and receives 53.00 * 27.5% * 20% = CHF 2.92. The Grundpreis is
    inside the base; the Messpreis and the Art. 19h para. 5 levies are outside it.

    `e_leg` is the electricity supplied from within the community that is
    eligible for the reduction. Under StromVV Art. 19h para. 4 a storage facility
    may not, over a billing period, supply more electricity within the community
    than it draws from it; the entitlement lapses for any excess. Use
    `eligible_leg_energy()` to apply that cap before calling this function.
    """
    if e_total <= 0 or delta <= 0:
        return 0.0
    share = min(max(e_leg / e_total, 0.0), 1.0)
    return cost.discountable * share * delta


def eligible_leg_energy(
    x_shared_direct: float,
    x_dis_load: float,
    x_ch_from_community: float,
) -> float:
    """
    Electricity eligible for the LEG reduction, applying StromVV Art. 19h para. 4.

    Direct local sharing is always eligible. Storage discharge to community load
    is eligible only up to the amount the storage drew *from the community*;
    energy charged from the grid and resold into the community does not qualify
    (grid charging is not a draw from the community).

    Parameters are annual energies in kWh:
      x_shared_direct        PV shared directly between members, bypassing storage
      x_dis_load             storage discharge serving community load
      x_ch_from_community    storage charging taken from community generation
    """
    return float(x_shared_direct + min(x_dis_load, x_ch_from_community))
