"""
energy_prices.py
================
Builds the three electricity price regimes used in the study. All regimes
describe the ENERGY component of the retail price only; the network tariff is a
separate dimension handled in tariffs.py.

    flat      load-weighted annual mean of the time-of-use tariff
    tou       Glattwerk's published HT/NT x summer/winter energy rates
    dynamic   ENTSO-E day-ahead hourly shape plus a constant markup

All three are revenue-neutral to one another: their load-weighted annual means
are equal, so differences between them reflect the temporal STRUCTURE of the
price signal rather than its LEVEL. This mirrors the treatment already applied
in price_processing.py, with the anchor moved from the wholesale mean to the
actual retail energy price.

Values in CHF/kWh excluding VAT. The published rates are inclusive of 8.1% VAT
and are divided by 1.081; the summer values are confirmed independently by
Glattwerk's worked LEG billing example (0.124 / 0.119 CHF/kWh).

See Supplementary Information S8.3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tariffs import ht_mask, summer_mask, VAT_RATE

PriceMode = str  # "flat" | "tou" | "dynamic"


# Glattwerk `glattpower basic`, energy component, Rp/kWh INCLUDING VAT.
# Source: Preise glattpower basic + Netznutzung Basis, valid from 01.01.2026.
_ENERGY_INCL_VAT = {
    ("winter", "ht"): 13.73,
    ("winter", "nt"): 13.19,
    ("summer", "ht"): 13.40,
    ("summer", "nt"): 12.86,
}

# CHF/kWh excluding VAT
ENERGY_RATES = {k: v / 100.0 / (1.0 + VAT_RATE) for k, v in _ENERGY_INCL_VAT.items()}


def tou_energy_price(index: pd.DatetimeIndex) -> pd.Series:
    """Glattwerk HT/NT x summer/winter energy price, CHF/kWh excl. VAT."""
    ht = ht_mask(index).to_numpy()
    summer = summer_mask(index).to_numpy()
    out = np.empty(len(index), dtype=float)
    out[summer & ht] = ENERGY_RATES[("summer", "ht")]
    out[summer & ~ht] = ENERGY_RATES[("summer", "nt")]
    out[~summer & ht] = ENERGY_RATES[("winter", "ht")]
    out[~summer & ~ht] = ENERGY_RATES[("winter", "nt")]
    return pd.Series(out, index=index, name="pi_imp_tou")


def load_weighted_mean(price: pd.Series, load: pd.Series) -> float:
    """Annual mean of `price` weighted by `load`. This is the revenue-neutral anchor."""
    w = load.reindex(price.index).astype(float)
    if w.sum() <= 0:
        raise ValueError("load weights sum to zero")
    return float((price * w).sum() / w.sum())


def flat_energy_price(index: pd.DatetimeIndex, load: pd.Series) -> pd.Series:
    """Constant price equal to the load-weighted annual mean of the ToU tariff."""
    level = load_weighted_mean(tou_energy_price(index), load)
    return pd.Series(level, index=index, name="pi_imp_flat")


def dynamic_energy_price(
    index: pd.DatetimeIndex,
    load: pd.Series,
    day_ahead_eur: pd.Series,
    eur_to_chf: float,
) -> tuple[pd.Series, float]:
    """
    Day-ahead hourly shape plus a constant markup.

    The markup is set so that the load-weighted annual mean equals the flat
    level, preserving revenue neutrality across the three regimes. An additive
    markup (rather than a multiplicative rescaling) reflects how dynamic retail
    products are actually constructed: the customer faces the wholesale price
    plus a fixed supplier margin covering balancing, certificates and retail
    costs, so the hourly spread of the wholesale market is passed through
    undistorted.

    Returns (price series in CHF/kWh excl. VAT, markup in CHF/kWh).
    """
    da = day_ahead_eur.reindex(index).astype(float) * eur_to_chf
    if da.isna().any():
        raise ValueError("day-ahead price series does not align with the time index")
    target = load_weighted_mean(tou_energy_price(index), load)
    markup = target - load_weighted_mean(da, load)
    return (da + markup).rename("pi_imp_dynamic"), float(markup)


@dataclass(frozen=True)
class PriceSet:
    """The three regimes plus the parameters used to build them."""
    flat: pd.Series
    tou: pd.Series
    dynamic: pd.Series
    level: float          # common load-weighted annual mean, CHF/kWh
    markup: float         # additive markup applied to the day-ahead series

    def get(self, mode: PriceMode) -> pd.Series:
        return {"flat": self.flat, "tou": self.tou, "dynamic": self.dynamic}[mode]


def build_price_set(
    index: pd.DatetimeIndex,
    load: pd.Series,
    day_ahead_eur: pd.Series,
    eur_to_chf: float,
) -> PriceSet:
    """Construct all three revenue-neutral energy price regimes."""
    tou = tou_energy_price(index)
    level = load_weighted_mean(tou, load)
    flat = pd.Series(level, index=index, name="pi_imp_flat")
    dyn, markup = dynamic_energy_price(index, load, day_ahead_eur, eur_to_chf)
    return PriceSet(flat=flat, tou=tou, dynamic=dyn, level=level, markup=markup)


# =========================================================
# Convenience: load a single regime for the community
# =========================================================
def load_energy_price(
    mode: PriceMode,
    index: pd.DatetimeIndex,
    load: pd.Series,
) -> pd.Series:
    """
    Energy price for one regime, in CHF/kWh excl. VAT.

    This is the single entry point for the energy price: every model calls it
    and nothing reads the price files directly. The day-ahead file is quoted in
    EUR/kWh and is converted here at EUR_TO_CHF; the other two regimes are built
    from the published retail tariff and are already in CHF. All three are
    anchored to the same load-weighted annual mean, so they differ in temporal
    structure and not in level (S8.3).
    """
    from utils import read_hourly_price_csv
    from config import PATHS
    from tariffs import EUR_TO_CHF

    if mode == "tou":
        return tou_energy_price(index)
    if mode == "flat":
        return flat_energy_price(index, load)
    if mode == "dynamic":
        da = read_hourly_price_csv(PATHS.PRICE_DYNAMIC)
        price, _ = dynamic_energy_price(index, load, da, EUR_TO_CHF)
        return price
    raise ValueError("mode must be one of: flat, tou, dynamic")
