"""
price_processing.py
===================
One-off preprocessing (run manually before main.py, not part of a model run):
convert the raw ENTSO-E day-ahead export (input_data/CH_price_dyn_2025.csv) into
the hourly day-ahead series used by the models:
  price_dynamic_2025.csv     hourly dynamic price

The flat and time-of-use regimes are not derived here: they are built from the
distribution system operator's published rates in energy_prices.py. The helper
functions below (build_tou_3block, revenue_neutralize) are kept for reference.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

# =========================================================
# Core functions
# =========================================================

def read_entsoe_day_ahead(path: str) -> pd.Series:
    """
    Read ENTSO-E exported Excel/CSV and return hourly day-ahead price as a Series.
    Output unit: EUR/kWh (converted from EUR/MWh by /1000).
    Index: UTC timestamps at the interval start time.
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df = df[["MTU (UTC)", "Day-ahead Price (EUR/MWh)"]].copy()

    # Parse time interval string: "01/01/2025 00:00:00 - 01/01/2025 01:00:00"
    df["start_time"] = df["MTU (UTC)"].astype(str).str.split(" - ").str[0]
    df["start_time"] = pd.to_datetime(
        df["start_time"],
        format="%d/%m/%Y %H:%M:%S",
        utc=True
    )

    df = df.set_index("start_time").sort_index()

    price = df["Day-ahead Price (EUR/MWh)"].astype(float)
    price = price / 1000.0  # EUR/kWh
    price.name = "price_dyn_eur_per_kwh"

    return price


def build_tou_3block(
    price_dyn: pd.Series,
    tz: str = None,
    evening_peak: Tuple[int, int] = (17, 22),
    night: Tuple[int, int] = (0, 6),
    weekend_offpeak: bool = False
) -> pd.Series:
    """
    Build a 3-block TOU tariff from a dynamic hourly price series.
    Blocks:
      - night: [night[0], night[1])
      - evening peak: [evening_peak[0], evening_peak[1])
      - mid: all remaining hours
    Prices per block = mean of dynamic prices within that block (before scaling).
    If weekend_offpeak=True, peak is removed on weekends.
    """
    s = price_dyn.copy()

    # Use local time for hour/weekend classification, but return in original tz
    original_tz = s.index.tz
    if tz is not None:
        s = s.tz_convert(tz)

    hour = s.index.hour
    is_weekend = (s.index.weekday >= 5)

    mask_night = (hour >= night[0]) & (hour < night[1])
    mask_peak = (hour >= evening_peak[0]) & (hour < evening_peak[1])

    if weekend_offpeak:
        mask_peak = mask_peak & (~is_weekend)

    mask_mid = ~(mask_night | mask_peak)

    p_night = float(s[mask_night].mean()) if mask_night.any() else float(s.mean())
    p_peak  = float(s[mask_peak].mean())  if mask_peak.any()  else float(s.mean())
    p_mid   = float(s[mask_mid].mean())   if mask_mid.any()   else float(s.mean())

    tou = pd.Series(index=s.index, dtype=float)
    tou[mask_night] = p_night
    tou[mask_peak]  = p_peak
    tou[mask_mid]   = p_mid

    # Return to original tz
    if tz is not None and original_tz is not None:
        tou = tou.tz_convert(original_tz)
    elif tz is not None and original_tz is None:
        # If original was tz-naive (unlikely here), just drop tz
        tou.index = tou.index.tz_localize(None)

    tou.name = "price_tou_eur_per_kwh_raw"
    return tou


def revenue_neutralize(tariff: pd.Series, target_mean: float) -> pd.Series:
    """
    Scale tariff so that its mean equals target_mean (revenue-neutral vs flat).
    """
    if tariff.isna().any():
        raise ValueError("tariff contains NaN; cannot revenue-neutralize.")
    m = float(tariff.mean())
    if m == 0:
        raise ValueError("tariff mean is 0; cannot revenue-neutralize.")
    scaled = tariff * (target_mean / m)
    scaled.name = f"{tariff.name}_rev_neutral"
    return scaled


# =========================================================
# Processing + saving (debug-friendly entry point)
# =========================================================

def process_prices_and_save(
    entsoe_file: str,
    out_dir: str,
    local_tz: str = "Europe/Zurich",
    weekend_offpeak: bool = True,
    evening_peak: Tuple[int, int] = (17, 22),
    night: Tuple[int, int] = (0, 6),
) -> dict:
    """
    Read the ENTSO-E day-ahead series (EUR/kWh) and save it as
    price_dynamic_2025.csv in out_dir. Returns the series and its annual mean.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) dynamic
    price_dyn = read_entsoe_day_ahead(entsoe_file)

    # Basic checks
    if not isinstance(price_dyn.index, pd.DatetimeIndex) or price_dyn.index.tz is None:
        raise ValueError("price_dyn index must be timezone-aware DatetimeIndex (UTC).")
    if price_dyn.index.duplicated().any():
        raise ValueError("Duplicate timestamps found in ENTSO-E price series.")
    if len(price_dyn) not in (8760, 8784):
        print(f"[WARN] Expected 8760 (or 8784 leap-year) hours, got {len(price_dyn)}.")

    price_flat = float(price_dyn.mean())

    # 2) save the hourly day-ahead series
    price_dyn.rename("price_dyn_eur_per_kwh").to_frame().to_csv(
        out_path / "price_dynamic_2025.csv"
    )

    # The flat and time-of-use series are not written: energy_prices.py builds
    # both regimes from the published retail rates instead.
    # price_tou = revenue_neutralize(
    #     build_tou_3block(price_dyn, tz=local_tz, evening_peak=evening_peak,
    #                      night=night, weekend_offpeak=weekend_offpeak),
    #     target_mean=price_flat)
    # price_tou.to_frame().to_csv(out_path / "price_tou_3block_2025.csv")
    # with open(out_path / "price_flat_2025.txt", "w") as f:
    #     f.write(f"{price_flat:.10f}")

    print("[OK] Saved prices to:", out_path.resolve())
    print("Dynamic mean:", price_flat)

    return {
        "price_dyn": price_dyn,
        "price_flat": price_flat,
    }


# =========================================================
# CLI-style debug run
# =========================================================
if __name__ == "__main__":
    # Adjust these paths to your project structure
    entsoe_file = "input_data/CH_price_dyn_2025.csv"      # or .csv
    out_dir = "input_data"                               # save back to input folder

    process_prices_and_save(
        entsoe_file=entsoe_file,
        out_dir=out_dir,
        local_tz="Europe/Zurich",
        weekend_offpeak=True,
        evening_peak=(17, 22),
        night=(0, 6),
    )