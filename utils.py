"""
utils.py
========
Shared helpers used by all models: reading the model-ready input timeseries
(inputs_filled.csv), assembling the consumption/generation matrices, building
the pi_loc grid, writing KPI/summary CSVs, and the Gini coefficient reported
alongside the CV in the KPI files.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Union, Optional

from config import PATHS


PathLike = Union[str, Path]


# =========================================================
# I/O: inputs
# =========================================================
def read_inputs_filled(path: PathLike = PATHS.INPUTS_FILLED, time_col: Optional[str] = None) -> Tuple[pd.DataFrame, list[str]]:
    """
    Read wide-format input file:
      time + PV_i / load_i columns
    Returns:
      df: indexed by UTC timestamps (tz-aware)
      unit_ids: sorted list of household ids as strings
    """
    path = Path(path)
    df = pd.read_csv(path)
    if time_col is None:
        time_col = df.columns[0]

    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()

    pv_cols = [c for c in df.columns if re.match(r"^PV_\d+$", str(c))]
    load_cols = [c for c in df.columns if re.match(r"^load_\d+$", str(c))]

    pv_ids = {re.findall(r"\d+", c)[0] for c in pv_cols}
    load_ids = {re.findall(r"\d+", c)[0] for c in load_cols}

    unit_ids = sorted(pv_ids.intersection(load_ids), key=lambda x: int(x))
    if not unit_ids:
        raise ValueError("No matched PV_i and load_i columns found in inputs_filled.csv")

    return df, unit_ids


def build_con_gen_matrices(df: pd.DataFrame, unit_ids: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build consumption (con) and PV generation (gen) matrices. Shapes: (T, N)."""
    con = pd.DataFrame(index=df.index)
    gen = pd.DataFrame(index=df.index)
    for i in unit_ids:
        con[i] = df[f"load_{i}"].astype(float)
        gen[i] = df[f"PV_{i}"].astype(float)
    return con, gen


# =========================================================
# I/O: prices
# =========================================================
def read_hourly_price_csv(path: PathLike) -> pd.Series:
    """
    Read hourly price CSV (one column, first column is datetime index).
    Returns tz-aware UTC Series.
    """
    path = Path(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.shape[1] != 1:
        raise ValueError(f"Price file {path} has {df.shape[1]} columns; expected 1.")
    s = df.iloc[:, 0].astype(float).sort_index()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s


def read_flat_price_txt(path: PathLike) -> float:
    return float(Path(path).read_text().strip())


# =========================================================
# Local sharing price grid
# =========================================================
def build_loc_price_grid(pi_exp: float, pi_imp_t: pd.Series, step: float = 0.01) -> np.ndarray:
    """
    Your agreed rule:
      pi_loc in [pi_exp, floor(mean(pi_imp)/step)*step], step = 0.01
    """
    mean_imp = float(pi_imp_t.mean())
    upper = float(np.floor(mean_imp / step) * step)
    if upper < pi_exp:
        return np.array([pi_exp], dtype=float)
    n = int(np.floor((upper - pi_exp) / step)) + 1
    grid = pi_exp + step * np.arange(n, dtype=float)
    return np.round(grid / step) * step


# =========================================================
# Output helpers
# =========================================================
def ensure_output_dir(path: PathLike = PATHS.OUTPUT_DIR) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_kpi_and_summary(
    model_name: str,
    mode: str,
    kpi: pd.DataFrame,
    summary_dict: dict,
    output_dir: PathLike = PATHS.OUTPUT_DIR,
) -> None:
    out = ensure_output_dir(output_dir)
    # kpi.to_csv(out / f"{model_name}_kpis_{mode}.csv")
    pd.DataFrame([summary_dict]).to_csv(out / f"{model_name}_summary_{mode}.csv", index=False)


# =========================================================
# Misc
# =========================================================
def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    xmin = np.min(x)
    if xmin < 0:
        x = x - xmin
    if np.allclose(x.sum(), 0.0):
        return 0.0
    xs = np.sort(x)
    n = xs.size # number of elements
    cum = np.cumsum(xs) # cumulative sum of sorted values
    return float((n + 1 - 2.0 * np.sum(cum / cum[-1])) / n) # Gini coefficient formula
# note: gini is not relevant for the results analysis at this moment
