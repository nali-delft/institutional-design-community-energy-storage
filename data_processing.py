"""
data_processing.py
==================
One-off preprocessing (run manually before main.py, not part of a model run):
read the raw wide PV/load table (input_data/inputs.csv), clean and gap-fill it,
and write the model-ready input_data/inputs_filled.csv with columns
PV_<id>, load_<id> on an hourly index.
"""
import re
import numpy as np
import pandas as pd

# ---------- 1) Read and split ----------
def read_wide_pv_load(path: str, time_col: str = None):
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if time_col is None:
        time_col = df.columns[0]

    # keep original column order for later reconstruction
    original_cols = list(df.columns)

    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()

    pv_cols = [c for c in df.columns if re.match(r"^PV_\d+$", str(c))]
    load_cols = [c for c in df.columns if re.match(r"^load_\d+$", str(c))]

    pv_map = {re.findall(r"\d+", c)[0]: c for c in pv_cols}      # id -> "PV_i"
    load_map = {re.findall(r"\d+", c)[0]: c for c in load_cols}  # id -> "load_i"
    hh_ids = sorted(set(pv_map.keys()).intersection(load_map.keys()), key=lambda x: int(x))

    pv_df = df[[pv_map[i] for i in hh_ids]].copy()
    pv_df.columns = [f"H{i}" for i in hh_ids]

    load_df = df[[load_map[i] for i in hh_ids]].copy()
    load_df.columns = [f"H{i}" for i in hh_ids]

    meta = {
        "time_col": time_col,
        "original_cols": original_cols,  # includes time_col
        "pv_map": pv_map,
        "load_map": load_map,
        "hh_ids": hh_ids,
    }
    return df, pv_df, load_df, meta

# ---------- 2) Gap-filling ----------
# Short gaps (<= 6 hours) are filled by time interpolation, longer gaps are
# imputed using month/weekday/hour-specific medians, with non-negativity
# enforced and nighttime PV (before 5:00 and after 21:00) set to zero.
def fill_missing(df: pd.DataFrame, kind: str, short_gap_max_hours: int = 6) -> pd.DataFrame:
    if kind not in ("pv", "load"):
        raise ValueError("kind must be 'pv' or 'load'")
    out = df.sort_index().copy()

    feat = pd.DataFrame(index=out.index)
    feat["month"] = out.index.month
    feat["hour"] = out.index.hour
    feat["is_weekend"] = (out.index.weekday >= 5).astype(int)

    for col in out.columns:
        s = out[col].astype(float)

        # short gaps: time interpolation
        s1 = s.interpolate(method="time", limit=short_gap_max_hours, limit_direction="both")

        # long gaps: median by (month, weekend, hour)
        base = pd.DataFrame({
            "y": s1,
            "month": feat["month"],
            "hour": feat["hour"],
            "is_weekend": feat["is_weekend"],
        })
        med = base.dropna().groupby(["month", "is_weekend", "hour"])["y"].median()

        miss = base.index[base["y"].isna()]
        if len(miss) > 0:
            keys = list(zip(feat.loc[miss, "month"], feat.loc[miss, "is_weekend"], feat.loc[miss, "hour"]))
            fill_vals = np.array([med.get(k, np.nan) for k in keys], dtype=float)
            s1.loc[miss] = fill_vals

        # fallback
        if s1.isna().any():
            if kind == "pv":
                fallback = base.dropna().groupby(["hour"])["y"].median()
                miss2 = s1.index[s1.isna()]
                s1.loc[miss2] = [fallback.get(h, 0.0) for h in feat.loc[miss2, "hour"]]
            else:
                s1 = s1.fillna(base["y"].median())

        # sanity
        s1 = s1.clip(lower=0.0)
        if kind == "pv":
            night = (feat["hour"] <= 5) | (feat["hour"] >= 21)
            s1.loc[night] = 0.0

        out[col] = s1

    return out

# ---------- 3) Restore original wide-table column names and order ----------
def rebuild_wide(df_index: pd.DatetimeIndex,
                 pv_filled: pd.DataFrame,
                 load_filled: pd.DataFrame,
                 meta: dict) -> pd.DataFrame:
    """
    Return wide DataFrame with columns like PV_1, load_1, ...
    in the SAME order as the original file (meta["original_cols"]).
    """
    out = pd.DataFrame(index=df_index)

    # put PV_i and load_i back using original column names
    for hid in meta["hh_ids"]:
        out[meta["pv_map"][hid]] = pv_filled[f"H{hid}"].to_numpy()
        out[meta["load_map"][hid]] = load_filled[f"H{hid}"].to_numpy()

    # restore original column order (excluding time_col since it's index now)
    ordered_cols = [c for c in meta["original_cols"] if c != meta["time_col"]]

    # Some columns might be non-PV/load (if you had extra columns). Keep them if needed:
    # If you want to preserve extra columns from the original df, merge them before reordering.
    # Here we only output PV/load columns present in ordered_cols.
    existing = [c for c in ordered_cols if c in out.columns]
    out = out[existing]

    return out

# ---------- 4) End-to-end pipeline ----------
def process_and_save_same_format(input_path: str, output_path: str,
                                 short_gap_max_hours: int = 6,
                                 local_tz: str = None):
    raw_df, pv_df, load_df, meta = read_wide_pv_load(input_path)

    # Optional: convert to local time so "same-hour" grouping is more meaningful, then convert back to UTC before saving
    if local_tz is not None:
        pv_df = pv_df.tz_convert(local_tz)
        load_df = load_df.tz_convert(local_tz)

    pv_filled = fill_missing(pv_df, kind="pv", short_gap_max_hours=short_gap_max_hours)
    load_filled = fill_missing(load_df, kind="load", short_gap_max_hours=short_gap_max_hours)

    if local_tz is not None:
        pv_filled = pv_filled.tz_convert("UTC")
        load_filled = load_filled.tz_convert("UTC")

    wide_out = rebuild_wide(raw_df.index, pv_filled, load_filled, meta)

    # write out with time column as first column, like your original file
    wide_out_reset = wide_out.reset_index().rename(columns={"index": meta["time_col"]})

    if output_path.lower().endswith((".xlsx", ".xls")):
        wide_out_reset.to_excel(output_path, index=False)
    else:
        wide_out_reset.to_csv(output_path, index=False)

    return wide_out_reset

# ---------------- Example ----------------
if __name__ == "__main__":
    inp = "input_data/inputs.csv"           # change to your input file
    out = "input_data/inputs_filled.csv"   # output: still in the PV_1, load_1, ... format
    process_and_save_same_format(
        input_path=inp,
        output_path=out,
        short_gap_max_hours=6,
        local_tz="Europe/Zurich",  # set to None if the data are not in local time
    )
