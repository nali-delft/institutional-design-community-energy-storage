"""
results_analysis.py
===================
Local plotting script for energy community paper. Reads the model results from
outputs/ and writes the paper figures to figures/ (paths are resolved relative
to this file, so it can be run from any working directory):

    python results_analysis.py

Requirements:
    pip install matplotlib numpy pandas seaborn

Output:
    figures/Fig1_heatmap.pdf                      (paper Fig 1)
    figures/Fig2_market_access_m3_vs_m4.pdf       (paper Fig 2)
    figures/Fig3_third_party_revenue.pdf          (paper Fig 3)
    figures/Fig4_cv_vs_piloc.pdf                  (paper Fig 4)
    figures/Fig5_fairness_outcome.pdf             (paper Fig 5)
    figures/Fig6_ldc.pdf                          (paper Fig 6)

    figures/fig5_fairness_outcome_data.csv        (per-household savings data)
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# UNIFIED PAPER STYLE
# -----------------------------------------------------------------------------
def paper_style():
    """
    Unified rcParams for all figures.

    Use as:
        with paper_style():
            fig, ax = plt.subplots(...)
            ...

    Tuned for ~14-inch wide multi-panel paper figures.  Tick / label sizes
    are small enough to read but not so large that subplots overlap.
    """
    return plt.rc_context({
        # -- fonts --
        "font.family":     "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size":        14,
        "axes.titlesize":   12,
        "axes.titleweight": "bold",
        "axes.labelsize":   12,
        "xtick.labelsize":  11,
        "ytick.labelsize":  11,
        "legend.fontsize":  11,
        "legend.title_fontsize": 11,
        "figure.titlesize": 12,
        # -- lines & markers --
        "axes.linewidth":   0.9,
        "lines.linewidth":  1.8,
        "lines.markersize": 5.5,
        # -- grid --
        "grid.linewidth":   0.6,
        "grid.alpha":       0.25,
        "grid.linestyle":   "--",
        # -- legend --
        "legend.frameon":     False,
        "legend.handlelength": 1.8,
        # -- ticks --
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        # -- output --
        "figure.dpi":   150,
        "savefig.dpi":  300,
        "savefig.bbox": "tight",
        # -- math --
        "mathtext.fontset": "dejavusans",
    })


# -----------------------------------------------------------------------------
# CONFIGURATION (paths are resolved relative to this file)
# -----------------------------------------------------------------------------
# Resolve paths relative to THIS file so the script runs from any working
# directory (not only the repo root), consistent with config.py.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs"     # folder containing all model output CSVs
FIG_DIR  = BASE_DIR / "figures"     # figures are written here
FIG_DIR.mkdir(exist_ok=True)

# Network tariff archetype whose results are read and plotted. Output files are
# tagged with it, so switching this value switches the whole figure set.
# Set via set_tariff() or the --tariff command-line option.
from config import DEFAULTS as _DEF
TARIFF = _DEF.TARIFF


def set_tariff(name: str) -> None:
    """Select the network tariff archetype to load results for."""
    global TARIFF
    TARIFF = name


def _tag() -> str:
    """Filename suffix identifying the current archetype."""
    return f"_{TARIFF}"

MODES   = ["flat", "tou", "dynamic"]

# =========================================================
# Which configurations are meaningful in which price regime
# =========================================================
# Settings B and C grant the storage access to the market. That is only an
# institutional choice where prices vary: the spread between the highest and
# lowest hourly price is 0.54 CHF/kWh under dynamic pricing, 0.008 under
# time-of-use and exactly zero under a flat tariff. With no spread there is
# nothing to arbitrage, so Setting B collapses onto Setting A, and Setting C
# differs from it only through the export price, which is a change in feed-in
# remuneration rather than in market access. Setting A is behind-the-meter
# balancing and is NOT market access, so it is reported under every regime,
# where the price signal genuinely governs how the storage is used.
MARKET_ACCESS_CONFIGS = {"M3B", "M3C", "M4B", "M4C"}


def configs_for_mode(configs, mode):
    """Drop the market-access settings outside dynamic pricing."""
    if mode == "dynamic":
        return list(configs)
    return [c for c in configs if c not in MARKET_ACCESS_CONFIGS]


DESIGNS = ["A", "B", "C"]
TARIFF_LABEL = {"T0": "energy-only benchmark", "T1": "volumetric only",
                "T2": "capacity only", "T3": "volumetric + capacity"}
MODE_LABEL   = {"flat": "Flat", "tou": "ToU", "dynamic": "Dynamic"}
# One wording for the three market-access settings, used by every legend and
# by the caption line under the figures.
DESIGN_NAME = {"A": "Behind-the-meter only (no market access)",
               "B": "Import-only market access (partial market access)",
               "C": "Full market access"}
DESIGN_LABEL = {d: f"Setting {d}: {DESIGN_NAME[d]}" for d in ("A", "B", "C")}

# Community households
HH_ALL  = ["1", "5", "6", "7", "9", "10", "11"]
HH_PV   = {"1", "5", "6"}
HH_NOPV = {"7", "9", "10", "11"}

# Color palette
C = dict(
    M0="#461F1D", M1="#4C72B0", M2="#55A868",
    M3="#C44E52", M4="#8172B2",
    A="#4C72B0", B="#DD8452", C_col="#C44E52",
    pv="#C97B3D", nopv="#3D5A80",
    flat="#2171b5", tou="#fd8d3c", dynamic="#238443",
)

DPI  = 300
FIGW = 12      # default wide figure width
ANNOT_SIZE = 10  # font size for in-axes text annotations (delta tags, on-curve labels, mode headers)
CAPTION_SIZE = 12  # font size for figure-bottom captions (A/B/C design legend)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

DESIGN_CAPTION = "    ".join(f"{d}: {DESIGN_NAME[d]}" for d in ("A", "B", "C"))


def add_design_caption(fig: plt.Figure, y: float = 0.0) -> None:
    """Add the A/B/C design-setting legend across the bottom of the figure.

    On figures narrower than about 8 inches the one-line caption is wider
    than the axes, so it is broken into one line per setting instead.
    """
    text = DESIGN_CAPTION
    if fig.get_figwidth() < 8:
        text = "\n".join(p.strip() for p in DESIGN_CAPTION.split("    ") if p.strip())
    fig.text(0.5, y, text, ha="center", va="bottom",
             fontsize=CAPTION_SIZE, color="#000000", linespacing=1.4)


def savefig(fig: plt.Figure, name: str) -> None:
    stem = f"{name}{_tag()}"
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ok  {stem}")


def cv(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    return float(x.std() / abs(mu)) if not np.isclose(mu, 0) else np.nan


# cache: (tariff, model, mode, design) -> optimal pi_loc
_OPT_PLOC_CACHE: dict[tuple, float] = {}

def get_opt_ploc(model: str, mode: str, design: str = "") -> float:
    """
    Return the fairness-optimal (CV-minimizing) pi_loc for a given
    model x mode x design combination.

    The CV is computed on relative per-household savings:
        CV({s_i}) = std({s_i}) / |mean({s_i})|     where  s_i = (C_i^M0 - C_i^Mx) / |C_i^M0|

    This is the criterion stated in Methods.  Note: this is NOT
    equivalent to minimizing total community cost, especially for
    M4 where pi_loc shifts cost between households AND between the
    community and the third-party operator.

    Ties are broken by returning the smallest pi_loc, guaranteeing
    a deterministic, reproducible result.

    Results are cached so each CSV is read only once.
    """
    # The optimum differs between archetypes, so the archetype is part of the key.
    key = (TARIFF, model, mode, design)
    if key in _OPT_PLOC_CACHE:
        return _OPT_PLOC_CACHE[key]

    # Load summary file (which carries cv_savings if pre-computed)
    if model == "M1":
        summary = pd.read_csv(DATA_DIR / f"model1_summary_{mode}{_tag()}.csv")
    elif model == "M2":
        summary = pd.read_csv(DATA_DIR / f"model2_summary_{mode}{_tag()}.csv")
    elif model == "M3":
        summary = pd.read_csv(DATA_DIR / f"model3_summary_{mode}_{design}{_tag()}.csv")
    elif model == "M4":
        summary = pd.read_csv(DATA_DIR / f"model4_summary_{mode}_{design}{_tag()}.csv")
    else:
        raise ValueError(f"Unknown model: {model}")

    if "cv_savings" in summary.columns and summary["cv_savings"].notna().any():
        # Use the precomputed CV column from the summary file
        min_cv     = summary["cv_savings"].min()
        candidates = summary[np.isclose(summary["cv_savings"], min_cv, rtol=1e-9)]
    else:
        # Fall back: compute CV from per-household allocation files
        m0   = load_m0(mode)
        base = m0["cost"]
        if model == "M1":
            kpis = load_m1_kpis(mode)
            cv_rows = []
            for ploc, kpi in kpis.items():
                sav = (base - kpi["cost"]) / base.abs()
                sav = sav.dropna().to_numpy()
                cv  = float(np.std(sav, ddof=0) / abs(np.mean(sav))) \
                      if (len(sav) >= 2 and np.mean(sav) != 0) else np.nan
                cv_rows.append({"pi_loc": float(ploc), "cv_savings": cv})
            cv_df = pd.DataFrame(cv_rows)
        else:
            alloc = load_alloc(model.lower(), mode, design)
            cv_rows = []
            for ploc, sub in alloc.groupby("pi_loc"):
                sub = sub.set_index("household")
                sav = (base - sub["cost"]) / base.abs()
                sav = sav.dropna().to_numpy()
                cv  = float(np.std(sav, ddof=0) / abs(np.mean(sav))) \
                      if (len(sav) >= 2 and np.mean(sav) != 0) else np.nan
                cv_rows.append({"pi_loc": float(ploc), "cv_savings": cv})
            cv_df = pd.DataFrame(cv_rows)

        min_cv     = cv_df["cv_savings"].min()
        candidates = cv_df[np.isclose(cv_df["cv_savings"], min_cv, rtol=1e-9)]

    opt = float(candidates["pi_loc"].min())   # deterministic tie-break
    _OPT_PLOC_CACHE[key] = opt
    return opt


def get_opt_ploc_for_label(label: str, mode: str) -> float:
    """
    Convenience wrapper: derive model / design from label string
    (e.g. 'M3A', 'M4C', 'M1') and call get_opt_ploc.
    """
    if label == "M0":
        return np.nan
    if label in ("M1", "M2"):
        return get_opt_ploc(label, mode)
    # labels like 'M3A', 'M4C'
    model  = label[:2]   # 'M3' or 'M4'
    design = label[2:]   # 'A', 'B', 'C'
    return get_opt_ploc(model, mode, design)


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_m0(mode: str) -> pd.DataFrame:
    """
    Model 0 has no per-household kpis CSV.
    Reconstruct per-household annual costs from model0_flows_{mode}.csv.

    The flows file contains columns:
        cost__{hh}   - hourly cost per household (CHF)
        x_imp__{hh}  - hourly import per household (kWh)
        x_exp__{hh}  - hourly export per household (kWh)
        x_self__{hh} - hourly self-consumption (kWh)

    Returns a DataFrame indexed by household id (str) with columns:
        cost, annual_load, annual_pv, import, export, self_consumption
    """
    # Prefer the per-household KPI file: unlike the flows file, its `cost`
    # column includes the network charge, so it is the correct baseline once a
    # network tariff is applied.
    kpi_path = DATA_DIR / f"model0_kpis_{mode}{_tag()}.csv"
    if kpi_path.exists():
        k = pd.read_csv(kpi_path, index_col=0)
        k.index = k.index.astype(str)
        return k

    flows = pd.read_csv(DATA_DIR / f"model0_flows_{mode}{_tag()}.csv", index_col=0)

    # infer household ids from cost__ columns
    cost_cols = [c for c in flows.columns if c.startswith("cost__")]
    hh_ids    = [c.replace("cost__", "") for c in cost_cols]

    records = {}
    for hh in hh_ids:
        imp  = float(flows[f"x_imp__{hh}"].sum())
        exp  = float(flows[f"x_exp__{hh}"].sum())
        self_ = float(flows[f"x_self__{hh}"].sum())
        cost  = float(flows[f"cost__{hh}"].sum())
        records[hh] = {
            "cost":             cost,
            "annual_load":      imp + self_,   # load = import + self-consumption
            "annual_pv":        exp + self_,   # pv   = export + self-consumption
            "import":           imp,
            "export":           exp,
            "self_consumption": self_,
        }

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = None
    return df


def load_m1_kpis(mode: str) -> dict[float, pd.DataFrame]:
    """Return {pi_loc: kpi_df} for all available pi_loc files."""
    result = {}
    for f in sorted(DATA_DIR.glob(f"model1_kpis_{mode}{_tag()}_piLoc_*.csv")):
        # filename: model1_kpis_dynamic_piLoc_0_06.csv  -> 0.06
        stem  = f.stem                          # model1_kpis_dynamic_piLoc_0_06
        token = stem.split("piLoc_")[1]         # 0_06
        ploc  = float(token.replace("_", "."))
        df    = pd.read_csv(f, index_col=0)
        df.index = df.index.astype(str)
        result[ploc] = df
    return result


def load_alloc(model: str, mode: str, design: str = "") -> pd.DataFrame:
    """Load allocation CSV for m2 / m3 / m4."""
    if model == "m2":
        path = DATA_DIR / f"model2_allocation_by_piLoc_{mode}{_tag()}.csv"
    else:
        path = DATA_DIR / f"model{model[-1]}_allocation_by_piLoc_{mode}_{design}{_tag()}.csv"
    df = pd.read_csv(path)
    df["household"] = df["household"].astype(str)
    return df


def load_m4_summary(mode: str, design: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"model4_summary_{mode}_{design}{_tag()}.csv")


def load_community_flows(model_num: int, mode: str, design: str) -> pd.DataFrame:
    """Load optimized community-level flows (for LDC)."""
    path = DATA_DIR / f"model{model_num}_opt_community_{mode}_{design}{_tag()}.csv"
    df   = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def get_hh_costs(alloc: pd.DataFrame, ploc: float,
                 index: pd.Index) -> pd.Series:
    sub = alloc[np.isclose(alloc["pi_loc"], ploc)]
    return sub.set_index("household")["cost"].reindex(index)


# -----------------------------------------------------------------------------
# MASTER TABLE  (one row per model x mode x design x pi_loc)
# -----------------------------------------------------------------------------

def build_master() -> pd.DataFrame:
    """
    Columns: label, model, mode, design, pi_loc,
             total_cost, savings_pct,
             cv_savings, pv_mean_savings, nopv_mean_savings,
             third_party_revenue  (M4 only, else NaN)
    """
    rows = []

    for mode in MODES:
        m0      = load_m0(mode)
        base    = m0["cost"]
        base_tot = float(base.sum())

        # -- M0 --
        rows.append(_row("M0", "M0", mode, "-", np.nan,
                         base_tot, 0.0, np.nan, np.nan, np.nan, np.nan))

        # -- M1 --
        m1_kpis = load_m1_kpis(mode)
        for ploc, kpi in m1_kpis.items():
            c = kpi["cost"]
            s = (base - c) / base.abs()
            rows.append(_row("M1", "M1", mode, "-", ploc,
                             float(c.sum()),
                             (base_tot - c.sum()) / base_tot * 100,
                             cv(s.values),
                             float(s[[h for h in HH_ALL if h in HH_PV]].mean() * 100),
                             float(s[[h for h in HH_ALL if h in HH_NOPV]].mean() * 100),
                             np.nan))

        # -- M2 --
        m2 = load_alloc("m2", mode)
        for ploc, grp in m2.groupby("pi_loc"):
            c = grp.set_index("household")["cost"].reindex(base.index)
            s = (base - c) / base.abs()
            rows.append(_row("M2", "M2", mode, "-", float(ploc),
                             float(c.sum()),
                             (base_tot - c.sum()) / base_tot * 100,
                             cv(s.values),
                             float(s[[h for h in HH_ALL if h in HH_PV]].mean() * 100),
                             float(s[[h for h in HH_ALL if h in HH_NOPV]].mean() * 100),
                             np.nan))

        # -- M3 --  (Settings B and C exist under dynamic pricing only)
        for des in DESIGNS:
            if f"M3{des}" not in configs_for_mode([f"M3{des}"], mode):
                continue
            m3 = load_alloc("m3", mode, des)
            for ploc, grp in m3.groupby("pi_loc"):
                c = grp.set_index("household")["cost"].reindex(base.index)
                s = (base - c) / base.abs()
                rows.append(_row(f"M3{des}", "M3", mode, des, float(ploc),
                                 float(c.sum()),
                                 (base_tot - c.sum()) / base_tot * 100,
                                 cv(s.values),
                                 float(s[[h for h in HH_ALL if h in HH_PV]].mean() * 100),
                                 float(s[[h for h in HH_ALL if h in HH_NOPV]].mean() * 100),
                                 np.nan))

        # -- M4 --
        for des in DESIGNS:
            if f"M4{des}" not in configs_for_mode([f"M4{des}"], mode):
                continue
            m4      = load_alloc("m4", mode, des)
            m4_summ = load_m4_summary(mode, des)
            for ploc, grp in m4.groupby("pi_loc"):
                c   = grp.set_index("household")["cost"].reindex(base.index)
                s   = (base - c) / base.abs()
                sub = m4_summ[np.isclose(m4_summ["pi_loc"], float(ploc))]
                tpr = float(sub["third_party_revenue_total"].iloc[0]) if len(sub) else np.nan
                rows.append(_row(f"M4{des}", "M4", mode, des, float(ploc),
                                 float(c.sum()),
                                 (base_tot - c.sum()) / base_tot * 100,
                                 cv(s.values),
                                 float(s[[h for h in HH_ALL if h in HH_PV]].mean() * 100),
                                 float(s[[h for h in HH_ALL if h in HH_NOPV]].mean() * 100),
                                 tpr))

    df = pd.DataFrame(rows)
    df.to_csv(FIG_DIR / f"master_table{_tag()}.csv", index=False, float_format="%.4f")
    return df


def _row(label, model, mode, design, ploc,
         total_cost, savings_pct, cv_sav,
         pv_mean, nopv_mean, tpr):
    return dict(label=label, model=model, mode=mode, design=design,
                pi_loc=ploc, total_cost=total_cost, savings_pct=savings_pct,
                cv_savings=cv_sav, pv_mean_savings=pv_mean,
                nopv_mean_savings=nopv_mean, third_party_revenue=tpr)


def at_opt_ploc(master: pd.DataFrame) -> pd.DataFrame:
    """
    Filter master to the optimal pi_loc row for each (label, mode) combination.
    For M0 the pi_loc is NaN and is kept as-is.
    Replaces the old at_ploc(master, OPT_PLOC) calls.
    """
    kept = []
    for (label, mode), grp in master.groupby(["label", "mode"], sort=False):
        if label == "M0":
            kept.append(grp)
            continue
        opt = get_opt_ploc_for_label(label, mode)
        kept.append(grp[np.isclose(grp["pi_loc"], opt)])
    return pd.concat(kept, ignore_index=True)


# -----------------------------------------------------------------------------
# FIG 1 - Global heatmap  (3 x 9)
# -----------------------------------------------------------------------------

def _heatmap_pivot(master: pd.DataFrame) -> pd.DataFrame:
    """Savings (%) at the fairness-optimal price, rows = price regime, columns = configuration."""
    sub     = at_opt_ploc(master)
    configs = ["M0","M1","M2","M3A","M3B","M3C","M4A","M4B","M4C"]
    pivot = pd.DataFrame(index=MODES, columns=configs, dtype=float)
    for mode in MODES:
        for cfg in configs_for_mode(configs, mode):
            row = sub[(sub["label"] == cfg) & (sub["mode"] == mode)]
            if not row.empty:
                pivot.loc[mode, cfg] = float(row["savings_pct"].iloc[0])
    return pivot


def _draw_heatmap(ax, pivot: pd.DataFrame, *, group_labels: bool = True,
                  xlabels: bool = True, cbar_label: str | None = None,
                  group_y: float = -0.2):
    """One archetype's savings grid on `ax`, with its own color scale."""
    configs = list(pivot.columns)
    vals = pivot.values.astype(float)
    im   = ax.imshow(vals, cmap="YlGn", aspect="auto", vmin=0, vmax=np.nanmax(vals))
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs if xlabels else [], fontweight="bold")
    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels([MODE_LABEL[m] for m in MODES], fontweight="bold")
    for i in range(len(MODES)):
        for j in range(len(configs)):
            v = vals[i, j]
            if not np.isnan(v):
                txt   = "—" if np.isclose(v, 0) else f"{v:.1f}%"
                color = "white" if v > np.nanmax(vals) * 0.65 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontweight="bold")
    for xsep in [2.5, 5.5]:
        ax.axvline(xsep, color="white", linewidth=3)
    if group_labels:
        ax.annotate("No CES", xy=(1.0, group_y), xycoords=("data","axes fraction"),
                    ha="center", fontweight="bold", color="#555555", style="italic")
        ax.annotate("Community-owned CES (M3)", xy=(4.0, group_y), xycoords=("data","axes fraction"),
                    ha="center", fontweight="bold", color=C["M3"], style="italic")
        ax.annotate("Third-party-owned CES (M4)", xy=(7.0, group_y), xycoords=("data","axes fraction"),
                    ha="center", fontweight="bold", color=C["M4"], style="italic")
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    if cbar_label:
        cb.set_label(cbar_label)
    return im


def fig1_heatmap(master: pd.DataFrame) -> None:
    """Savings grid for the archetype currently selected (one panel)."""
    pivot = _heatmap_pivot(master)
    with paper_style():
        fig, ax = plt.subplots(figsize=(12, 4.5))
        _draw_heatmap(ax, pivot, cbar_label="Community cost savings vs. M0 (%)")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        add_design_caption(fig, y=0.0)
        savefig(fig, "Fig1_heatmap")
    pivot.to_csv(FIG_DIR / f"fig1_heatmap_data{_tag()}.csv", float_format="%.4f",
                 index_label="mode")


def fig1_heatmap_all(tariffs=("T0", "T1", "T2", "T3")) -> None:
    """
    Main-text Figure 1: the savings grid under every network tariff archetype,
    one panel per archetype stacked vertically. Each panel keeps its own color
    scale, because the no-charge case reaches 69 % and would otherwise wash out
    the contrasts within the three charging archetypes.
    """
    global TARIFF
    keep = TARIFF
    pivots = {}
    for t in tariffs:
        set_tariff(t); pivots[t] = _heatmap_pivot(build_master())
    set_tariff(keep)
    with paper_style():
        fig, axes = plt.subplots(len(tariffs), 1, figsize=(12, 3.0 * len(tariffs)))
        for k, (ax, t) in enumerate(zip(axes, tariffs)):
            last = (k == len(tariffs) - 1)
            _draw_heatmap(ax, pivots[t], group_labels=last, xlabels=True,
                          cbar_label="Savings vs. M0 (%)", group_y=-0.30)
            ax.set_title(f"{t}  {TARIFF_LABEL.get(t, t)}", fontweight="bold", loc="left", pad=6)
        fig.tight_layout(h_pad=1.6)
        fig.subplots_adjust(bottom=0.11)
        add_design_caption(fig, y=0.0)
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"Fig1_heatmap_all.{ext}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("  ok  Fig1_heatmap_all")
    pd.concat({t: p for t, p in pivots.items()}, names=["tariff", "mode"]).to_csv(
        FIG_DIR / "fig1_heatmap_all_data.csv", float_format="%.4f")


# -----------------------------------------------------------------------------
# FIG 2 - Market access value: M3 vs M4 across price regimes
# -----------------------------------------------------------------------------

def _draw_market_access(ax, master: pd.DataFrame, *, legend: bool = True) -> dict:
    """
    Draw the A -> B -> C progression for M3 and M4 on `ax`, under dynamic
    pricing, for the archetype currently selected. Returns the plotted values.

    Settings B and C are market access, and market access is an institutional
    choice only where prices vary (see MARKET_ACCESS_CONFIGS): under a flat or
    time-of-use tariff Setting B collapses onto Setting A and Setting C differs
    from it only through the export price, so the progression has no content
    there. Figure 1 applies the same rule.
    """
    sub  = at_opt_ploc(master)
    mode = "dynamic"

    def _vals(model_tag):
        out = []
        for des in DESIGNS:
            r = sub[(sub["label"] == f"{model_tag}{des}") & (sub["mode"] == mode)]
            out.append(float(r["savings_pct"].iloc[0]) if len(r) else np.nan)
        return out

    vals = {own: _vals(own) for own in ("M3", "M4")}
    y_top = max(v for arr in vals.values() for v in arr if not np.isnan(v)) * 1.12

    STYLE = {
        "M3": dict(marker="o", mfc=C["M3"], label="M3 (community-owned CES)"),
        "M4": dict(marker="^", mfc="white", label="M4 (third-party-owned CES)"),
    }
    for own in ("M3", "M4"):
        v, st = vals[own], STYLE[own]
        ax.plot(range(len(DESIGNS)), v, linestyle="-", marker=st["marker"],
                linewidth=2.0, markersize=8, color=C[own],
                markerfacecolor=st["mfc"], markeredgecolor=C[own],
                markeredgewidth=1.6, label=st["label"])

    ax.set_xlim(-0.2, len(DESIGNS) - 0.8)
    ax.set_ylim(0, y_top)
    ax.set_xticks(range(len(DESIGNS)))
    ax.set_xticklabels(DESIGNS, fontweight="bold")
    ax.set_ylabel("Community cost savings vs. M0 (%)")
    ax.grid(axis="y")
    if legend:
        ax.legend(loc="upper left")
    return vals


def _annotate_market_access(ax, vals: dict) -> None:
    """
    Increment labels for the A->B and B->C segments. Called after the figure
    layout is final, because the offset is computed perpendicular to each
    segment in display space and scaled to the label box, so that the box
    clears the line at any slope. M3 labels go on the upper side of the pair,
    M4 labels on the lower side; along the segment they are placed away from
    the midpoint, where the two lines tend to cross.
    """
    for own in ("M3", "M4"):
        v = vals[own]; other = vals["M4" if own == "M3" else "M3"]
        for k in range(1, len(DESIGNS)):
            if np.isnan(v[k]) or np.isnan(v[k-1]):
                continue
            inc  = v[k] - v[k-1]
            sign = "+" if inc >= 0 else "−"
            p0 = ax.transData.transform((k - 1, v[k-1])); p1 = ax.transData.transform((k, v[k]))
            d  = p1 - p0; L = np.hypot(*d) + 1e-9
            nrm = np.array([-d[1], d[0]]) / L
            if nrm[1] < 0:
                nrm = -nrm
            cos_t, sin_t = abs(d[0]) / L, abs(d[1]) / L
            # half box ~ 22 x 7 pt for a bold 10 pt "+34.5 pp"; keep a 6 pt margin
            dist = 7 * cos_t + 22 * sin_t + 6
            mid_own = (v[k] + v[k-1]) / 2; mid_other = (other[k] + other[k-1]) / 2
            above = mid_own >= mid_other if not np.isnan(mid_other) else (own == "M3")
            off = nrm * dist if above else -nrm * dist
            tpos = {("M3", 1): 0.40, ("M4", 1): 0.62, ("M3", 2): 0.70, ("M4", 2): 0.30}[(own, k)]
            xa = (k - 1) + tpos; ya = v[k-1] + tpos * (v[k] - v[k-1])
            ax.annotate(f"{sign}{abs(inc):.1f} pp",
                        xy=(xa, ya), xytext=tuple(off), textcoords="offset points",
                        ha="center", va="center", color=C[own],
                        fontsize=ANNOT_SIZE, fontweight="bold", zorder=10,
                        bbox=dict(fc="white", ec="none", alpha=1.0, pad=1.5))


def fig2_market_access(master: pd.DataFrame) -> None:
    """Marginal value of market access, A -> B -> C, for M3 and M4 (dynamic pricing)."""
    with paper_style():
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        vals = _draw_market_access(ax, master)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        fig.canvas.draw()
        _annotate_market_access(ax, vals)
        add_design_caption(fig, y=0.00)
        savefig(fig, "Fig2_market_access_m3_vs_m4")

    plot_rows = [{"owner": own, "mode": "dynamic", "design": des,
                  "config": f"{own}{des}", "savings_pct": v}
                 for own in ("M3", "M4") for des, v in zip(DESIGNS, vals[own])]
    pd.DataFrame(plot_rows).to_csv(
        FIG_DIR / f"fig2_market_access_data{_tag()}.csv",
        index=False, float_format="%.4f",
    )


def figS_market_access_by_tariff(tariffs=("T0", "T1", "T2", "T3")) -> None:
    """Supplementary: the Figure 2 panel under every network tariff archetype."""
    global TARIFF
    keep = TARIFF
    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
        panel_vals = {}
        for ax, t in zip(axes.ravel(), tariffs):
            set_tariff(t)
            panel_vals[t] = _draw_market_access(ax, build_master(), legend=(t == tariffs[0]))
            ax.set_title(f"{t}  {TARIFF_LABEL.get(t, t)}", fontweight="bold", pad=8)
        set_tariff(keep)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.11)
        fig.canvas.draw()
        for ax, t in zip(axes.ravel(), tariffs):
            _annotate_market_access(ax, panel_vals[t])
        add_design_caption(fig, y=0.00)
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"Fig2_market_access_all.{ext}", dpi=DPI,
                        bbox_inches="tight")
        plt.close(fig)
        print("  ok  Fig2_market_access_all")

# -----------------------------------------------------------------------------
# FIG 3 - Third-party revenue vs pi_loc  (M4 financial position across settings)
# -----------------------------------------------------------------------------

def _draw_third_party_revenue(ax, master: pd.DataFrame, *, legend_handles: bool = False):
    """
    Operator income statement per setting (A, B, C) under dynamic pricing, for
    the archetype currently selected:

        stacked bars   storage service revenue (+), market arbitrage margin
                       (+ or -), fixed fee (+, zero here) and the network
                       charge on the storage's own connection (-, net of the
                       clause-7 refund on discharge)
        black line     M4 - M3 community cost difference

    By the accounting identity  M4_cost - M3_cost = operator net revenue,
    the bars sum exactly to the line, so the operator's gain is visibly the
    community's loss relative to community ownership.

    Settings B and C are market access and are defined only where prices
    vary, so the panel is drawn for dynamic pricing only (Figures 1 and 2).
    Returns the arrays for the data file.
    """
    groups = [("dynamic", des) for des in DESIGNS]
    x = np.arange(len(groups))
    bw = 0.55

    C_SERV  = "#6BAED6"   # storage service fee
    C_MKT_P = "#F4A261"   # market margin (positive)
    C_MKT_N = "#A5A5A5"   # market margin (negative)
    C_FIX   = "#CCCCCC"   # fixed fee
    C_NET   = "#C44E52"   # network charge on the storage connection
    C_LINE  = "#222222"   # M4 - M3 community cost difference

    serv_vals, mkt_vals, fix_vals, net_vals, diff_vals = [], [], [], [], []
    for (mode, des) in groups:
        m4s  = load_m4_summary(mode, des)
        opt4 = get_opt_ploc("M4", mode, des)
        sub4 = m4s[np.isclose(m4s["pi_loc"], opt4)]
        if len(sub4):
            serv = float(sub4["storage_service_revenue"].iloc[0])
            mkt  = float(sub4["market_margin"].iloc[0])
            fix  = float(sub4.get("total_fixed_fee_alloc", pd.Series([0])).iloc[0])
            net  = float(sub4.get("third_party_network_cost", pd.Series([0])).iloc[0])
        else:
            serv, mkt, fix, net = np.nan, np.nan, 0.0, 0.0

        m4_row = master[(master["label"] == f"M4{des}") & (master["mode"] == mode)]
        m3_row = master[(master["label"] == f"M3{des}") & (master["mode"] == mode)]
        opt3   = get_opt_ploc("M3", mode, des)
        m4_at  = m4_row[np.isclose(m4_row["pi_loc"], opt4)]
        m3_at  = m3_row[np.isclose(m3_row["pi_loc"], opt3)]
        diff   = (float(m4_at["total_cost"].iloc[0]) - float(m3_at["total_cost"].iloc[0])
                  if len(m4_at) and len(m3_at) else np.nan)
        serv_vals.append(serv); mkt_vals.append(mkt); fix_vals.append(fix)
        net_vals.append(net);   diff_vals.append(diff)

    serv_arr = np.array(serv_vals, dtype=float)
    mkt_arr  = np.array(mkt_vals,  dtype=float)
    fix_arr  = np.array(fix_vals,  dtype=float)
    net_arr  = np.array(net_vals,  dtype=float)
    diff_arr = np.array(diff_vals, dtype=float)
    mkt_pos = np.where(mkt_arr >= 0, mkt_arr, 0.0)
    mkt_neg = np.where(mkt_arr <  0, mkt_arr, 0.0)

    # positive components stack upward from zero
    ax.bar(x, fix_arr, bw, color=C_FIX, alpha=0.85, edgecolor="white", linewidth=0.5,
           label="Fixed fee" if np.any(fix_arr != 0) else "_nolegend_")
    ax.bar(x, serv_arr, bw, bottom=fix_arr, color=C_SERV, alpha=0.85,
           edgecolor="white", linewidth=0.5, label="Storage service revenue")
    ax.bar(x, mkt_pos, bw, bottom=fix_arr + serv_arr, color=C_MKT_P, alpha=0.85,
           edgecolor="white", linewidth=0.5, label="Market arbitrage margin (+)")
    # negative components hang below zero
    ax.bar(x, mkt_neg, bw, color=C_MKT_N, alpha=0.85, edgecolor="white", linewidth=0.5,
           label="Market arbitrage margin (−)")
    ax.bar(x, -net_arr, bw, bottom=mkt_neg, color=C_NET, alpha=0.85,
           edgecolor="white", linewidth=0.5,
           label="Network charge on storage connection (−)")
    # net position = community cost difference
    ax.plot(x, diff_arr, color=C_LINE, marker="o", linewidth=1.6, markersize=8,
            zorder=10, label="Net = M4 − M3 community cost difference")

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([d for (_, d) in groups], fontweight="bold")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.1f}"))
    ax.set_ylabel("Operator financial position (kCHF/year)")
    ax.grid(axis="y", alpha=0.25)
    return dict(groups=groups, serv=serv_arr, mkt=mkt_arr, fix=fix_arr,
                net=net_arr, diff=diff_arr)


def fig3_third_party_revenue(master: pd.DataFrame) -> None:
    """Third-party operator income statement across settings (dynamic pricing)."""
    with paper_style():
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        d = _draw_third_party_revenue(ax, master)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
                  framealpha=0.95, borderaxespad=0.0)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.36)
        add_design_caption(fig, y=0.00)
        savefig(fig, "Fig3_third_party_revenue")

    plot_rows = [{"mode": mode, "design": des,
                  "storage_service_revenue": serv, "market_margin": mkt,
                  "fixed_fee": fix, "storage_connection_network_charge": net,
                  "m4_minus_m3_cost_diff": diff}
                 for (mode, des), serv, mkt, fix, net, diff
                 in zip(d["groups"], d["serv"], d["mkt"], d["fix"], d["net"], d["diff"])]
    pd.DataFrame(plot_rows).to_csv(
        FIG_DIR / f"fig3_third_party_revenue_data{_tag()}.csv",
        index=False, float_format="%.4f",
    )


def figS_third_party_revenue_by_tariff(tariffs=("T0", "T1", "T2", "T3")) -> None:
    """Supplementary: the Figure 3 panel under every network tariff archetype."""
    global TARIFF
    keep = TARIFF
    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
        for ax, t in zip(axes.ravel(), tariffs):
            set_tariff(t)
            _draw_third_party_revenue(ax, build_master())
            ax.set_title(f"{t}  {TARIFF_LABEL.get(t, t)}", fontweight="bold", pad=8)
        set_tariff(keep)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.045))
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)
        add_design_caption(fig, y=0.00)
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"Fig3_third_party_revenue_all.{ext}", dpi=DPI,
                        bbox_inches="tight")
        plt.close(fig)
        print("  ok  Fig3_third_party_revenue_all")

# -----------------------------------------------------------------------------
# FIG 5 - Distributional outcome: per-household savings at the CV-minimizing pi_loc
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# FIG 4 - CV vs pi_loc  (how the sharing price affects the dispersion of savings)
# -----------------------------------------------------------------------------

CV_CONFIGS = ["M1","M2","M3A","M3B","M3C","M4A","M4B","M4C"]
CV_STYLE = {   # linestyle, marker, color per configuration (shared by Figs 4, 6 and 7)
    "M1":  ("-",  "s", "#8c564b"), "M2":  ("-",  "^", "#2ca02c"),
    "M3A": ("-",  "o", "#1f77b4"), "M3B": ("-",  "D", "#9467bd"), "M3C": ("-",  "*", "#17becf"),
    "M4A": ("--", "o", "#d62728"), "M4B": ("--", "D", "#ff7f0e"), "M4C": ("--", "*", "#e377c2"),
}

def _draw_cv_panel(ax, master: pd.DataFrame, mode: str, *, ylabel: bool = True,
                   xlabel: bool = True, title: str | None = None) -> list:
    """CV of relative savings against the sharing price, one price regime, on `ax`."""
    rows = []
    for cfg in configs_for_mode(CV_CONFIGS, mode):
        sub = master[(master["label"] == cfg) & (master["mode"] == mode)]
        if sub.empty:
            continue
        ss = sub.sort_values("pi_loc").dropna(subset=["pi_loc", "cv_savings"])
        ls, mk, col = CV_STYLE[cfg]
        ax.plot(ss["pi_loc"], ss["cv_savings"], linestyle=ls, marker=mk,
                markersize=6 if mk != "*" else 9, color=col, alpha=0.9,
                linewidth=2.4, label=cfg)
        best = ss.loc[ss["cv_savings"].idxmin()]
        ax.scatter(best["pi_loc"], best["cv_savings"], marker="X", s=80, color=col,
                   zorder=6, edgecolors="white", linewidths=1.2)
        for pi, cv in zip(ss["pi_loc"], ss["cv_savings"]):
            rows.append({"mode": mode, "config": cfg, "pi_loc": pi, "cv_savings": cv,
                         "is_optimal": bool(np.isclose(pi, best["pi_loc"]))})
    ax.set_title(title if title is not None else MODE_LABEL[mode], fontweight="bold")
    ax.set_xticks(np.arange(0.06, 0.121, 0.01))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    if xlabel:
        ax.set_xlabel(r"Local sharing price  $\pi^{\mathrm{loc}}$  (CHF/kWh)")
    ax.grid()
    ax.set_ylim(bottom=0)
    if ylabel:
        ax.set_ylabel("CV of relative savings")
    return rows


def _cv_legend_handles():
    return [Line2D([], [], color=CV_STYLE[c][2], linestyle=CV_STYLE[c][0],
                   marker=CV_STYLE[c][1], markersize=6 if CV_STYLE[c][1] != "*" else 9,
                   linewidth=2.4, label=c) for c in CV_CONFIGS]


def fig4_cv_vs_piloc(master: pd.DataFrame) -> None:
    """Figure 4 for the archetype currently selected: 1 x 3 panels (Flat / ToU / Dynamic)."""
    plot_rows = []
    with paper_style():
        fig, axes = plt.subplots(1, len(MODES), figsize=(13, 5.5))
        for col, mode in enumerate(MODES):
            plot_rows += _draw_cv_panel(axes[col], master, mode, ylabel=(col == 0))
        fig.legend(handles=_cv_legend_handles(), loc="lower center", ncol=8,
                   frameon=False, bbox_to_anchor=(0.5, 0.085))
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.26)
        add_design_caption(fig, y=0.02)
        savefig(fig, "Fig4_cv_vs_piloc")
    pd.DataFrame(plot_rows).to_csv(
        FIG_DIR / f"fig4_cv_vs_piloc_data{_tag()}.csv", index=False, float_format="%.6f")


def fig4_cv_all(tariffs=("T0", "T1", "T2", "T3")) -> None:
    """Main-text Figure 4: rows = network tariff archetype, columns = price regime."""
    global TARIFF
    keep = TARIFF
    with paper_style():
        fig, axes = plt.subplots(len(tariffs), len(MODES), figsize=(13, 3.6 * len(tariffs)))
        for r, t in enumerate(tariffs):
            set_tariff(t); m = build_master()
            for c, mode in enumerate(MODES):
                last = (r == len(tariffs) - 1)
                _draw_cv_panel(axes[r][c], m, mode, ylabel=(c == 0), xlabel=last,
                               title=(MODE_LABEL[mode] if r == 0 else ""))
                if c == 0:
                    axes[r][c].set_ylabel("CV of relative savings")
                    axes[r][c].annotate(f"{t}  {TARIFF_LABEL.get(t, t)}", xy=(-0.30, 0.5),
                                        xycoords="axes fraction", rotation=90, ha="center",
                                        va="center", fontweight="bold")
        set_tariff(keep)
        fig.legend(handles=_cv_legend_handles(), loc="lower center", ncol=8,
                   frameon=False, bbox_to_anchor=(0.5, 0.040))
        fig.tight_layout(rect=(0.02, 0.0, 1.0, 1.0))
        fig.subplots_adjust(bottom=0.12, left=0.09)
        add_design_caption(fig, y=0.0)
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"Fig4_cv_all.{ext}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("  ok  Fig4_cv_all")


FAIR_CONFIGS = ["M1", "M2", "M3A", "M3B", "M3C", "M4A", "M4B", "M4C"]

def _fairness_data(master: pd.DataFrame):
    """Per-household savings (%) at each configuration's fairness-optimal price.

    Returns (plot_data, records): plot_data[(mode, cfg)] = {household: saving},
    records = flat rows for the CSV export.
    """
    plot_data, records = {}, []
    for mode in MODES:
        m0 = load_m0(mode); base = m0["cost"]
        for cfg in configs_for_mode(FAIR_CONFIGS, mode):
            opt = get_opt_ploc_for_label(cfg, mode)
            sub = master[(master["label"] == cfg) & (master["mode"] == mode)
                         & np.isclose(master["pi_loc"].fillna(-1), opt)]
            if sub.empty:
                continue
            model_tag = sub["model"].iloc[0]; design = sub["design"].iloc[0]
            if model_tag == "M1":
                hh_costs = load_m1_kpis(mode)[get_opt_ploc("M1", mode)]["cost"]
            elif model_tag == "M2":
                hh_costs = get_hh_costs(load_alloc("m2", mode), get_opt_ploc("M2", mode, ""), base.index)
            elif model_tag == "M3":
                hh_costs = get_hh_costs(load_alloc("m3", mode, design), get_opt_ploc("M3", mode, design), base.index)
            else:
                hh_costs = get_hh_costs(load_alloc("m4", mode, design), get_opt_ploc("M4", mode, design), base.index)
            sav = (base - hh_costs) / base.abs() * 100
            plot_data[(mode, cfg)] = {hh: float(sav.loc[hh]) for hh in HH_ALL
                                      if hh in sav.index and not np.isnan(sav.loc[hh])}
            for hh_id, sav_val in sav.dropna().items():
                is_pv = hh_id in HH_PV
                records.append({"mode": mode, "config": cfg, "model": model_tag, "design": design,
                                "pi_loc_optimal": float(opt), "household": hh_id, "is_pv": int(is_pv),
                                "group": "PV" if is_pv else "Non-PV", "cost_m0": float(base.loc[hh_id]),
                                "cost_mx": float(hh_costs.loc[hh_id]), "savings_pct": float(sav_val)})
    return plot_data, records


def _draw_fairness_panel(ax, plot_data, mode: str, *, ylabels: bool = True,
                         xlabel: bool = False, title: str | None = None):
    """Dumbbell of PV-mean and non-PV-mean savings per configuration, one regime, on `ax`."""
    configs = FAIR_CONFIGS; n_cfg = len(configs)
    y_for_cfg = {c: -i for i, c in enumerate(configs)}
    x_max = 0.0
    for cfg in configs:
        d = plot_data.get((mode, cfg), {})
        pv_vals  = [d[hh] for hh in HH_PV   if hh in d]
        npv_vals = [d[hh] for hh in HH_NOPV if hh in d]
        if not pv_vals or not npv_vals:
            continue
        y = y_for_cfg[cfg]
        pv_mean, np_mean = float(np.mean(pv_vals)), float(np.mean(npv_vals))
        x_max = max(x_max, pv_mean, np_mean)
        ax.plot([np_mean, pv_mean], [y, y], color="#9CA3AF", linewidth=2.0, alpha=0.6,
                zorder=2, solid_capstyle="round")
        ax.scatter(np_mean, y, s=110, color=C["nopv"], alpha=0.95, edgecolor="white",
                   linewidth=1.4, zorder=4)
        ax.scatter(pv_mean, y, s=110, color=C["pv"], alpha=0.95, edgecolor="white",
                   linewidth=1.4, zorder=4)
        delta = pv_mean - np_mean
        label = "Δ ≈ 0" if round(abs(delta)) == 0 else f"Δ {'+' if delta >= 0 else '−'}{abs(delta):.0f}"
        yfrac = (y - (-(n_cfg - 0.5))) / (0.5 - (-(n_cfg - 0.5)))
        ax.text(0.985, yfrac, label, transform=ax.transAxes, ha="right", va="center",
                fontweight="bold", fontsize=ANNOT_SIZE, color="#444444",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2.0))
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_yticks(list(y_for_cfg.values()))
    ax.set_yticklabels(configs if ylabels else [], fontweight="bold")
    ax.set_ylim(-(n_cfg - 0.2), 0.8)
    if title is not None:
        ax.set_title(title, fontweight="bold", pad=6)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    x_hi = max(x_max, 10.0)
    ax.set_xlim(-0.1 * x_hi - 2, 1.35 * x_hi)
    if xlabel:
        ax.set_xlabel("Per-household savings vs. M0 (%)")


def _fairness_legend_handles():
    return [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=C["nopv"],
               markeredgecolor="white", markeredgewidth=1.0, markersize=10,
               label="Non-PV households, mean"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=C["pv"],
               markeredgecolor="white", markeredgewidth=1.0, markersize=10,
               label="PV households, mean"),
        Line2D([0], [0], color="#888888", linewidth=2.0, label="Δ = PV mean − non-PV mean"),
    ]


def _export_fairness(records, path):
    if not records:
        return
    df = pd.DataFrame(records)[["mode", "config", "model", "design", "pi_loc_optimal",
                                "household", "group", "is_pv", "cost_m0", "cost_mx", "savings_pct"]]
    df = df.sort_values(["mode", "config", "is_pv", "household"],
                        ascending=[True, True, False, True]).reset_index(drop=True)
    df.to_csv(path, index=False, float_format="%.4f")


def fig5_fairness_outcome(master: pd.DataFrame) -> None:
    """Figure 5 for the archetype currently selected: 1 x 3 panels (Flat / ToU / Dynamic)."""
    plot_data, records = _fairness_data(master)
    with paper_style():
        fig, axes = plt.subplots(1, len(MODES), figsize=(12, 5.5))
        for col, mode in enumerate(MODES):
            _draw_fairness_panel(axes[col], plot_data, mode, ylabels=(col == 0),
                                 xlabel=(col == 1), title=MODE_LABEL[mode])
        fig.legend(handles=_fairness_legend_handles(), loc="upper center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, 1.005))
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
        add_design_caption(fig, y=0.02)
        savefig(fig, "Fig5_fairness_outcome")
    _export_fairness(records, FIG_DIR / f"fig5_fairness_outcome_data{_tag()}.csv")


def fig5_fairness_all(tariffs=("T0", "T1", "T2", "T3")) -> None:
    """Main-text Figure 5: rows = network tariff archetype, columns = price regime."""
    global TARIFF
    keep = TARIFF
    with paper_style():
        fig, axes = plt.subplots(len(tariffs), len(MODES), figsize=(12, 3.9 * len(tariffs)))
        for r, t in enumerate(tariffs):
            set_tariff(t); plot_data, _ = _fairness_data(build_master())
            for c, mode in enumerate(MODES):
                last = (r == len(tariffs) - 1)
                _draw_fairness_panel(axes[r][c], plot_data, mode, ylabels=(c == 0),
                                     xlabel=(last and c == 1),
                                     title=(MODE_LABEL[mode] if r == 0 else None))
            axes[r][0].annotate(f"{t}  {TARIFF_LABEL.get(t, t)}", xy=(-0.28, 0.5),
                                xycoords="axes fraction", rotation=90, ha="center", va="center",
                                fontweight="bold")
        set_tariff(keep)
        fig.legend(handles=_fairness_legend_handles(), loc="upper center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.98))
        add_design_caption(fig, y=0.0)
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"Fig5_fairness_all.{ext}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("  ok  Fig5_fairness_all")


# -----------------------------------------------------------------------------
# FIG 6 - Load duration curve
# -----------------------------------------------------------------------------

LDC_CONFIGS = [  # (label, model_num, design, color, linestyle, linewidth)
    ("M0",  0, None, "#333333", "-",  3.0),
    ("M1",  1, None, "#8c564b", "-",  2.6),
    ("M2",  2, None, "#2ca02c", "-",  2.6),
    ("M3A", 3, "A",  "#1f77b4", "-",  2.6),
    ("M3B", 3, "B",  "#9467bd", "-",  2.6),
    ("M3C", 3, "C",  "#17becf", "-",  2.6),
    ("M4A", 4, "A",  "#d62728", "--", 2.6),
    ("M4B", 4, "B",  "#ff7f0e", "--", 2.6),
    ("M4C", 4, "C",  "#e377c2", "--", 2.6),
]

def _ldc_series(label, model_num, design, mode):
    if label == "M0" or label == "M1":
        f = pd.read_csv(DATA_DIR / f"model{model_num}_flows_{mode}{_tag()}.csv", index_col=0, parse_dates=True)
        return f["total_x_imp"].values, f["total_x_exp"].values
    if label == "M2":
        f = pd.read_csv(DATA_DIR / f"model2_opt_flows_{mode}{_tag()}.csv")
        g = f.groupby("time")[["x_imp", "x_exp"]].sum()
        return g["x_imp"].values, g["x_exp"].values
    comm = load_community_flows(model_num, mode, design)
    return comm["x_imp"].values, comm["x_exp"].values


def _draw_ldc_panel(ax, mode: str, *, ylabel: bool = True, xlabel: bool = True,
                    title: str | None = None) -> list:
    """Load duration curves of grid import (+) and export (-), one regime, on `ax`."""
    rows = []
    hours = np.arange(1, 8761)
    for label, model_num, design, color, ls, lw in LDC_CONFIGS:
        if label not in configs_for_mode([label], mode):   # B and C: dynamic only
            continue
        try:
            x_imp, x_exp = _ldc_series(label, model_num, design, mode)
        except FileNotFoundError:
            print(f"  [warn] LDC: file not found for {label} {mode}, skipping"); continue
        imp_vals = np.sort(x_imp)[::-1]; exp_vals = -np.sort(x_exp)[::-1]
        ax.plot(hours, imp_vals, color=color, linewidth=lw, linestyle=ls, label=label, alpha=0.9)
        ax.plot(hours, exp_vals, color=color, linewidth=lw, linestyle=ls, alpha=0.9)
        rows += [{"mode": mode, "config": label, "hour": h, "import_kwh": a, "export_kwh": b}
                 for h, a, b in zip(hours, imp_vals, exp_vals)]
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    if title is not None:
        ax.set_title(title, fontweight="bold")
    if xlabel:
        ax.set_xlabel("Hours (sorted descending)")
    ax.grid()
    if ylabel:
        ax.set_ylabel("Load duration curve (kWh/h)\n← export  |  import →")
    return rows


def _ldc_legend_handles():
    return [Line2D([], [], color=c, linestyle=ls, linewidth=lw, label=lab)
            for lab, _, _, c, ls, lw in LDC_CONFIGS]


def fig6_ldc() -> None:
    """Load duration curves for the archetype currently selected: 1 x 3 panels."""
    plot_rows = []
    with paper_style():
        fig, axes = plt.subplots(1, len(MODES), figsize=(FIGW, 5), sharey=True)
        for col, mode in enumerate(MODES):
            plot_rows += _draw_ldc_panel(axes[col], mode, ylabel=(col == 0), title=MODE_LABEL[mode])
        # legend in the first panel, whose upper right is empty under a flat tariff
        axes[0].legend(handles=_ldc_legend_handles(), loc="upper right", ncol=2,
                       framealpha=0.95, columnspacing=1.0, handlelength=1.8)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        add_design_caption(fig, y=0.02)
        savefig(fig, "Fig6_ldc")
    pd.DataFrame(plot_rows).to_csv(FIG_DIR / f"fig6_ldc_data{_tag()}.csv", index=False, float_format="%.6f")


def fig6_ldc_all(tariffs=("T0", "T1", "T2", "T3")) -> None:
    """Main-text LDC figure: rows = network tariff archetype, columns = price regime."""
    global TARIFF
    keep = TARIFF
    with paper_style():
        fig, axes = plt.subplots(len(tariffs), len(MODES), figsize=(FIGW, 3.4 * len(tariffs)),
                                 sharey=True)
        for r, t in enumerate(tariffs):
            set_tariff(t)
            for c, mode in enumerate(MODES):
                last = (r == len(tariffs) - 1)
                _draw_ldc_panel(axes[r][c], mode, ylabel=(c == 0), xlabel=last,
                                title=(MODE_LABEL[mode] if r == 0 else None))
                if c == 0:
                    axes[r][c].set_ylabel("← export  |  import →  (kWh/h)")
                    axes[r][c].annotate(f"{t}  {TARIFF_LABEL.get(t, t)}", xy=(-0.30, 0.5),
                                        xycoords="axes fraction", rotation=90, ha="center",
                                        va="center", fontweight="bold")
        set_tariff(keep)
        axes[0][0].legend(handles=_ldc_legend_handles(), loc="upper right", ncol=2,
                          framealpha=0.95, columnspacing=1.0, handlelength=1.8, fontsize=9)
        fig.tight_layout(rect=(0.02, 0.0, 1.0, 1.0))
        fig.subplots_adjust(bottom=0.08, left=0.10)
        add_design_caption(fig, y=0.0)
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"Fig6_ldc_all.{ext}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("  ok  Fig6_ldc_all")


# -----------------------------------------------------------------------------
# SHARED HELPER
# -----------------------------------------------------------------------------

def _set_group_xticks(ax, groups):
    """Set x-ticks as 'Flat-A', 'Flat-B', ... with color-coded mode labels."""
    n = len(groups)
    ax.set_xticks(range(n))
    labels = []
    for g in groups:
        mode, des = g.split("-")
        labels.append(f"{MODE_LABEL[mode]}\n{des}")
    ax.set_xticklabels(labels)
    # draw mode separators
    for sep in [2.5, 5.5]:
        ax.axvline(sep, color="#CCCCCC", linewidth=1.5, linestyle="--")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main(tariff: str | None = None):
    if tariff is not None:
        set_tariff(tariff)
    print("=" * 60)
    print(f"  Results plotting   [tariff {TARIFF}]")
    print("=" * 60)

    print("\n[1/7] Building master table ...")
    master = build_master()
    print(f"      {len(master)} rows x {len(master.columns)} columns")

    # Each figN_* writes figures/FigN_*<tag>.pdf; see README.md for the mapping
    # between these file names and the figure numbers of the paper.
    print("\n[2/7] Fig1_heatmap ...")
    fig1_heatmap(master)

    print("\n[3/7] Fig2_market_access_m3_vs_m4 ...")
    fig2_market_access(master)

    print("\n[4/7] Fig3_third_party_revenue ...")
    fig3_third_party_revenue(master)

    print("\n[5/7] Fig4_cv_vs_piloc ...")
    fig4_cv_vs_piloc(master)

    print("\n[6/7] Fig5_fairness_outcome ...")
    fig5_fairness_outcome(master)

    print("\n[7/7] Fig6_ldc ...")
    fig6_ldc()

    print(f"\nAll {TARIFF} figures saved to ./{FIG_DIR}/")


def main_all(tariffs=None):
    """Render the full figure set once per network tariff archetype, then the
    composite figures that place the four archetypes side by side. The
    composites are the versions used in the paper (see README.md)."""
    from config import TARIFF_GRID
    for t in (tariffs or TARIFF_GRID):
        main(t)
    print("\nComposite figures across the four archetypes ...")
    fig1_heatmap_all()
    figS_market_access_by_tariff()        # -> Fig2_market_access_all
    figS_third_party_revenue_by_tariff()  # -> Fig3_third_party_revenue_all
    fig4_cv_all()
    fig5_fairness_all()
    fig6_ldc_all()
    print(f"\nAll figures saved to ./{FIG_DIR}/")


if __name__ == "__main__":
    main_all()