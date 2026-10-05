"""
fig_tariffs.py
==============
Cross-archetype figures covering the network tariff dimension.

    Fig8_delta_sensitivity   sensitivity of community savings to the statutory
                             network charge reduction rate delta (paper Figure 6)
    Fig9_sizing              storage duration and power rating (paper Figure 8)

Both read the per-archetype master tables written by results_analysis.py.
Run results_analysis.main_all() first so every archetype is available.

    python fig_tariffs.py
"""
from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import TARIFF_GRID, DELTA_GRID
import results_analysis
from results_analysis import (
    FIG_DIR, DATA_DIR, MODES, MODE_LABEL, C, paper_style, set_tariff, DPI, ANNOT_SIZE,
    DESIGN_LABEL, get_opt_ploc_for_label, CV_STYLE,
)


def savefig(fig, name: str) -> None:
    """Cross-archetype figures span all four tariffs, so they carry no tag."""
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ok  {name}")
from tariffs import ARCHETYPES

LABELS = ["M0", "M1", "M2", "M3A", "M3B", "M3C", "M4A", "M4B", "M4C"]

# Settings B and C grant market access, which is only an institutional choice
# where prices vary (spread 0.54 CHF/kWh dynamic, 0.008 TOU, 0.000 flat). They
# are therefore shown under dynamic pricing only. Setting A is behind-the-meter
# balancing, not market access, and is shown under every regime.
MARKET_ACCESS_CONFIGS = {"M3B", "M3C", "M4B", "M4C"}


def labels_for_mode(mode):
    return LABELS if mode == "dynamic" else [l for l in LABELS if l not in MARKET_ACCESS_CONFIGS]

TARIFF_COLOR = {"T0": "#461F1D", "T1": "#4C72B0", "T2": "#C44E52", "T3": "#55A868"}
TARIFF_LABEL = {
    "T0": "T0  energy-only benchmark",
    "T1": "T1  volumetric only",
    "T2": "T2  capacity only",
    "T3": "T3  volumetric + capacity",
}


def _master(tariff: str) -> pd.DataFrame:
    p = FIG_DIR / f"master_table_{tariff}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run results_analysis.main_all() to build every "
            f"archetype's master table first."
        )
    return pd.read_csv(p)


def _at_opt(df: pd.DataFrame) -> pd.DataFrame:
    """One row per label x mode, taken at the CV-minimizing sharing price."""
    sub = df.dropna(subset=["cv_savings"])
    idx = sub.groupby(["label", "mode"])["cv_savings"].idxmin()
    keep = df.loc[idx]
    m0 = df[df.label == "M0"].drop_duplicates(["label", "mode"])
    return pd.concat([m0, keep]).drop_duplicates(["label", "mode"])


# =========================================================
# Figure 8 - sensitivity to the LEG reduction rate
# =========================================================
def _row_at_opt_ploc(df: pd.DataFrame, label: str, mode: str, tariff: str) -> pd.Series:
    """Summary row at the CV-minimizing local sharing price, as in Figures 1-5."""
    keep = results_analysis.TARIFF
    set_tariff(tariff)
    try:
        opt = get_opt_ploc_for_label(label, mode)
    finally:
        set_tariff(keep)
    hit = df[np.isclose(df["pi_loc"], opt)]
    return hit.iloc[0] if not hit.empty else df.iloc[0]


def _summary(model: str, mode: str, design: str, tariff: str) -> pd.DataFrame:
    stem = (f"model{model}_summary_{mode}_{design}_{tariff}.csv" if design
            else f"model{model}_summary_{mode}_{tariff}.csv")
    p = DATA_DIR / stem
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def fig8_delta_sensitivity(deltas=DELTA_GRID, tariffs=("T1", "T2", "T3")) -> None:
    """
    Sensitivity of community cost to the statutory network-fee reduction.

    The reduction enters ex-post settlement and is linear in delta, so the whole
    sweep is obtained exactly by rescaling the reduction computed at the
    calibration run. No additional dispatch is required.
    """
    specs = [("1", ""), ("2", ""), ("3", "A"), ("3", "B"), ("3", "C"),
             ("4", "A"), ("4", "B"), ("4", "C")]
    rows = []
    for t in tariffs:
        for mode in MODES:
            m0 = _summary("0", mode, "", t)
            base_cost = float(m0["total_cost"].iloc[0]) if not m0.empty else np.nan
            for num, des in specs:
                label = f"M{num}{des}"
                # Settings B and C are market access, which is an institutional
                # choice only where prices vary; report them under dynamic
                # pricing only, as in Figures 1, 2 and 3.
                if mode != "dynamic" and label in MARKET_ACCESS_CONFIGS:
                    continue
                df = _summary(num, mode, des, t)
                if df.empty:
                    continue
                r = _row_at_opt_ploc(df, label, mode, t)
                d0 = float(r.get("delta_leg", 0.0) or 0.0)
                if "total_network_reduction" in df.columns:
                    red0 = float(r.get("total_network_reduction", 0.0) or 0.0)
                else:
                    # The reduction accrues to household withdrawals, which are
                    # identical under community and third-party ownership of the
                    # same storage (same dispatch, same flows); the model-4
                    # summary does not repeat it, so take it from model 3.
                    df3 = _summary("3", mode, des, t)
                    if df3.empty or "total_network_reduction" not in df3.columns:
                        continue
                    r3 = _row_at_opt_ploc(df3, f"M3{des}", mode, t)
                    d0 = float(r3.get("delta_leg", 0.0) or 0.0)
                    red0 = float(r3.get("total_network_reduction", 0.0) or 0.0)
                if d0 == 0:
                    continue
                per_unit = red0 / d0
                for d in deltas:
                    cost = float(r["total_cost"]) - (per_unit * d - red0)
                    rows.append(dict(tariff=t, label=label, mode=mode, delta=d,
                                     cost=cost,
                                     savings_pct=(base_cost - cost) / base_cost * 100
                                     if np.isfinite(base_cost) and base_cost else np.nan))
    if not rows:
        print("  !  no delta data available; run the models with a tariff first")
        return
    df = pd.DataFrame(rows)

    order = ["M1", "M2", "M3A", "M3B", "M3C", "M4A", "M4B", "M4C"]
    # one color and line style per configuration, shared with Figures 4 and 7,
    # so that a line means the same thing in every figure
    palette = {lab: CV_STYLE[lab][2] for lab in order}
    styles = {lab: CV_STYLE[lab][0] for lab in order}
    with paper_style():
        fig, axes = plt.subplots(len(tariffs), 3, figsize=(14, 4.2 * len(tariffs)),
                                 sharex=True, squeeze=False)
        for i, t in enumerate(tariffs):
            for j, mode in enumerate(MODES):
                ax = axes[i][j]
                sub = df[(df.tariff == t) & (df["mode"] == mode)]
                for lab in order:
                    sl = sub[sub.label == lab].sort_values("delta")
                    if sl.empty:
                        continue
                    ax.plot(sl["delta"], sl["savings_pct"], marker="o", ms=5, lw=2.4,
                            color=palette[lab], linestyle=styles[lab], label=lab)
                ax.grid(True)
                ax.axhline(0, color="0.4", lw=0.8)
                if i == 0:
                    ax.set_title(MODE_LABEL[mode])
                if j == 0:
                    ax.set_ylabel("Community cost savings vs. M0 (%)")
                    ax.annotate(TARIFF_LABEL[t], xy=(-0.24, 0.5), xycoords="axes fraction",
                                rotation=90, ha="center", va="center", fontweight="bold")
                if i == len(tariffs) - 1:
                    ax.set_xlabel(r"Statutory network-fee reduction $\delta$")
        axes[0][-1].legend(ncol=2, fontsize=9, loc="upper left")
        fig.tight_layout(rect=(0.02, 0.0, 1.0, 1.0))
        fig.subplots_adjust(left=0.09)
        savefig(fig, "Fig8_delta_sensitivity")

    df.to_csv(FIG_DIR / "fig8_delta_sensitivity_data.csv", index=False, float_format="%.4f")


def main() -> None:
    print("Fig8_delta_sensitivity ...")
    fig8_delta_sensitivity()
    print("Fig9_sizing ...")
    fig9_sizing()
    print(f"done -> {FIG_DIR}")



# =========================================================
# Figure 9 - storage duration and power rating
# =========================================================
DESIGN_STYLE = {"A": ("#4C72B0", "o", DESIGN_LABEL["A"]),
                "B": ("#DD8452", "s", DESIGN_LABEL["B"]),
                "C": ("#C44E52", "^", DESIGN_LABEL["C"])}


def fig9_sizing(mode: str = "dynamic") -> None:
    """
    Community savings and grid peak as the storage duration is varied, with the
    energy capacity held at one day of average consumption. Because E is fixed,
    varying E/P is equivalent to varying the power rating, shown on the upper
    axis. One column per network tariff archetype: the four differ not only in
    where the optimum lies but in whether an interior optimum exists at all.
    """
    p = DATA_DIR / "sweep_sizing.csv"
    if not p.exists():
        print("  !  outputs/sweep_sizing.csv not found; run sweep_sizing.py first")
        return
    d = pd.read_csv(p)
    d = d[d["mode"] == mode]

    tariffs = list(TARIFF_GRID)
    with paper_style():
        fig, axes = plt.subplots(2, len(tariffs), figsize=(15, 7.2),
                                 sharex=True, squeeze=False)
        for ax in axes[1][1:]:
            ax.sharey(axes[1][0])
        for j, t in enumerate(tariffs):
            sub = d[d.tariff == t]
            ax_s, ax_p = axes[0][j], axes[1][j]

            for des, (col, mk, lab) in DESIGN_STYLE.items():
                s = sub[sub.design == des].sort_values("ep_hours")
                ax_s.plot(s["ep_hours"], s["m3_savings_pct"], color=col, marker=mk,
                          lw=2.4, ms=7, label=lab if j == 0 else None)
                ax_p.plot(s["ep_hours"], s["peak_import_kw"], color=col, marker=mk,
                          lw=2.4, ms=7)

            ax_s.axhline(0, color="0.3", lw=0.8)
            ax_s.set_title(TARIFF_LABEL[t], pad=26)
            for ax in (ax_s, ax_p):
                ax.grid(True)
                ax.set_xticks(sorted(d["ep_hours"].unique()))
            ax_p.set_xlabel("Storage duration $E/P$  (h)")

            # power rating on the upper axis of the top row
            top = ax_s.secondary_xaxis("top")
            hrs = sorted(d["ep_hours"].unique())
            top.set_xticks(hrs)
            top.set_xticklabels([f"{d['E_kwh'].iloc[0]/h:.0f}" for h in hrs], fontsize=9)
            # The corresponding power rating is stated in the caption rather
            # than labeled per panel, which would displace the panel titles.

        axes[0][0].set_ylabel("Community cost savings\nvs. M0 (%)")
        axes[1][0].set_ylabel("Peak grid import  (kW)")
        fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), frameon=False,
                   columnspacing=1.5)
        # figure title omitted: the caption identifies the figure
        # fig.suptitle(f"Storage sizing under four network tariff archetypes "
                     # f"({MODE_LABEL[mode].lower()} pricing, community-owned storage)", y=1.04)
        fig.tight_layout()
        savefig(fig, "Fig9_sizing")

    d.to_csv(FIG_DIR / "fig9_sizing_data.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
