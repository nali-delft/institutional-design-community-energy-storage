"""
fig_tariffs.py
==============
Cross-archetype figures covering the network tariff dimension.

    Fig 7   community cost and grid peak across the four tariff archetypes
    Fig 8   sensitivity of the ranking to the LEG reduction rate delta

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
# Figure 7 - cost and peak across archetypes
# =========================================================
def fig7_tariff_comparison(y_cap: float = 40.0) -> None:
    """
    Community cost savings for every configuration under the four archetypes,
    one panel per price regime.

    Layout choices. Panels are sized in proportion to the configurations they
    show (Settings B and C appear under dynamic pricing only, see
    MARKET_ACCESS_CONFIGS), and M0 is omitted since it is the reference and
    zero by construction. The y-axis is shared and capped at `y_cap` so that
    the bulk of the bars, which lie between 4 and 35 %, remain readable; the
    single bar above the cap (M3C without a network charge, 69.4 %) is drawn
    to the cap with a break marker and its value printed above it.
    """
    frames = []
    for t in TARIFF_GRID:
        d = _at_opt(_master(t)).assign(tariff=t)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    panel_labels = {m: [l for l in labels_for_mode(m) if l != "M0"] for m in MODES}
    widths = [len(panel_labels[m]) for m in MODES]

    with paper_style():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True,
                                 gridspec_kw=dict(width_ratios=widths, wspace=0.08))
        w = 0.2
        for ax, mode in zip(axes, MODES):
            sub = df[df["mode"] == mode]
            labels = panel_labels[mode]
            x = np.arange(len(labels))
            for k, t in enumerate(TARIFF_GRID):
                s = sub[sub.tariff == t].set_index("label").reindex(labels)
                vals = s["savings_pct"].to_numpy(dtype=float)
                shown = np.minimum(vals, y_cap)
                bars = ax.bar(x + (k - 1.5) * w, shown, w,
                              label=TARIFF_LABEL[t] if mode == MODES[0] else None,
                              color=TARIFF_COLOR[t], edgecolor="white", linewidth=0.4)
                # bars above the cap: break marker and value label
                for xi, v, b in zip(x + (k - 1.5) * w, vals, bars):
                    if np.isfinite(v) and v > y_cap:
                        for dy in (-1.6, -0.6):
                            ax.plot([xi - 0.55 * w, xi + 0.55 * w],
                                    [y_cap + dy - 0.5, y_cap + dy + 0.5],
                                    color="white", lw=2.2, zorder=5, clip_on=False)
                            ax.plot([xi - 0.55 * w, xi + 0.55 * w],
                                    [y_cap + dy - 0.5, y_cap + dy + 0.5],
                                    color=TARIFF_COLOR[t], lw=0.9, zorder=6, clip_on=False)
                        ax.annotate(f"{v:.1f}", xy=(xi, y_cap), xytext=(0, 5),
                                    textcoords="offset points", ha="center", va="bottom",
                                    fontsize=ANNOT_SIZE - 1, fontweight="bold",
                                    color=TARIFF_COLOR[t], clip_on=False)
            ax.axhline(0, color="0.3", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0)
            ax.set_xlim(-0.6, len(labels) - 0.4)
            ax.set_title(MODE_LABEL[mode])
            ax.grid(axis="y")
            # light separators between the ownership groups
            for lab_a, lab_b in (("M2", "M3A"), ("M3C", "M4A"), ("M3A", "M4A")):
                if lab_a in labels and lab_b in labels and labels.index(lab_b) == labels.index(lab_a) + 1:
                    ax.axvline(labels.index(lab_a) + 0.5, color="0.85", lw=0.8, zorder=0)
        axes[0].set_ylim(0, y_cap * 1.08)
        axes[0].set_ylabel("Community cost savings vs. M0 (%)")
        fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), frameon=False)
        fig.tight_layout()
        savefig(fig, "Fig7_tariff_comparison")

    df.to_csv(FIG_DIR / "fig7_tariff_comparison_data.csv", index=False, float_format="%.4f")


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


# =========================================================
# Figure 8 - distributional outcomes across archetypes
# =========================================================
def fig8_distribution() -> None:
    """
    How the network tariff shapes the distribution of benefits, not only its size.

    Two panels under dynamic pricing: the dispersion of household savings, and
    the gap between generating and non-generating households. Fig. 7 reports the
    same configurations by total saving; this reports who receives it.
    """
    frames = []
    for t in TARIFF_GRID:
        d = _at_opt(_master(t)).assign(tariff=t)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["mode"] == "dynamic") & (df["label"] != "M0")]
    df["pv_gap"] = df["pv_mean_savings"] - df["nopv_mean_savings"]

    labs = [l for l in LABELS if l != "M0"]
    panels = [("cv_savings", "CV of household savings", None),
              ("pv_gap", "PV minus non-PV mean saving (%)", 0.0)]

    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        x = np.arange(len(labs)); w = 0.2
        for ax, (col, ylab, zero) in zip(axes, panels):
            for k, t in enumerate(TARIFF_GRID):
                sdf = df[df.tariff == t].set_index("label").reindex(labs)
                ax.bar(x + (k - 1.5) * w, sdf[col], w,
                       label=TARIFF_LABEL[t] if col == "cv_savings" else None,
                       color=TARIFF_COLOR[t], edgecolor="white", linewidth=0.4)
            if zero is not None:
                ax.axhline(zero, color="0.3", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels(labs, rotation=45, ha="right")
            ax.set_ylabel(ylab); ax.grid(axis="y")
        axes[0].set_title("Coefficient of variation of household savings")
        axes[1].set_title("Gap between PV and non-PV households")
        fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
        # figure title omitted: the caption identifies the figure
        # fig.suptitle("Distribution of community benefits under four network tariff archetypes "
                     # "(dynamic pricing)", y=1.02)
        fig.tight_layout()
        savefig(fig, "Fig8_distribution")

    df[["label", "tariff", "pi_loc", "savings_pct", "cv_savings",
        "pv_mean_savings", "nopv_mean_savings", "pv_gap"]].to_csv(
        FIG_DIR / "fig8_distribution_data.csv", index=False, float_format="%.4f")


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
    print("Fig7_tariff_comparison ...")
    fig7_tariff_comparison()
    print("Fig8_distribution ...")
    fig8_distribution()
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
