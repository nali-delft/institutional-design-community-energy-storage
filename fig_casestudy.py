"""
fig_casestudy.py
================
Descriptive figure of the case study: what the seven metered units look like
before any institutional arrangement is imposed on them.

The point the figure has to make is that the units are heterogeneous by design.
NEST is a research building whose modules serve different purposes -- dwellings
of two to four occupants, a combined work-and-living unit, an office unit and a
fitness and wellness unit -- so annual consumption spans a factor of seven and
only three of the seven carry photovoltaics. That spread is the raw material an
energy community works on, and it is also why the distributional results are not
an artefact of an artificially uniform community.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PATHS
from results_analysis import paper_style, FIG_DIR, DPI
from utils import read_inputs_filled, build_con_gen_matrices

LOAD_C, PV_C = "#4C72B0", "#DD8452"


def main() -> None:
    df, unit_ids = read_inputs_filled(PATHS.INPUTS_FILLED)
    con, gen = build_con_gen_matrices(df, unit_ids)

    ann = pd.DataFrame({"load": con.sum() / 1000.0, "pv": gen.sum() / 1000.0})
    # Use presentation labels rather than the building operator's internal IDs.
    ann.index = pd.Index(range(1, len(ann) + 1), name="unit")
    idx = con.index
    loc = idx.tz_convert("Europe/Zurich") if idx.tz is not None else idx
    month = pd.Series(loc.month, index=idx)
    m_load = con.sum(axis=1).groupby(month).sum() / 1000.0
    m_pv = gen.sum(axis=1).groupby(month).sum() / 1000.0

    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

        ax = axes[0]
        x = np.arange(len(ann)); w = 0.38
        ax.bar(x - w / 2, ann["load"], w, label="Annual consumption", color=LOAD_C)
        ax.bar(x + w / 2, ann["pv"], w, label="Annual PV generation", color=PV_C)
        ax.set_xticks(x); ax.set_xticklabels([f"Unit {u}" for u in ann.index], rotation=45, ha="right")
        # No artificial padding on the x-axis: the axis ends just past the last
        # bar, and the legend is given room vertically instead.
        ax.set_xlim(-0.7, len(ann) - 0.3)
        ax.set_ylabel("MWh per year"); ax.grid(axis="y")
        ax.set_title("Per-unit consumption and generation")
        # headroom so the legend clears the tallest bar
        ax.set_ylim(0, ann.values.max() * 1.28)
        ax.legend(frameon=False, loc="upper right", borderaxespad=0.6, fontsize=9)

        ax = axes[1]
        xm = np.arange(1, 13)
        ax.bar(xm - 0.19, m_load.reindex(xm).values, 0.38, label="Community consumption", color=LOAD_C)
        ax.bar(xm + 0.19, m_pv.reindex(xm).values, 0.38, label="Community PV generation", color=PV_C)
        ax.set_xticks(xm)
        ax.set_xticklabels(["1","2","3","4","5","6","7","8","9","10","11","12"])
        ax.set_ylabel("MWh per month"); ax.grid(axis="y")
        ax.set_ylim(0, max(m_load.max(), m_pv.max()) * 1.34)
        ax.set_title("Community seasonal profile")
        ax.legend(frameon=False, loc="upper right", borderaxespad=0.6, fontsize=9)

        fig.tight_layout()
        # The case study is independent of the network tariff, so the file
        # carries no archetype tag (paper Figure 9).
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"Fig0_casestudy.{ext}", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("  ok  Fig0_casestudy")

    out = ann.copy()
    out["peak_kW"] = con.max().to_numpy()
    out.round(3).to_csv(FIG_DIR / "fig0_casestudy_data.csv")
    print(f"  units {list(ann.index)}")
    print(f"  load {ann['load'].min():.2f}-{ann['load'].max():.2f} MWh, PV on {int((ann['pv']>0).sum())} of {len(ann)}")


if __name__ == "__main__":
    main()
