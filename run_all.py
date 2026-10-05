"""
run_all.py
==========
One-command full reproduction. Runs the three pipeline stages in order:

  1. main.py               solve Models 0-4 under every network tariff
                           archetype (T0-T3)  -> outputs/   (requires Gurobi)
  2. results_analysis.py   build the per-archetype figure set  -> figures/
  3. sweep_sizing.py       storage duration sweep E/P in {2,...,10} h
                           -> outputs/sweep_sizing.csv       (requires Gurobi)
  4. rolling_horizon.py    perfect foresight vs 24 h and 48 h rolling windows
                           -> outputs/rolling_horizon.csv    (requires Gurobi)
  5. fig_tariffs.py        build the cross-archetype figures   -> figures/
  6. fig_casestudy.py      build the case study figure         -> figures/

Usage:
    python run_all.py

Note: Stage 1 needs a valid Gurobi license (Models 2-4 solve MILPs). If you
only want to rebuild the figures from the shipped outputs/, run
`python results_analysis.py` directly instead - that stage needs no license.
"""
from __future__ import annotations

import fig_casestudy
import fig_tariffs
import main
import results_analysis
import rolling_horizon
import sweep_sizing


if __name__ == "__main__":
    print("========== STAGE 1/6: models under every archetype (main.py) ==========")
    main.main()

    print("\n========== STAGE 2/6: per-archetype figures (results_analysis.py) ==========")
    results_analysis.main_all()

    print("\n========== STAGE 3/6: storage duration sweep (sweep_sizing.py) ==========")
    sweep_sizing.run()

    print("\n========== STAGE 4/6: rolling horizon (rolling_horizon.py) ==========")
    rolling_horizon.main()

    print("\n========== STAGE 5/6: cross-archetype figures (fig_tariffs.py) ==========")
    fig_tariffs.main()

    print("\n========== STAGE 6/6: case study figure (fig_casestudy.py) ==========")
    fig_casestudy.main()

    print("\nDone. Results in outputs/, figures in figures/.")
