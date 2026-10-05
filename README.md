# Institutional design of community energy storage — source code and data

Source code, input data and model results accompanying

> Li, N., Koirala, B. *Institutional design of community energy storage: value
> creation, capture, and distribution.* iScience (under revision).

The code models a seven-unit energy community (the NEST research building,
Empa, Switzerland) under alternative institutional arrangements — storage
ownership, market access, retail price regime and network tariff design — and
traces how the resulting value is created, captured and distributed. It
reproduces every quantitative figure and table of the paper.

## Contents

| Path | What it is |
|------|------------|
| `main.py` | Stage 1: solves Models 0–4 for every tariff archetype and price regime, writes `outputs/` |
| `models/model0.py` … `model4.py` | The five institutional configurations (see below) |
| `tariffs.py`, `tariff_milp.py`, `network_settlement.py` | Network tariff archetypes T0–T3, their dispatch terms and ex-post settlement, the statutory LEG reduction (StromVV Art. 19h) and the storage refund |
| `energy_prices.py` | The three retail energy price regimes (flat, time-of-use, dynamic) |
| `results_analysis.py` | Stage 2: builds the per-archetype and composite paper figures from `outputs/` |
| `sweep_sizing.py` | Storage duration sensitivity (E/P = 2–10 h) → `outputs/sweep_sizing.csv` |
| `rolling_horizon.py` | Perfect foresight vs 24 h / 48 h rolling horizon → `outputs/rolling_horizon.csv` |
| `fig_tariffs.py`, `fig_casestudy.py` | Cross-archetype figures and the case-study figure |
| `run_all.py` | One-command full reproduction (all stages in order) |
| `config.py`, `utils.py` | Paths, default parameters, shared helpers |
| `data_processing.py`, `price_processing.py` | One-off preprocessing of the raw inputs (already applied) |
| `input_data/` | Model inputs: measured hourly load and PV profiles of the seven units (calendar year 2025) and the ENTSO-E day-ahead price series |
| `outputs/` | Model results written by Stage 1 (shipped separately, see below) |
| `figures/` | All figures and their underlying data (`fig*_data.csv`, `master_table*.csv`) |

### Institutional configurations

- **Model 0** — reference: no community, no storage.
- **Model 1** — local electricity sharing without storage.
- **Model 2** — individual behind-the-meter batteries plus local sharing.
- **Model 3** — community-owned shared storage, market-access Settings A/B/C.
- **Model 4** — third-party-owned shared storage, Settings A/B/C; the operator
  charges a per-kWh storage service fee `tau` (`DEFAULTS.PI_TAU`, 0.02 CHF/kWh)
  and no fixed participation fee in the baseline.

Market-access settings for shared storage: **A** internal balancing only,
**B** import-only market access, **C** full market access (charge from and
discharge to the market). Retail price regimes: `flat`, `tou`, `dynamic`
(`energy_prices.py`). Network tariff archetypes: **T0** energy-only benchmark
(no network charge), **T1** volumetric, **T2** contracted-capacity,
**T3** volumetric + contracted-capacity (`tariffs.py`); T1–T3 are calibrated to
recover the same Model 0 network revenue. Settings B and C are evaluated under
dynamic pricing only, where market access is an economically meaningful choice,
giving the (5 x 3 + 4 x 1) x 4 = 76 scenarios of the paper (`market_access_modes`
in `main.py`; listing all three regimes there runs all 108 combinations). Every
scenario is swept over the local sharing price `pi_loc` in 0.01 CHF/kWh steps.

All monetary values are in CHF excluding VAT.

## Requirements

- Python 3.13 (tested with 3.13.0)
- A **Gurobi** license — Models 2–4, the sizing sweep and the rolling-horizon
  test solve MILPs with `gurobipy`. Free academic licenses:
  https://www.gurobi.com/academia/. Not needed to rebuild the figures from the
  shipped results.
- Python packages in `requirements.txt` (pinned to the versions used):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to reproduce

All scripts resolve paths relative to the repository root and can be run from
there.

| Goal | Command | Gurobi |
|------|---------|--------|
| Rebuild all figures from the shipped `outputs/` | `python results_analysis.py && python fig_tariffs.py && python fig_casestudy.py` | no |
| Full reproduction from scratch | `python run_all.py` | yes |

`run_all.py` runs, in order: `main.py` (all models and archetypes, about
7 minutes on a laptop), `results_analysis.main_all()`, `sweep_sizing.py`
(about 12 minutes), `rolling_horizon.py`, `fig_tariffs.py`, `fig_casestudy.py`.
Run settings
(models, price regimes, archetypes, settings A/B/C) are configured in the
`Settings` dataclass at the top of `main.py`; physical and economic defaults in
`config.py`.

This repository mirrors the archived release on Zenodo
(https://doi.org/10.5281/zenodo.22847518), which is the citable version of record.

**Model results (`outputs/`)** are distributed as a separate archive
(`outputs.zip`, ~450 MB unpacked) because of their size. Unpack it into the
repository root so that `outputs/model0_summary_flat_T0.csv` etc. exist; the
figure scripts then run without a Gurobi license.

## Figure and table mapping

File names in `figures/` predate the final figure order of the paper; the
correspondence is:

| Paper | File in `figures/` | Produced by |
|-------|--------------------|-------------|
| Figure 1 | `Fig1_heatmap_all` | `results_analysis.py` |
| Figure 2 | `Fig2_market_access_all` | `results_analysis.py` |
| Figure 3 | `Fig3_third_party_revenue_all` | `results_analysis.py` |
| Figure 4 | `Fig4_cv_all` | `results_analysis.py` |
| Figure 5 | `Fig5_fairness_all` | `results_analysis.py` |
| Figure 6 | `Fig8_delta_sensitivity` | `fig_tariffs.py` |
| Figure 7 | `Fig6_ldc_all` | `results_analysis.py` |
| Figure 8 | `Fig9_sizing` | `fig_tariffs.py` (reads `outputs/sweep_sizing.csv`) |
| Figure 9 | `Fig0_casestudy` | `fig_casestudy.py` |
| Figure 10 | — | conceptual framework, drawn manually |
| SI Table 6 (rolling horizon) | `outputs/rolling_horizon.csv` | `rolling_horizon.py` |
| SI network tariff parameters | `tariffs.py` (`ARCHETYPES`) | — |

The single-archetype versions (`*_T0` … `*_T3`), `Fig7_tariff_comparison` and
`Fig8_distribution` are diagnostic figures not shown in the paper. Each figure
has a companion `fig*_data.csv` with the plotted numbers, and
`master_table_T*.csv` collects all scenario-level indicators per archetype.

## Input data

`input_data/inputs_filled.csv` holds the gap-filled hourly load and PV
profiles of the seven NEST units (`load_<id>`, `PV_<id>`, UTC timestamps) used by
all models; `inputs.csv` is the raw export from which `data_processing.py`
builds it. `CH_price_dyn_2025.csv` is the ENTSO-E day-ahead export for the
Swiss bidding zone; `price_processing.py` derives `price_dynamic_2025.csv`,
whose hourly shape `energy_prices.py` scales to the retail level (it also
writes `price_tou_3block_2025.csv` and `price_flat_2025.txt`, which the current
models do not use: the flat and time-of-use regimes are built from the
distribution system operator's published rates in `energy_prices.py`). Published
tariff sheets underlying `tariffs.py` and `energy_prices.py` are cited in the
Supplementary Information (Section S8).

## License

Code and data are released under the MIT License — see `LICENSE`.
