"""
main.py
=======
Entry point that runs the full model pipeline. For each network tariff
archetype (T0-T3) and each retail price regime (flat / tou / dynamic) it runs
Models 0-4 (Models 3 and 4 each across the three market-access settings A/B/C)
and writes every result to the outputs/ folder.
Configure a run via the Settings dataclass below, then:

    python main.py

Requires a Gurobi license (Models 2-4 solve MILPs with gurobipy).
Downstream: run results_analysis.py (and fig_tariffs.py, fig_casestudy.py) to
build the paper figures from outputs/; see README.md for the figure mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Optional
import inspect
import traceback
from time import perf_counter

# ---- import your model runners ----
from models.model0 import run_model0_for_price_mode
from models.model1 import run_model1_for_price_mode
from models.model2 import run_model2_for_price_mode
from models.model3 import run_model3_for_price_mode
from models.model4 import run_model4_for_price_mode


from config import DEFAULTS, TARIFF_GRID

PriceMode = Literal["flat", "tou", "dynamic"]
DesignABC = Literal["A", "B", "C"]
TariffName = Literal["T0", "T1", "T2", "T3"]


@dataclass
class Settings:
    # Global
    pi_exp: float = 0.06
    loc_step: float = 0.01
    price_modes: List[PriceMode] = field(default_factory=lambda: ["flat", "tou", "dynamic"])

    # Which models to run
    run_models: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # Network tariff archetypes to scan. T0 reproduces the originally submitted
    # model (no network charge); T1-T3 span the structural space of network cost
    # recovery (Supplementary Information S8.4).
    tariffs: List[TariffName] = field(default_factory=lambda: list(TARIFF_GRID))

    # Statutory LEG reduction of the network usage tariff. 0.40 is the
    # case-study calibration (StromVV Art. 19h para. 1). Applied under T1-T3.
    delta_leg: float = DEFAULTS.DELTA_LEG

    # Market-access settings for shared storage (Models 3 and 4). Settings B and
    # C are evaluated only under the price regimes listed in market_access_modes:
    # market access is an economically meaningful choice only where prices vary
    # within the day, so the paper reports B and C under dynamic pricing, giving
    # (5 x 3 + 4 x 1) x 4 = 76 scenarios. Listing all three regimes runs 108.
    designs_abc: List[DesignABC] = field(default_factory=lambda: ["A", "B", "C"])
    market_access_modes: List[PriceMode] = field(default_factory=lambda: ["dynamic"])
    model3_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
        save_flows=False,
        # you can put optional knobs here safely:
        # eta_ch=0.95, eta_dis=0.95, ces_power_hours=4.0, soc0_frac=0.0,
    ))

    # Model 4 (third-party CES) is run once per design. The operator charges a
    # per-kWh storage service fee tau (DEFAULTS.PI_TAU = 0.02 CHF/kWh) with a
    # zero fixed fee, matching the paper; those defaults live in config.py.
    model4_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
        # optional knobs (see run_model4_for_price_mode):
        # tau=0.02, fixed_fee_annual=0.0, eta_ch=0.95, eta_dis=0.95,
    ))

    # Behavior
    fail_fast: bool = True   # if False: continue even if one run fails
    verbose_traceback: bool = True


def _filter_kwargs(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs that the function actually accepts."""
    sig = inspect.signature(func)
    params = sig.parameters
    # if **kwargs exists, keep everything
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _run_step(
    title: str,
    func,
    kwargs: Dict[str, Any],
    *,
    fail_fast: bool,
    verbose_traceback: bool,
) -> bool:
    """Run a step with timing + robust error handling. Returns success flag."""
    t0 = perf_counter()
    try:
        func(**kwargs)
        dt = perf_counter() - t0
        print(f"  [ok] {title} finished in {dt:.2f}s")
        return True
    except Exception as e:
        dt = perf_counter() - t0
        print(f"  [FAILED] {title} after {dt:.2f}s: {type(e).__name__}: {e}")
        if verbose_traceback:
            print("  ---- traceback ----")
            print(traceback.format_exc())
            print("  -------------------")
        if fail_fast:
            raise
        return False


def designs_for(s: Settings, mode: PriceMode) -> List[DesignABC]:
    """Settings to run for a price regime: A always, B and C only under market_access_modes."""
    return [d for d in s.designs_abc if d == "A" or mode in s.market_access_modes]


def main(settings: Optional[Settings] = None) -> None:
    s = settings or Settings()

    print("===================================================")
    print(" Running models (0-4)")
    print("===================================================")
    print(f"pi_exp={s.pi_exp}, loc_step={s.loc_step}")
    print(f"price_modes={s.price_modes}")
    print(f"run_models={s.run_models}")
    print(f"tariffs={s.tariffs}, delta_leg={s.delta_leg}")
    print(f"designs={s.designs_abc}, market_access_modes={s.market_access_modes}")
    print("---------------------------------------------------")

    n_tasks = 0
    for tariff in s.tariffs:
        for mode in s.price_modes:
            if 0 in s.run_models: n_tasks += 1
            if 1 in s.run_models: n_tasks += 1
            if 2 in s.run_models: n_tasks += 1
            if 3 in s.run_models: n_tasks += len(designs_for(s, mode))
            if 4 in s.run_models: n_tasks += len(designs_for(s, mode))
    print(f"Total planned runs: {n_tasks}")
    print("===================================================")

    done = 0
    ok = 0
    t_all = perf_counter()

    for tariff in s.tariffs:
      print(f"\n############### NETWORK TARIFF: {tariff} ###############")
      for mode in s.price_modes:
        print(f"\n================== PRICE MODE: {mode} ==================\n")

        # Model 0
        if 0 in s.run_models:
            done += 1
            title = f"[{done}/{n_tasks}] Model 0 | {tariff} | mode={mode}"
            kwargs = dict(mode=mode, pi_exp=s.pi_exp, tariff=tariff)
            ok += int(_run_step(
                title, run_model0_for_price_mode, kwargs,
                fail_fast=s.fail_fast, verbose_traceback=s.verbose_traceback
            ))

        # Model 1
        if 1 in s.run_models:
            done += 1
            title = f"[{done}/{n_tasks}] Model 1 | {tariff} | mode={mode}"
            kwargs = dict(mode=mode, pi_exp=s.pi_exp, loc_step=s.loc_step,
                          tariff=tariff, delta=s.delta_leg)
            ok += int(_run_step(
                title, run_model1_for_price_mode, kwargs,
                fail_fast=s.fail_fast, verbose_traceback=s.verbose_traceback
            ))

        # Model 2
        if 2 in s.run_models:
            done += 1
            title = f"[{done}/{n_tasks}] Model 2 | {tariff} | mode={mode}"
            kwargs = dict(mode=mode, pi_exp=s.pi_exp, loc_step=s.loc_step,
                          tariff=tariff, delta=s.delta_leg)
            ok += int(_run_step(
                title, run_model2_for_price_mode, kwargs,
                fail_fast=s.fail_fast, verbose_traceback=s.verbose_traceback
            ))

        # Model 3
        if 3 in s.run_models:
            for design in designs_for(s, mode):
                done += 1
                title = f"[{done}/{n_tasks}] Model 3 | {tariff} | mode={mode}, design={design}"
                base_kwargs = dict(mode=mode, design=design, pi_exp=s.pi_exp, loc_step=s.loc_step,
                                   tariff=tariff, delta=s.delta_leg)
                base_kwargs.update(s.model3_kwargs)
                kwargs = _filter_kwargs(run_model3_for_price_mode, base_kwargs)
                ok += int(_run_step(
                    title, run_model3_for_price_mode, kwargs,
                    fail_fast=s.fail_fast, verbose_traceback=s.verbose_traceback
                ))

        # Model 4
        if 4 in s.run_models:
            for design in designs_for(s, mode):
                done += 1
                title = f"[{done}/{n_tasks}] Model 4 | {tariff} | mode={mode}, design={design}"

                base_kwargs = dict(
                    mode=mode,
                    design=design,
                    pi_exp=s.pi_exp,
                    loc_step=s.loc_step,
                    tariff=tariff,
                    delta=s.delta_leg,
                )
                base_kwargs.update(s.model4_kwargs)

                kwargs = _filter_kwargs(run_model4_for_price_mode, base_kwargs)
                ok += int(_run_step(
                    title, run_model4_for_price_mode, kwargs,
                    fail_fast=s.fail_fast, verbose_traceback=s.verbose_traceback
                ))

    dt_all = perf_counter() - t_all
    print("\n===================================================")
    print(f"Finished. Success: {ok}/{n_tasks}. Total time: {dt_all:.2f}s")
    print("Check outputs/ folder.")
    print("===================================================")


if __name__ == "__main__":
    main()
