# config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# project root directory (the folder containing this file)
BASE_DIR = Path(__file__).resolve().parent

PriceMode = Literal["flat", "tou", "dynamic"]
DesignABC = Literal["A", "B", "C"]
TariffArchetype = Literal["T0", "T1", "T2", "T3"]


@dataclass(frozen=True)
class Paths:
    BASE_DIR: Path = BASE_DIR
    INPUT_DIR: Path = BASE_DIR / "input_data"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    MODELS_DIR: Path = BASE_DIR / "models"

    INPUTS_FILLED: Path = INPUT_DIR / "inputs_filled.csv"

    PRICE_DYNAMIC: Path = INPUT_DIR / "price_dynamic_2025.csv"
    PRICE_TOU: Path = INPUT_DIR / "price_tou_3block_2025.csv"
    PRICE_FLAT: Path = INPUT_DIR / "price_flat_2025.txt"


@dataclass(frozen=True)
class Defaults:
    # Currency: all monetary values are in CHF and EXCLUDE VAT (8.1%). VAT is a
    # uniform proportional markup and cancels in every relative metric reported
    # (savings %, CV, Gini). ENTSO-E day-ahead prices are quoted in EUR and are
    # converted at the 2025 annual average, 1 EUR = 0.9370 CHF (tariffs.EUR_TO_CHF).
    #
    # PI_EXP is the statutory minimum feed-in remuneration for PV below 30 kW
    # (Energieverordnung Art. 12 para. 1bis), applied by Glattwerk AG as the local
    # DSO. All PV systems in the case study are below 30 kW, so this floor binds.
    PI_EXP: float = 0.06
    LOC_STEP: float = 0.01
    PI_TAU: float = 0.02  # Model 4: per-kWh storage service fee tau (CHF/kWh)

    ETA_CH: float = 0.95
    ETA_DIS: float = 0.95
    CES_POWER_HOURS: float = 4.0
    SOC0_FRAC: float = 0.0

    CAP_BY_SAVING: bool = True
    ALLOC_RULE: str = "equal"

    # ---- network tariff (see tariffs.py and Supplementary Information S8.4) ----
    # An energy-only benchmark plus three archetypes spanning volumetric x
    # capacity recovery, T1-T3 calibrated to recover the same Model 0 network
    # revenue (CHF 3,697.6/yr). Parameters are defined in tariffs.ARCHETYPES:
    #   T0 no network charge      (energy-only benchmark)
    #   T1 volumetric only        8.10 / 7.10 Rp/kWh (HT / NT)
    #   T2 capacity only          21.86 CHF/kW/yr on CONTRACTED connection capacity
    #   T3 volumetric + capacity  3.49 / 3.21 Rp/kWh + 12.18 CHF/kW/yr contracted
    TARIFF: str = "T1"

    # Statutory reduction of the network usage tariff for LEG participants.
    # 0.40 = StromVV Art. 19h para. 1, the case-study calibration (single
    # building, one transformer circuit). See DELTA_GRID for the sensitivity set.
    DELTA_LEG: float = 0.40


PATHS = Paths()
DEFAULTS = Defaults()

# Network tariff archetypes scanned in the main analysis.
TARIFF_GRID: tuple[str, ...] = ("T0", "T1", "T2", "T3")

# LEG reduction rates for the sensitivity analysis. Each value corresponds to a
# regime in force in a European jurisdiction (Supplementary Information S8.5):
#   0.00 no reduction | 0.20 CH crossing transformer | 0.40 CH same circuit
#   0.60 ~ AT local EEG | 1.00 collective self-consumption without grid use
DELTA_GRID: tuple[float, ...] = (0.0, 0.20, 0.40, 0.60, 1.00)
