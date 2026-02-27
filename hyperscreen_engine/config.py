"""HyperScreen configuration.

This project is meant to be published on GitHub and archived on Zenodo.
To keep it portable across machines, *all* executable paths and run-time
parameters can be overridden via environment variables.

Environment variables
---------------------
GNINA_PATH
    Path to the `gnina` executable. Default: 'gnina' (expects it in PATH)
OBABEL_PATH
    Path to the `obabel` executable. Default: 'obabel'
HYPERSCREEN_DB
    SQLite DB path. Default: 'hyperscreen.db'

PREP_WORKERS
DOCKING_PROCESSES
DOCKING_TIMEOUT
FAST_EXHAUST
TOP_PERCENT_MD
"""

from __future__ import annotations

import os


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


# Executables
GNINA_PATH = os.environ.get("GNINA_PATH", "gnina")
OBABEL_PATH = os.environ.get("OBABEL_PATH", "obabel")

# Pipeline knobs
FAST_EXHAUST = os.environ.get("FAST_EXHAUST", "8")
TOP_PERCENT_MD = _get_float("TOP_PERCENT_MD", 0.05)

PREP_WORKERS = _get_int("PREP_WORKERS", 12)
DOCKING_PROCESSES = _get_int("DOCKING_PROCESSES", 4)
DOCKING_TIMEOUT = _get_int("DOCKING_TIMEOUT", 20)

# Outputs
DATABASE_PATH = os.environ.get("HYPERSCREEN_DB", "hyperscreen.db")
