"""The binned_likelihood grader's clock: how a dataset's wall is split between the mass grid and the GOF toys
(deadlines), for verify.py's capped verification."""
from __future__ import annotations

from typing import Tuple


WRAPUP_S = 5
GRID_FRAC = 0.7
MAX_JSON_BYTES = 256 * 2 ** 20


def deadlines(t0: float, budget: float) -> Tuple[float, float]:
    """verify.py's (STAGE_DEADLINE, GRID_DEADLINE) for `budget` seconds from monotonic `t0`: GOF toys stop WRAPUP_S
    before the end, the grid at GRID_FRAC.  Also used by the dev chain's capped verification (_verify_one)."""
    return t0 + budget - WRAPUP_S, t0 + GRID_FRAC * budget


