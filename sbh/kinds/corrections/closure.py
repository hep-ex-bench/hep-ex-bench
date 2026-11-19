"""The corrections kind's closure (`sbh closure`) -- EVALUATOR-ONLY (sbh/harbor_pack.py EVALUATOR_ONLY): the
criteria of a reference answering with correction functions, and the further baselines closure.json keeps."""
from __future__ import annotations

import math
from typing import Any, Dict


CLOSURE_PULL_RMS = (0.3, 2.0)
CLOSURE_MAX_PULL = 4.0
CLOSURE_SLOPE_TOL = 0.15
CLOSURE_SLOPE_K = 3.0
CLOSURE_BIAS_K = 3.5


def slope_ok(slope: float, se: float) -> bool:
    """The response criterion of a role whose truth takes >= 3 values: |slope - 1| <= max(CLOSURE_SLOPE_TOL,
    CLOSURE_SLOPE_K err)."""
    return bool(abs(slope - 1.0) <= max(CLOSURE_SLOPE_TOL, CLOSURE_SLOPE_K * se))


def function_criteria(pub: dict, priv: dict, max_pull: float, pull_rms=CLOSURE_PULL_RMS, bias_rho=None):
    """Closure criteria from the summary's functions block: every dataset completed and every requested correction
    answered; per group the pull RMS sqrt(mean z^2) of the point pulls within pull_rms; per role whose truth takes
    >= 3 values slope_ok; every case's mean point pull within max_pull; with bias_rho ({group: rho}) per role the
    mean pull over its N answered datasets within
    CLOSURE_BIAS_K sqrt(rho + (1 - rho) / N) -- the datasets share part of their error (simulation, source events),
    so the mean's error does not fall as 1 / sqrt(N).  Returns (criteria, max |mean pull|)."""
    fn = pub.get("functions") or {}
    lo, hi = pull_rms
    crit = {"recovery == 1": pub["recovery"] == 1.0,
            "every requested correction answered": bool(fn.get("n_requested")) and fn["n_answered"] == fn["n_requested"]}
    for g, m in (fn.get("groups") or {}).items():
        crit[f"{g} pull RMS in [{lo}, {hi}]"] = m.get("pull_rms") is not None and lo <= m["pull_rms"] <= hi
    for r, m in (fn.get("roles") or {}).items():
        if m.get("slope") is not None:
            crit[f"{r} response |slope-1| <= max({CLOSURE_SLOPE_TOL}, {CLOSURE_SLOPE_K:g} err)"] = slope_ok(
                m["slope"], m["slope_se"])
    group_of = {r: g for g, m in (fn.get("groups") or {}).items() for r in m.get("roles") or []}
    for r, m in (fn.get("roles") or {}).items():
        rho = (bias_rho or {}).get(group_of.get(r))
        if rho is not None and m.get("zbar_mean") is not None and m.get("n_answered"):
            gate = CLOSURE_BIAS_K * math.sqrt(rho + (1.0 - rho) / m["n_answered"])
            crit[f"{r} |mean pull over the datasets| <= {gate:.2f} (bias gate, rho {rho})"] = bool(
                abs(m["zbar_mean"]) <= gate)
    zs = [abs(v) for s in priv["per_seed"] for v in s["pulls"].values() if v is not None]
    mp = max(zs) if zs else float("inf")
    crit[f"max |mean pull| <= {max_pull}"] = mp <= max_pull
    return crit, mp


def closure_criteria(pub: dict, priv: dict, task, a, offsets=None, scored_calibration=None):
    """(criteria, max |mean pull|, further closure.json fields): function_criteria with the task's own bands, the
    baseline id (function_baseline_id) and the kind's further baselines (scoring.closure_extras: the closure
    comparison's)."""
    from .scoring import closure_extras
    if offsets is not None or scored_calibration or getattr(a, "max_leakage", None) is not None:
        raise SystemExit("closure: --calibrate-bias, --scored-calibration and --max-leakage gate POIs and logged "
                         "nuisances; a corrections task has neither")
    mp = a.max_pull if a.max_pull is not None else getattr(task, "closure_max_pull", None) or CLOSURE_MAX_PULL
    band = getattr(task, "closure_pull_rms", None) or CLOSURE_PULL_RMS
    rho = getattr(task, "closure_bias_rho", None)
    crit, max_pull = function_criteria(pub, priv, mp, band, rho)
    missing = priv.get("peak_baseline_missing")
    if missing is not None:
        from .comparison import noun
        what = noun(task)
        crit[f"every dataset's {what} compared"] = not missing
        for m in missing:
            print(f"closure: no {what} baseline for {m['seed']} {m['flavour']}: {m['why']}")
    fc = {"pull_rms": list(band), "max_pull": mp,
          **({"bias_rho": rho, "bias_k": CLOSURE_BIAS_K} if rho else {})}
    fb = task.function_baseline_id()
    fields: Dict[str, Any] = {**({"function_baseline": fb} if fb is not None else {}), "function_closure": fc,
                              **closure_extras(task, priv)}
    return crit, max_pull, fields
