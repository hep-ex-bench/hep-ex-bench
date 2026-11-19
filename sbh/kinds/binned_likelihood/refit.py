"""Model refits on the grader's JAX twin for GRADING criterion 7 (model.py model_consistency) -- EVALUATOR-ONLY.

pyhf (the grader's pinned copy) builds the model exactly as the run gate does; only the FIT runs on the JAX twin
(sbh/kinds/binned_likelihood/jax_engine.py: Migrad with exact gradients, combine's 1e-9 yield floor), validated against pyhf and
combine and much faster than pyhf/SciPy on large models.

No negative-rate rule: combine's semantics are the reference.  The floored likelihood is fitted and only its mu is
compared, so a negative rate in an empty bin (which pyhf's Poisson rewards) gains nothing.  Unfloored rates at the
best fit (``min_rate``, ``n_floor``, ``n_sample_floor``, ``kink``; check_model's ``min_expected_rate`` /
``n_negative_rate_bins``) are reported only, never pass/fail.

Fallback: if the twin cannot build (unsupported modifier, failed rate cross-check, no jax) or does not converge,
pyhf refits; every record names its ``engine`` (``fallback`` says why pyhf).

jax lives only in the root-only SBH_GRADER_LIB (verifier image, never the agent's); this module is imported lazily so
the pyhf path never touches jax.
"""
from __future__ import annotations

import importlib
import time

ENGINE_JAX, ENGINE_PYHF = "jax", "pyhf"
_TWIN: dict = {}


def twin_module():
    """(the jax_engine module, None) or (None, why); imported once per process with the grader libs on sys.path."""
    if "mod" not in _TWIN:
        try:
            from ... import use_grader_lib
            use_grader_lib()
            _TWIN["mod"], _TWIN["err"] = importlib.import_module("sbh.kinds.binned_likelihood.jax_engine"), None
        except Exception as e:  # noqa: BLE001
            _TWIN["mod"], _TWIN["err"] = None, f"{type(e).__name__}: {e}"
    return _TWIN["mod"], _TWIN["err"]


def build_twin(spec, model, share=None, shareable: bool = False):
    """(twin, None) or (None, why): the JAX twin of pyhf `model` from `spec` (no `_meta`), combine's floor on.
    `share`/`shareable`: compiled-graph sharing between structure-identical workspaces (mass-grid points)."""
    mod, err = twin_module()
    if mod is None:
        return None, f"jax engine unavailable: {err}"
    try:
        em, why = mod.build(spec, model, share=share, shareable=shareable)
    except Exception as e:  # noqa: BLE001
        return None, f"twin build failed: {type(e).__name__}: {e}"
    return (em, None) if em is not None else (None, f"twin build: {why}")


def fit_twin(em, model, data, init=None):
    """(pars in pyhf order, info) for the MLE of `data` on twin `em` from `init` (default: model init), or (None,
    info) with info["fallback"] if it did not converge or raised, so the caller refits with pyhf.
    info: engine, fit_s, twice_nll (= 2 (nll + full_nll_const), pyhf's convention where the floor is inactive),
    min_rate (UNFLOORED lowest expected rate), n_floor / n_sample_floor / kink.  On fallback these are prefixed
    twin_, so a pyhf best fit never carries the twin's numbers under the plain names."""
    import numpy as np
    t0 = time.perf_counter()
    try:
        d = np.asarray(data, dtype=float)
        n = int(model.config.nmaindata)
        main, aux = d[:n], d[n:]
        r = em.fit(main, aux, init=None if init is None else np.asarray(init, float), want_sigma=False)
        floor = {"min_rate": float(r["min_rate"]), "n_floor": int(r["n_floor"]),
                 "n_sample_floor": int(r["n_sample_floor"]), "kink": bool(r["kink"])}
        if not (r["valid"] and np.all(np.isfinite(r["x"])) and np.isfinite(r["nll"])):
            return None, {"fit_s": round(time.perf_counter() - t0, 3), "fallback": "twin fit did not converge",
                          **{f"twin_{k}": v for k, v in floor.items()}}
        pars = np.asarray(r["x"], dtype=float)
        twice_nll = float(2.0 * (r["nll"] + em.full_nll_const(main, aux)))
    except Exception as e:  # noqa: BLE001
        return None, {"fallback": f"twin fit failed: {type(e).__name__}: {e}"[:300],
                      "fit_s": round(time.perf_counter() - t0, 3)}
    return pars, {"fit_s": round(time.perf_counter() - t0, 3), **floor, "engine": ENGINE_JAX,
                  "twice_nll": twice_nll}


def twin_refit(spec, model, data, init=None):
    """model.check_model's `fit` hook: (pars, info) from the twin, or (None, info) with info["fallback"] so pyhf
    refits (a twin-side error never means "workspace does not fit").  info["build_s"]: twin build incl. cross-check
    (and the first call's jax import)."""
    t0 = time.perf_counter()
    try:
        em, why = build_twin(spec, model)
        build_s = round(time.perf_counter() - t0, 3)
        if em is None:
            return None, {"fallback": why[:300], "build_s": build_s}
        pars, info = fit_twin(em, model, data, init=init)
    except Exception as e:  # noqa: BLE001
        return None, {"fallback": f"twin refit failed: {type(e).__name__}: {e}"[:300],
                      "build_s": round(time.perf_counter() - t0, 3)}
    return pars, {"build_s": build_s, **info}


grader_refit = twin_refit


