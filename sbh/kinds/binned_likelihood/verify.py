#!/usr/bin/env python3
"""Verify a submission's exported statistical model with pyhf + cabinetry.

A submission may export MODEL.json (a pyhf workspace whose observations are the
events it selected) next to RESULT.json.  Per seed, independently of the
submission's own fit:

  0. NOMINAL FIT   — cabinetry (the twin above PYHF_NOMINAL_MAX_FREE free
                     parameters on jax); after a failure the twin on jax, and
                     pyhf's own MLE where cabinetry is absent
                     (_fallback_nominal_fit), every attempt in fit_attempts.
                     Unfittable on them is "fit failed: <reason> (decisive
                     attempt: ...)".  RESULT.json is read as reported
                     (_read_result).
  1. CONSISTENCY   — MLE of the declared model vs the reported mu, and the
                     model's own mu measurement (mu_fit_block), which the
                     scorer uses in place of RESULT.json's.
  2. SATURATED GOF — asymptotic saturated-model p-value (recorded only).
  3. IMPACTS       — post-fit nuisance impacts on mu (cabinetry ranking,
                     post-fit refits only) and the Asimov Z_exp.
  4. TOYS (--toys N) — toys at the fit: calibrated saturated-GOF p, bias,
                     toy-spread / quoted-uncertainty.  Toy p counts ties within
                     GOF_TIE_EPS; None when gof_dof <= 0.
  5. COVERAGE (--toys N --suite <dir>) — SCORED: toys at the truth-pinned
                     conditional fit (POI = true mu, nuisances profiled on the
                     observations, aux data held), every toy refit -> global,
                     window and combined GOF p's (_gof_pvalues), coverage of the
                     68% interval (a toy without a POI sigma is covered only if
                     mu-hat == truth), pull mean / width.  Needs the sealed
                     manifest: evaluator-side only.  SBH_GOF_SEQ makes the toy
                     count sequential (sbh/kinds/binned_likelihood/seq_gof.py).
  6. MASS GRID     — with MODEL_GRID/ (mH_<value>.json) or MODEL_GRID.json:
                     profile-fit each fixed-mH point inside the task's mass
                     range (Hessian width of mu), scan mu conditionally per point
                     (mu_scan: the single-mass likelihood the marginal mu rule
                     marginalises, scoring.marginal_mu), read each template's
                     yield and centroid and check them against the task's yield
                     curve (grid_content), record the coverage of the range
                     (grid_coverage), build 2*DeltalnL(mH), take its minimum and
                     0.5 crossings, check every point is MODEL.json's model
                     (_grid_signature) and link the grid to MODEL.json: the
                     grid's own workspace nearest the reported mH is
                     MODEL.json's, else its fitted point there reproduces
                     MODEL.json's fit (_grid_link); linked by the first route,
                     the profile takes the lower of that point's 2NLL and
                     MODEL.json's (_profile_at_model, no fit).  The scorer
                     reads the scored mass and the mu density from this block.
                     No truths.  --stage grid redoes only this block on a
                     stored file (a scanning task's every dataset, any task's
                     combine-graded ones; _regrid), --stage link only the link
                     (no fit); neither needs a suite.  --prior-curve adds the
                     reference's sigma0_ref per point.
  7. COMBINE       — with a CMS combine declaration (MODEL_COMBINE/datacard.txt
                     + shapes, or a text2workspace workspace.root) and combine
                     available, stages 1-5 run in combine (combine_battery),
                     same record schema, engine "combine"; MODEL.json is
                     measured exactly as on the MODEL.json path
                     (_combine_record: stage 0's nominal fit, unfittable =
                     "fit failed"; the twin's observed-data fit on the jax
                     engine; mu_fit, the scored mu; pyhf_fit.twice_nll) and the
                     declared pyhf grid goes through stage 6 against it.
                     combine's own MH scan, where its model floats MH, is kept
                     as combine_mass_grid beside a declared grid and as
                     mass_grid where none is declared; either mass_grid is
                     stamped declared_grid and combine_grid_rule
                     (COMBINE_GRID_RULE).  A declaration that cannot be graded
                     falls back to MODEL.json with the reason in engine_fallback
                     (SBH_COMBINE_BATTERY=0 forces this).  MODEL.json's nominal
                     fit beside combine's r is kept as ``pyhf_crosscheck``
                     (report-only).  combine is only in the `dev` env
                     (SBH_COMBINE_BIN); the eval env never needs ROOT.

  OUTPUTS are read as the contract reads them (contract.read_output: never
  through a symlink) and within its size caps (model.MODEL_MAX_BYTES,
  model.grid_cap; the run's task): MODEL.json above its cap is the model's
  failure (_size_gate), a grid above its cap is not read (mass_grid
  "over_cap"; the scorer applies the task's contract).

  FAILURES: a fit of the model that raises is the model's ("fit failed",
  fit_error names it: ModelFitError); an exception of the grader's own code is
  the evaluator's, "evaluator_error" with the error text.  What the declared
  model's own numbers make impossible is the model's too, named before
  anything raises: a constraint width that is not a positive finite number
  (model.constraint_width_problem: pyhf's fit fails on it) is "fit failed"
  before any fit, on MODEL.json and on a grid point, on both engines; a toy
  whose rate no Poisson draw takes (check_toy_rates: a main or
  Poisson-constraint rate at the generation point that is not a number in
  [0, POISSON_RATE_MAX], numpy's own limit) is its toy stage's "fit failed",
  the GOF charged; the twin's fit is valid only at a finite NLL; a grid point
  whose fit ends at a 2NLL that is not a number is a failed point and the
  grid's tie set ranks only finite ones.  Each stage after the
  measurement (impacts, Z_exp, the toys, the sequential GOF, the claims'
  cross-checks, the grid) runs in its own guard, so an error there stays on
  its block (impacts, z_exp_asimov, toys, freq_toys or gof_seq,
  quality_crosscheck, mass_grid) and mu_fit and mass_grid are kept; the
  record's own "verify error" is left for a raise before the measurement
  exists.  Nothing a submission reports raises here: the optional claims are
  read through verify_fit.finite_number (one that is not a finite number is a
  claim that disagrees), and the grid's arithmetic is total (verify_grid:
  a ratio beyond the float range is no ratio).  evaluator_errors lists the
  blocks whose error leaves a scored number out, report_only_errors the
  others (impacts, gof_seq, pyhf_crosscheck, combine_model).  Every record
  is stamped "errors_marked" (ERRORS_MARKED, mark_errors): its unmarked
  errors are the model's; a record without the stamp was graded before the
  grader marked its own.  Each dataset's grader_attempts counts its
  verifications and those that ended with an evaluator error
  (evaluator_errors), carried from the file a run replaces (a record without
  them counts an evaluator error only where it is stamped); --stage retry
  (the post stage, before the score) verifies once more every completed
  dataset whose error has not recurred, so a deterministic fault is
  reproduced before the run is first scored, unless the fault is systemic
  (every place the dataset faults lies where a fault sits on more than half
  of the completed datasets, is_systemic, SYSTEMIC_FAULT, decided over the
  whole file): then nothing is verified again and those attempts are counted
  apart (n_systemic), never as reproducing a fault.  Every dataset's
  placeholder (unfinished_record, GRADING_UNFINISHED) is written before any is
  graded, so a job that dies leaves an evaluator fault on the datasets it did
  not finish, never a dataset with no record.  The grid's
  points each run within what is left of GRID_DEADLINE: a point that runs
  out is a hole in the grid.  SHIPPED keeps the grader of a
  task kept on the rule it shipped with (Task.score_rule SHIPPED_RULE) as it
  shipped: no stage guards, the claims read with float(), pyhf's own toy
  rates, its combine record (_combine_record_shipped), no stamp, no grid or
  retry stage, no placeholder, and none of the model-boundary refusals above
  (the twin as it built it: jax_engine.build shipped).

  TOY RATES: on the pyhf engine toys are drawn at the generation point's main
  rates floored at GEN_RATE_FLOOR (_pyhf_toy_pdf), as the twin and combine draw
  them; where no rate is below it the draws are pyhf's own, draw for draw.

  ENGINES (SBH_TOY_ENGINE, see ENGINE below): the nominal fit (stage 0), its
  asymptotic GOF and the consistency check are pyhf/cabinetry's; ``jax``
  (alias ``evermore``) runs stages 3-6 on the model's JAX twin with combine's
  yield floor.  Default ``pyhf``; the grader stage (grader.py) runs ``jax``.

CAP (Task.model_verify_cap, with --suite): each seed runs within the cap in
the suite's closure.json (post.py verify_cap; none for the closure's reference
run or a combine-graded seed); a seed not done in time fails ("fit failed",
ran_out).  verify_wall_s and "verify_cap" are recorded.

Usage:
  python -m sbh.kinds.binned_likelihood.verify --run-dir <run dir with runs/seed_*> \
      [--suite DIR] [--gof-window LO HI] [--result-name RESULT.json] [--out model_verify.json]
  python -m sbh.kinds.binned_likelihood.verify --run-dir <evaluation dir> --stage link [--task NAME] \
      [--out model_verify_toys.json]
  python -m sbh.kinds.binned_likelihood.verify --run-dir <evaluation dir> --stage grid --task NAME \
      [--out model_verify_toys.json] [--jobs N]
  python -m sbh.kinds.binned_likelihood.verify --run-dir <run dir> --stage retry --suite DIR --toys N \
      [--out model_verify.json] [--prior-curve] [--jobs N]
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import json
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import pyhf
try:
    import cabinetry
except ImportError:
    cabinetry = None

logging.getLogger("cabinetry").setLevel(logging.WARNING)
logging.getLogger("pyhf").setLevel(logging.WARNING)


if not __package__:
    import sys
    if str(Path(__file__).resolve().parents[3]) not in sys.path:
        sys.path.append(str(Path(__file__).resolve().parents[3]))
    __package__ = "sbh.kinds.binned_likelihood"
from .verify_fit import (  # noqa: F401
    _TwinFitError, _err, _evt, _twin_build, TWIN_RESTART_DNLL, _twin_obs_fit, _PYHF_HESSE_BUG, fit_error,
    ModelFitError, model_fit, _pyhf_postfit_impacts, _z_exp, PROFILE_TOL, PROFILE_MAX_FITS, _profile_side,
    mu_fit_block, _read_result, result_fields, _crosschecks, _crosschecks_shipped, finite_number, NOT_A_NUMBER,
    MU_SUPPORT, SCAN_VERSION, SCAN_K, SCAN_FIXED, _conditional_fitters, mu_scan_set, conditional_scan,
    POISSON_RATE_MAX, poisson_aux_rates, check_toy_rates)
from .verify_gof import (  # noqa: F401
    _deviance_terms, _saturated_q, _twin_gof, _pyhf_gof, GOF_TIE_EPS, _toy_p, _pooled_p, _gof_dof, _gof_undefined,
    _gof_stats, _bound_window)
from .verify_grid import (  # noqa: F401
    LINK_MU_REL, LINK_MU_ABS, LINK_TWICE_NLL_TOL, REACH_TOL, MAX_GAP_IN_RANGE, _grid_link, _load_grid,
    _model_twice_nll, _profile_at_model, restore_fitted_profile, refresh_profile, _reported_numbers,
    _grid_signature, _measurement_cfg, _measurement_sig, _fixed_poi, _GridTwin, _pyhf_point, scan_point,
    sigma0_width, template_facts, grid_content, grid_coverage, _envelope_crossings, _interval_summary)
from ...budget import BudgetExceeded, budget  # noqa: E402
from ...contract import load_output_json  # noqa: E402
from .model import MODEL_MAX_BYTES, GridOverCap, constraint_width_problem, grid_bytes, grid_cap, read_point  # noqa: F401
from .verify_combine import (  # noqa: F401
    _pyhf_crosscheck, COMBINE_MAX_IMPACTS, _WS_PROBE, _FITDIAG_PROBE, _combine_bindir, _run_dev, _probe_json,
    _datacard_observation, _shapes_data_sums, _export_fitdiag_shapes, _newest, _limit_arrays, _fitdiag_toys,
    COMBINE_SEED_AT_FIT, COMBINE_SEED_PINNED, COMBINE_MH_SCAN_POINTS, COMBINE_MH_SCAN_NSIGMA,
    COMBINE_TOY_STATUS_OK, MINOS_EDGE_RTOL, _minos_coverage, combine_battery)

N_TOYS = 0
N_TOYS_FIT = None
STAGE_DEADLINE = None
GRID_DEADLINE = None
SKIP_IMPACTS = False
TASK_NAME = None
PRIOR_CURVE = False
LINK_IDENTICAL = None
SHIPPED = None
GEN_RATE_FLOOR = 1e-9
COMBINE_GRID_RULE = 2
ERRORS_MARKED_KEY, ERRORS_MARKED = "errors_marked", 1
_ENGINE_RAW = os.environ.get("SBH_TOY_ENGINE", "pyhf")
ENGINE = {"pyhf": "pyhf", "jax": "jax", "evermore": "jax"}.get(_ENGINE_RAW.strip().lower())
SEQ_GOF = None
if os.environ.get("SBH_GOF_SEQ", "").strip().lower() not in ("", "0", "off", "no", "false"):
    from sbh.kinds.binned_likelihood import seq_gof as _seq_gof_mod
    SEQ_GOF = _seq_gof_mod.config_from_env()


PYHF_NOMINAL_MAX_FREE = int(os.environ.get("SBH_PYHF_NOMINAL_MAX_FREE", "200"))


def _twin_nominal_fit(ev, model, data):
    """The nominal fit on the twin, as a cabinetry-like result: best fit and autodiff-Hessian sigmas from init;
    goodness_of_fit the asymptotic p of combine's saturated q at the fit (_gof_dof dof); best_twice_nll pyhf's
    own -2 log L at that point (for _grid_link).  None when the fit is invalid."""
    from types import SimpleNamespace
    from scipy.stats import chi2
    d = np.asarray(data, dtype=float)
    main, aux = d[:ev.n_main], d[ev.n_main:]
    r = ev.fit(main, aux, want_sigma=True)
    if not r["valid"]:
        return None
    x = np.asarray(r["x"], dtype=float)
    try:
        dof = _gof_dof(model, [x, model.config.suggested_init()])
        gof = float(chi2.sf(ev.q_saturated(x, main, aux), dof)) if dof > 0 else float("nan")
    except Exception:  # noqa: BLE001
        gof = float("nan")
    sig = r["sigma"] if r["sigma"] is not None else np.full(ev.npars, np.nan)
    return SimpleNamespace(bestfit=x, uncertainty=np.asarray(sig, dtype=float), goodness_of_fit=gof,
                           best_twice_nll=2.0 * _evt().pyhf_nll(model, x, d))


def _twin_attempt(why, r):
    """A fit_attempts record of one twin nominal fit from init (``r``: jax_engine fit result, None when the twin was not
    built, ``why`` its build reason)."""
    att = {"engine": "jax", "role": "nominal", "start": "init", "strategy": 0, "tol": 0.1,
           "steps": "per start: MIGRAD, MIGRAD, SIMPLEX + MIGRAD (jax_engine._migrad)", "build": why}
    if r is None:
        return {**att, "valid": False, "error": f"twin not built: {why or 'unknown'}"[:300]}
    return {**att, "valid": bool(r["valid"]), **({"kink": bool(r["kink"])} if "kink" in r else {}),
            "error": None if r["valid"] else "no valid minimum from any start"}


def _fallback_nominal_fit(spec, model, data, shipped=False):
    """Nominal fit when cabinetry failed or is absent: the twin's fit from init (jax engine; ``shipped``: the twin as the
    shipped grader built it), else pyhf's MLE without cabinetry, as a cabinetry-like result -> (result or None,
    [fit_attempts records]); None = unfittable on both engines, each attempt's error in its record."""
    from types import SimpleNamespace
    attempts = []
    if ENGINE == "jax":
        ev, why = _twin_build(spec, model, shipped=shipped)
        r = None
        if ev is not None:
            d = np.asarray(data, dtype=float)
            r = ev.fit(d[:ev.n_main], d[ev.n_main:], want_sigma=True)
        attempts.append(_twin_attempt(why, r))
        if r is not None and r["valid"]:
            attempts[-1]["used"] = True
            sig = r["sigma"] if r["sigma"] is not None else np.full(ev.npars, np.nan)
            return SimpleNamespace(bestfit=np.asarray(r["x"], dtype=float), uncertainty=np.asarray(sig, dtype=float),
                                   goodness_of_fit=float("nan")), attempts
    if cabinetry is not None:
        return None, attempts
    att = {"engine": "pyhf", "role": "nominal", "start": "init", "steps": "MIGRAD, HESSE (pyhf.infer.mle.fit)",
           "valid": False, "error": None}
    attempts.append(att)
    try:
        fitted, twice_nll = pyhf.infer.mle.fit(data, model, return_uncertainties=True, return_fitted_val=True)
        fitted = np.asarray(fitted, dtype=float)
    except Exception as e:  # noqa: BLE001
        att["error"] = fit_error(e)
        return None, attempts
    if fitted.ndim != 2 or not np.all(np.isfinite(fitted[:, 0])):
        att["error"] = "non-finite best fit"
        return None, attempts
    att.update(valid=True, used=True)
    return SimpleNamespace(bestfit=fitted[:, 0], uncertainty=fitted[:, 1], goodness_of_fit=float("nan"),
                           best_twice_nll=float(np.asarray(twice_nll, dtype=float))), attempts


def _fit_failed(seed, attempts):
    """The record of a dataset whose model the grader could not fit: status "fit failed: <the last attempt's error>
    (decisive attempt: <the last attempt>; before it <the earlier attempts' errors>)", every attempt in
    fit_attempts."""
    last = attempts[-1] if attempts else {}
    reason = last.get("error") or "no attempt succeeded"
    where = f"{last.get('engine')} from {last.get('start')}" if last else "none"
    others = "; ".join(f"{a.get('engine')}: {a.get('error')}" for a in attempts[:-1] if a.get("error"))
    status = f"fit failed: {reason} (decisive attempt: {where}" + (f"; before it {others}" if others else "") + ")"
    return {"seed": seed, "status": status[:600], "fit_attempts": attempts}


def _size_gate(seed_dir):
    """The contract's size cap on MODEL.json, before it is parsed (_grid_knobs: the run's task; no task, no cap): above
    it is the model's failure, "invalid workspace: ..." (a record, the contract's failed run on that dataset); None
    within it.  The grid's cap is _grid_stage's (mass_grid "over_cap": the scorer applies the task's contract)."""
    cap = _grid_knobs(seed_dir).get("model_cap")
    try:
        n = os.lstat(Path(seed_dir) / "out" / "MODEL.json").st_size
    except OSError:
        n = None
    if cap is not None and n is not None and n > cap:
        return {"seed": Path(seed_dir).name, "status": f"invalid workspace: MODEL.json is {n / 2 ** 20:.1f} MiB, above "
                f"the {cap / 2 ** 20:g} MiB cap (a larger file counts as a failed run on that dataset)"}
    return None


def mark_errors(rec, shipped=False):
    """``rec`` (a record this grader just wrote) stamped ERRORS_MARKED_KEY = ERRORS_MARKED, unless ``shipped`` (a task
    kept on the rule it shipped with: its record as that grader wrote it).  Returns it."""
    if isinstance(rec, dict) and not shipped:
        rec[ERRORS_MARKED_KEY] = ERRORS_MARKED
    return rec


def _stage_error(e) -> dict:
    """The block of a stage whose grader code raised: its error and the evaluator's mark (module docstring,
    FAILURES)."""
    return {"error": _err(e), "evaluator_error": _err(e)}


def _shipped(seed_dir=None) -> bool:
    """Whether this dataset is graded as its grader shipped, the task being kept on the rule it shipped with (SHIPPED,
    else the task's flag via _grid_knobs)."""
    if SHIPPED is not None:
        return bool(SHIPPED)
    return bool(_grid_knobs(seed_dir).get("shipped"))


def _fit_refusal(seed, model, shipped=False):
    """The record of a dataset whose model no fit can minimise, before any fit is made, or None: a constraint width
    that is not a positive finite number (model.constraint_width_problem), on which pyhf's own fit fails, refused alike
    on both engines and both grading kinds as "fit failed: ...", the model's failure; never under ``shipped`` (the
    shipped grader fitted it)."""
    why = None if shipped else constraint_width_problem(model.config)
    if not why:
        return None
    return {"seed": seed, "status": f"fit failed: {why} (pyhf's fit fails on it, so no fit is made)"[:600],
            "fit_attempts": []}


def _nominal_fit(spec, model, data, shipped=False):
    """Stage 0, MODEL.json's nominal fit (module docstring) -> (the fit result, None when unfittable on every attempt;
    fit_attempts; the cabinetry failure that sent it to _fallback_nominal_fit, or None; the nominal_fit_twin block, or
    None; (twin, its build reason) of a twin built for it, or None).  ``shipped``: the twin as the shipped grader built
    it (jax_engine.build)."""
    nominal_fallback, nominal_twin, res, attempts, built = None, None, None, [], None
    n_free = int(np.sum(~np.asarray(model.config.suggested_fixed(), bool)))
    if ENGINE == "jax" and n_free > PYHF_NOMINAL_MAX_FREE:
        ev0, why0 = _twin_build(spec, model, shipped=shipped)
        built = (ev0, why0)
        res = _twin_nominal_fit(ev0, model, data) if ev0 is not None else None
        nominal_twin = {"engine": "jax", "n_free": n_free, "max_free_pyhf": PYHF_NOMINAL_MAX_FREE,
                        **({} if res is not None else {"failed": why0 or "twin fit invalid"})}
        attempts.append({**_twin_attempt(why0, {"valid": res is not None} if ev0 is not None else None),
                         **({"used": True} if res is not None else {})})
    try:
        if res is not None:
            pass
        elif cabinetry is None:
            raise ImportError("no cabinetry here (a grader runtime): the nominal fit on the grader's engine")
        else:
            res = cabinetry.fit.fit(model, data, goodness_of_fit=True)
            attempts.append({"engine": "cabinetry", "role": "nominal", "start": "init", "strategy": 1, "tol": 0.1,
                             "steps": "MIGRAD (cabinetry.fit.fit)", "valid": True, "error": None, "used": True})
    except Exception as e:
        nominal_fallback = f"cabinetry: {type(e).__name__}: {e}"[:300]
        if cabinetry is not None:
            attempts.append({"engine": "cabinetry", "role": "nominal", "start": "init", "strategy": 1, "tol": 0.1,
                             "steps": "MIGRAD (cabinetry.fit.fit)", "valid": False, "error": fit_error(e)})
        res, more = _fallback_nominal_fit(spec, model, data, shipped)
        attempts += more
    return res, attempts, nominal_fallback, nominal_twin, built


def _twin_measurement(spec, model, data, res, built=None, shipped=False):
    """(twin, why the stages stay on pyhf, its observed-data fit, the record's twin_fit block) on the jax engine: the
    model's twin (``built``: the one the nominal fit built, else built here) and its observed-data fit from the nominal
    fit's point (_twin_obs_fit), the start of every twin stage and the toys' generation point.  A twin not built, or
    whose fit is invalid, leaves the stages on pyhf with the reason: (None, why, None, None).  (None, None, None, None)
    on the pyhf engine."""
    if ENGINE != "jax":
        return None, None, None, None
    ev, ev_reason = built if built is not None else _twin_build(spec, model, shipped=shipped)
    if ev is None:
        return None, ev_reason, None, None
    poi, n_main = model.config.poi_index, model.config.nmaindata
    d = np.asarray(data, dtype=float)
    obs_fit = _twin_obs_fit(ev, d[:n_main], d[n_main:], res.bestfit)
    if not obs_fit["valid"]:
        return None, "observed-data fit invalid on the twin", None, None
    twin_fit = {"mu": float(obs_fit["x"][poi]),
                "twice_nll": float(2.0 * (obs_fit["nll"] + ev.full_nll_const(d[:n_main], d[n_main:]))),
                "delta_mu_vs_pyhf": float(obs_fit["x"][poi] - float(res.bestfit[poi])),
                "min_expected_rate": obs_fit["min_rate"],
                "n_floor": obs_fit["n_floor"],
                "n_sample_floor": obs_fit["n_sample_floor"],
                "kink": obs_fit["kink"],
                "from": obs_fit["from"]}
    return ev, ev_reason, obs_fit, twin_fit


def _mu_measurement(model, data, res, ev=None, obs_fit=None) -> dict:
    """The scored mu measurement (mu_fit_block); a failure costs this block (mu unscorable on this dataset): the
    model's when its fit raised (ModelFitError, "fit failed: ..."), else the evaluator's (_stage_error)."""
    try:
        return mu_fit_block(model, data, res, ev=ev, obs_fit=obs_fit)
    except ModelFitError as e:
        return {"error": f"fit failed: {e}"}
    except Exception as e:
        return _stage_error(e)


def _guarded_grid_stage(rec, out_dir, reported_mh, model_spec, shipped=False):
    """_grid_stage in a guard of its own: a raise there outside its own guards (the grid's files, say) is the
    evaluator's, on mass_grid (_stage_error; a scanning task's block carries scan_version); ``shipped``: raised, as the
    shipped grader did."""
    try:
        _grid_stage(rec, out_dir, reported_mh, model_spec=model_spec)
    except Exception as e:
        if shipped:
            raise
        scan = bool(_grid_knobs(Path(out_dir).parent).get("scan"))
        rec["mass_grid"] = {**_stage_error(e), **({"scan_version": SCAN_VERSION} if scan else {})}


def _guarded_crosschecks(rec, self_quality, claimed_zexp, shipped=False):
    """_crosschecks in a guard of its own: a raise in its code is the evaluator's, on quality_crosscheck (_stage_error,
    beside any cross-check already made); ``shipped``: raised."""
    try:
        _crosschecks(rec, self_quality, claimed_zexp, shipped=shipped)
    except Exception as e:
        if shipped:
            raise
        qc = rec.get("quality_crosscheck")
        rec["quality_crosscheck"] = {**(qc if isinstance(qc, dict) else {}), **_stage_error(e)}


def _combine_grid(rec, out_dir, reported_mh, spec, shipped=False):
    """A combine-graded record's mass_grid (COMBINE_GRID_RULE): with a grid declared (model.grid_bytes: MODEL_GRID/ or
    MODEL_GRID.json there) combine's own MH scan moves to combine_mass_grid and the declared grid goes through stage 6
    against MODEL.json as on the MODEL.json path; without one combine's scan (its model floating MH) stays as mass_grid.
    The mass_grid that results is stamped declared_grid (whether a grid was declared) and combine_grid_rule."""
    declared = grid_bytes(out_dir) is not None
    if declared:
        if "mass_grid" in rec:
            rec["combine_mass_grid"] = rec.pop("mass_grid")
        _guarded_grid_stage(rec, out_dir, reported_mh, spec, shipped)
    if isinstance(rec.get("mass_grid"), dict):
        rec["mass_grid"].update(declared_grid=declared, combine_grid_rule=COMBINE_GRID_RULE)


def _pyhf_crosscheck_shipped(model, data, mu_combine):
    """MODEL.json's pyhf best fit for a combine-graded seed (r-hat vs mu-hat, negative-rate flag), as the shipped grader
    made it (its own cabinetry fit).  Report-only; never raises."""
    try:
        res = cabinetry.fit.fit(model, data)
        poi = model.config.poi_index
        nu = np.asarray(model.expected_actualdata(pyhf.tensorlib.astensor(res.bestfit)), dtype=float)
        return {"mu": float(res.bestfit[poi]), "mu_unc": float(res.uncertainty[poi]),
                "delta_mu_combine_minus_pyhf": float(mu_combine - res.bestfit[poi]),
                "min_expected_rate": float(nu.min()), "negative_expected_rate": bool(nu.min() < 0),
                "n_negative_rate_bins": int(np.sum(nu < 0))}
    except Exception as e:
        return {"error": _err(e)}


def _combine_record_shipped(seed_dir, model, data, fields, combine_blk, reported_mh, self_quality, claimed_zexp):
    """The record of a dataset graded in combine as the shipped grader wrote it (SHIPPED; a task kept on the rule it
    shipped with): combine_battery's ``fields``, MODEL.json's own cabinetry fit beside combine's r
    (_pyhf_crosscheck_shipped), the claims read with float(), and a declared pyhf grid profiled with no link where
    combine's model does not float MH (or, on a task scored on the mass-marginal mu, beside combine's own scan); no
    nominal fit of MODEL.json, no mu_fit, no stage guards."""
    rec = {"seed": seed_dir.name, "status": "ok", **fields}
    rec["pyhf_crosscheck"] = _pyhf_crosscheck_shipped(model, data, rec["mu_model"])
    _crosschecks(rec, self_quality, claimed_zexp, shipped=True)
    if "mass_grid" not in rec:
        _grid_stage(rec, seed_dir / "out", reported_mh)
    elif _grid_knobs(seed_dir)["scan"] and _load_grid(seed_dir / "out"):
        rec["combine_mass_grid"] = rec.pop("mass_grid")
        _grid_stage(rec, seed_dir / "out", reported_mh)
    rec["combine_model"] = combine_blk
    return rec


def _combine_record(seed_dir, spec, model, data, fields, combine_blk, reported_mh, self_quality, claimed_zexp,
                    shipped=False):
    """The record of a dataset graded in combine (stage 7): combine_battery's ``fields`` (engine "combine") and
    MODEL.json measured exactly as on the MODEL.json path: its nominal fit (_nominal_fit; unfittable is _fit_failed's
    "fit failed" record, combine's block beside it: the model's failure as on any dataset), the twin's observed-data fit
    on the jax engine (twin_fit), mu_fit (the scored mu, _mu_measurement; the twin's reason for leaving it on pyhf in
    its engine_fallback) and pyhf_fit.twice_nll (the grid link's fit route); then pyhf_crosscheck (MODEL.json's nominal
    fit beside combine's r, report-only), the claims (_crosschecks) and the grid (_combine_grid), each in its own
    guard.  ``shipped``: the shipped grader's record (_combine_record_shipped)."""
    if shipped:
        return _combine_record_shipped(seed_dir, model, data, fields, combine_blk, reported_mh, self_quality,
                                       claimed_zexp)
    refused = _fit_refusal(seed_dir.name, model)
    if refused is not None:
        return {**refused, "combine_model": combine_blk}
    res, attempts, nominal_fallback, nominal_twin, built = _nominal_fit(spec, model, data)
    if res is None:
        rec = _fit_failed(seed_dir.name, attempts)
        rec["combine_model"] = combine_blk
        return rec
    rec = {"seed": seed_dir.name, "status": "ok", **fields,
           **({"nominal_fit_fallback": nominal_fallback} if nominal_fallback else {}),
           **({"nominal_fit_twin": nominal_twin} if nominal_twin else {}),
           "fit_attempts": attempts}
    tn = getattr(res, "best_twice_nll", None)
    if tn is not None and np.isfinite(float(tn)):
        rec["pyhf_fit"] = {"twice_nll": float(tn)}
    ev, ev_reason, obs_fit, twin_fit = _twin_measurement(spec, model, data, res, built)
    if twin_fit is not None:
        rec["twin_fit"] = twin_fit
    rec["mu_fit"] = _mu_measurement(model, data, res, ev=ev, obs_fit=obs_fit)
    if ev_reason:
        rec["mu_fit"]["engine_fallback"] = ev_reason
    if twin_fit is not None:
        twin_fit["sigma_hess"] = rec["mu_fit"].get("sigma_hess")
    rec["pyhf_crosscheck"] = _pyhf_crosscheck(model, res, rec.get("mu_model"))
    _guarded_crosschecks(rec, self_quality, claimed_zexp, shipped)
    _combine_grid(rec, seed_dir / "out", reported_mh, spec, shipped)
    rec["combine_model"] = combine_blk
    return rec


def verify_seed(seed_dir, result_name, mu_truth=None):
    mj = seed_dir / "out" / "MODEL.json"
    rj = seed_dir / "out" / result_name
    if not os.path.lexists(mj):
        return {"seed": seed_dir.name, "status": "no MODEL.json"}
    over = _size_gate(seed_dir)
    if over is not None:
        return over
    try:
        spec = load_output_json(mj, max_bytes=_grid_knobs(seed_dir).get("model_cap"))
        meta = spec.pop("_meta", None)
        ws = pyhf.Workspace(spec)
        model = ws.model()
        data = ws.data(model)
    except Exception as e:
        return {"seed": seed_dir.name, "status": f"invalid workspace: {type(e).__name__}: {e}"}
    shipped = _shipped(seed_dir)
    reported, reported_mh, self_quality, claimed_zexp = _read_result(rj)
    window, window_info = _window_mask(model, meta)

    combine_blk, combine_why = None, None
    if (seed_dir / "out" / "MODEL_COMBINE").is_dir():
        if not COMBINE_BATTERY:
            combine_blk = {"skipped": "SBH_COMBINE_BATTERY=0"}
        else:
            try:
                fields, combine_blk, combine_why = combine_battery(
                    seed_dir / "out", spec, reported, reported_mh, mu_truth=mu_truth, n_toys=N_TOYS,
                    gof_fallback=True)
            except Exception as e:
                fields, combine_blk, combine_why = (None, {"error": _err(e)} if shipped else _stage_error(e),
                                                    f"battery error: {_err(e)}")
            if fields is not None:
                return _combine_record(seed_dir, spec, model, data, fields, combine_blk, reported_mh, self_quality,
                                       claimed_zexp, shipped)

    poi = model.config.poi_index
    n_main = model.config.nmaindata
    d = np.asarray(data, dtype=float)
    rec = _fit_refusal(seed_dir.name, model, shipped)
    res, attempts, nominal_fallback, nominal_twin, built = ((None, [], None, None, None) if rec is not None
                                                            else _nominal_fit(spec, model, data, shipped))
    if res is None:
        rec = rec or _fit_failed(seed_dir.name, attempts)
        if combine_blk is not None:
            rec["combine_model"] = combine_blk
        return rec
    mu_hat = float(res.bestfit[poi])
    mu_unc = float(res.uncertainty[poi])
    gof_p = float(res.goodness_of_fit)

    consistency = None
    if reported:
        try:
            tol = 0.25 * max(reported.get("total", 0.3), 0.05) + 0.05
            consistency = {"mu_model": mu_hat, "mu_model_unc": mu_unc,
                           "mu_reported": reported["value"],
                           "difference": mu_hat - reported["value"],
                           "tolerance": tol,
                           "pass": bool(abs(mu_hat - reported["value"]) <= tol)}
        except (KeyError, TypeError, ValueError, OverflowError):
            consistency = None

    nu_hat = np.asarray(model.expected_actualdata(pyhf.tensorlib.astensor(res.bestfit)), dtype=float)
    try:
        dof = _gof_dof(model, [res.bestfit, model.config.suggested_init()])
    except Exception:
        dof = None
    rec = {"seed": seed_dir.name, "status": "ok",
           **({"nominal_fit_fallback": nominal_fallback} if nominal_fallback else {}),
           **({"nominal_fit_twin": nominal_twin} if nominal_twin else {}),
           "fit_attempts": attempts,
           "mu_model": mu_hat, "mu_model_unc": mu_unc,
           "consistency": consistency,
           "gof_p_saturated": gof_p,
           "gof_dof": dof,
           "min_expected_rate": float(nu_hat.min()),
           "negative_expected_rate": bool(nu_hat.min() < 0),
           "n_negative_rate_bins": int(np.sum(nu_hat < 0))}
    tn = getattr(res, "best_twice_nll", None)
    if tn is not None and np.isfinite(float(tn)):
        rec["pyhf_fit"] = {"twice_nll": float(tn)}

    ev, ev_reason, obs_fit, twin_fit = _twin_measurement(spec, model, data, res, built, shipped)
    if twin_fit is not None:
        rec["twin_fit"] = twin_fit
    rec["engine"] = "jax" if ev is not None else "pyhf"
    if combine_why:
        ev_reason = "; ".join(x for x in (f"combine: {combine_why}", ev_reason) if x)
    if ev_reason:
        rec["engine_fallback"] = ev_reason
    rec["gof_window"] = window_info

    rec["mu_fit"] = _mu_measurement(model, data, res, ev=ev, obs_fit=obs_fit)
    if ev is not None and isinstance(rec["mu_fit"], dict):
        rec["twin_fit"]["sigma_hess"] = rec["mu_fit"].get("sigma_hess")

    import time
    t0 = time.perf_counter()
    try:
        if SKIP_IMPACTS:
            impacts, info = None, {"skipped": "SKIP_IMPACTS (the grader stage: not scored)"}
        elif ev is not None:
            impacts, info = ev.postfit_impacts(d[:n_main], d[n_main:], res.bestfit, res.uncertainty)
        else:
            info = {"engine": "pyhf"}
            try:
                impacts = _pyhf_postfit_impacts(model, data, res)
            except Exception as e:
                impacts = {"error": str(e)}
            info["wall_s"] = round(time.perf_counter() - t0, 2)
    except Exception as e:  # noqa: BLE001
        if shipped:
            raise
        impacts, info = _stage_error(e), {"engine": rec["engine"], "wall_s": round(time.perf_counter() - t0, 2)}
    rec["impacts"] = impacts
    rec["impacts_info"] = info

    try:
        z_exp, rec["z_exp_info"] = _z_exp(model, res, ev=ev, obs_fit=obs_fit)
    except Exception as e:  # noqa: BLE001
        if shipped:
            raise
        z_exp, rec["z_exp_info"] = _stage_error(e), {"engine": rec["engine"]}
    rec["z_exp_asimov"] = z_exp
    grid_first = STAGE_DEADLINE is not None or GRID_DEADLINE is not None
    if grid_first:
        _guarded_grid_stage(rec, seed_dir / "out", reported_mh, spec, shipped)

    gen_floor = None if shipped else GEN_RATE_FLOOR
    if (N_TOYS_FIT if N_TOYS_FIT is not None else N_TOYS) > 0:
        try:
            rec["toys"] = toy_ensembles(model, data, res, mu_hat, ev=ev, obs_fit=obs_fit, dof=dof, window=window,
                                        window_info=window_info, gen_floor=gen_floor, check_rates=not shipped)
        except ModelFitError as e:
            rec["toys"] = {"error": f"fit failed: {e}"}
        except Exception as e:
            rec["toys"] = {"error": f"{type(e).__name__}: {e}", "evaluator_error": _err(e)}
        if isinstance(rec["toys"], dict):
            rec["toys"]["engine"] = rec["engine"]
            if ev_reason:
                rec["toys"]["engine_fallback"] = ev_reason
    if N_TOYS > 0 and mu_truth is not None and not (SEQ_GOF and SEQ_GOF["mode"] == "score"):
        try:
            try:
                ft = freq_coverage(model, data, float(mu_truth), ev=ev, obs_fit=obs_fit, dof=dof, window=window,
                                   window_info=window_info, gen_floor=gen_floor, check_rates=not shipped)
                ft["engine"] = rec["engine"]
                if ev_reason:
                    ft["engine_fallback"] = ev_reason
            except _TwinFitError as e:
                ft = freq_coverage(model, data, float(mu_truth), dof=dof, window=window, window_info=window_info,
                                   gen_floor=gen_floor, check_rates=not shipped)
                ft["engine"], ft["engine_fallback"] = "pyhf", str(e)
            rec["freq_toys"] = ft
        except ModelFitError as e:
            rec["freq_toys"] = {"error": f"fit failed: {e}"}
        except Exception as e:
            rec["freq_toys"] = _stage_error(e)
    if SEQ_GOF and mu_truth is not None:
        import sys
        cfg = SEQ_GOF
        if STAGE_DEADLINE is not None:
            left = max(STAGE_DEADLINE - time.monotonic(), 1.0)
            cfg = dict(SEQ_GOF, max_wall_s=min(left, SEQ_GOF.get("max_wall_s") or left))
        try:
            _seq_gof_mod.grader_stage(cfg, rec, seed_dir, model, data, float(mu_truth), ev=ev, obs_fit=obs_fit,
                                      dof=dof, grader=sys.modules[__name__], ev_reason=ev_reason,
                                      check_rates=not shipped)
        except Exception as e:  # noqa: BLE001
            if shipped:
                raise
            rec["freq_toys" if cfg["mode"] == "score" else "gof_seq"] = {**_stage_error(e), "seq_mode": cfg["mode"]}
    _guarded_crosschecks(rec, self_quality, claimed_zexp, shipped)
    if not grid_first:
        _guarded_grid_stage(rec, seed_dir / "out", reported_mh, spec, shipped)
    if combine_blk is not None:
        rec["combine_model"] = combine_blk
    return rec


_GRID_KNOBS = {}


def _task_of_seed(seed_dir):
    """The task named by one seed's record.json (its dataset's suite manifest), or None."""
    try:
        ds = json.loads((Path(seed_dir) / "record.json").read_text()).get("dataset")
        if ds:
            _repo_on_path()
            from sbh.suite import load_manifest
            return load_manifest(Path(ds).parents[1]).get("task")
    except Exception:  # noqa: BLE001
        pass
    return None


def _grid_knobs(seed_dir=None):
    """{"task", "mass_range", "yield_curve", "scan", "link_identical", "model_cap", "grid_cap", "shipped"} of the run's
    task:
    TASK_NAME, else the seed's record.json; a task without a mass range leaves both None (every point in range, no
    content checks).  The yield curve comes from Task.grid_yield_curve() (None, or a task without the
    method, = checks skipped).  ``scan``: the per-point mu scans run only for a task scored on the mass-marginal mu
    (Task.mu_marginal_over_mass); the other tasks read only the 2 DeltalnL curve, and the scans (~0.6 s a point) would
    only eat their grader's GRID_FRAC budget.  An unknown task (the grader stage's staged dataset carries no
    record.json) scans nothing either.  ``link_identical`` (Task.grid_link_identical, True for an unknown task):
    _grid_link's identical-workspace route applies.  The size caps (the contract's: model.MODEL_MAX_BYTES on MODEL.json,
    model.grid_cap on the grid) apply to a known task; an unknown one reads what the grader stage staged within its
    caps.  ``shipped``: the task keeps the rule it shipped with (Task.score_rule SHIPPED_RULE), so its datasets are
    graded as its grader shipped (SHIPPED)."""
    name = TASK_NAME or (_task_of_seed(seed_dir) if seed_dir is not None else None)
    if not name:
        return {"task": None, "mass_range": None, "yield_curve": None, "scan": False, "link_identical": True,
                "model_cap": None, "grid_cap": None, "shipped": False}
    if name not in _GRID_KNOBS:
        knobs = {"task": name, "mass_range": None, "yield_curve": None, "scan": False, "link_identical": True,
                 "model_cap": None, "grid_cap": None, "shipped": False}
        try:
            _repo_on_path()
            from sbh.task import load_task
            task = load_task(name)
            mr = getattr(task, "mass_range", None)
            knobs["mass_range"] = (float(mr[0]), float(mr[1])) if mr is not None else None
            curve_fn = getattr(task, "grid_yield_curve", None)
            knobs["yield_curve"] = curve_fn() if callable(curve_fn) else None
            knobs["scan"] = bool(getattr(task, "mu_marginal_over_mass", False))
            knobs["link_identical"] = bool(getattr(task, "grid_link_identical", True))
            knobs["model_cap"], knobs["grid_cap"] = MODEL_MAX_BYTES, grid_cap(task)
            from sbh.score import SHIPPED_RULE, score_rule
            knobs["shipped"] = score_rule(task) == SHIPPED_RULE
        except Exception as e:  # noqa: BLE001
            knobs["error"] = _err(e)
        _GRID_KNOBS[name] = knobs
    return _GRID_KNOBS[name]


def _grid_stage(rec, out_dir, reported_mh, model_spec=None):
    """Declared mH grid -> mass profile + the per-point mu scans (scored; a failure costs only this block).  With
    ``model_spec`` every point is compared with MODEL.json and the grid linked to it (_grid_link: the grid's own
    workspace nearest the reported mH is MODEL.json's, where the task takes that route (LINK_IDENTICAL, else
    _grid_knobs' link_identical), else its fitted point there reproduces MODEL.json's fit).  The task's mass range,
    yield curve, whether to scan and the grid's size cap come from _grid_knobs: a grid above its cap is not read, its
    block {"over_cap": why} (the scorer applies the task's contract, Task.grid_over_cap_fails: a failed run on that
    dataset, or no grid).  PRIOR_CURVE adds sigma0_ref.  A scanning task's mass_grid block carries scan_version, also
    on error (the scorer treats a block without it as verified before the scan).  An exception of the grader's own
    code here is the evaluator's: mass_grid {"error", "evaluator_error"}."""
    knobs = _grid_knobs(Path(out_dir).parent)
    try:
        grid = _load_grid(out_dir, max_bytes=knobs.get("grid_cap"))
    except GridOverCap as e:
        why = f"{e}: not read (the contract's size cap)"
        rec["mass_grid"] = {"over_cap": why, "usable": False, "usable_reason": why,
                            **({"scan_version": SCAN_VERSION} if knobs["scan"] else {})}
        return
    if grid:
        try:
            shipped = _shipped(Path(out_dir).parent)
            rec["mass_grid"] = mass_grid_check(grid, reported_mh, engine=ENGINE, model_spec=model_spec,
                                               mass_range=knobs["mass_range"], yield_curve=knobs["yield_curve"],
                                               prior_curve=PRIOR_CURVE, scan=knobs["scan"],
                                               point_budget=not shipped, shipped=shipped)
        except Exception as e:
            rec["mass_grid"] = {"error": _err(e), "evaluator_error": _err(e),
                                **({"scan_version": SCAN_VERSION} if knobs["scan"] else {})}
        if model_spec is not None:
            identical = knobs.get("link_identical", True) if LINK_IDENTICAL is None else bool(LINK_IDENTICAL)
            _grid_link(rec, reported_mh, points=grid if identical else None, model_spec=model_spec)


def _load_point(spec_or_path):
    """(spec without _meta, _meta) of one grid point (model.read_point: a file never read through a symlink)."""
    spec = read_point(spec_or_path)
    if not isinstance(spec, dict):
        raise ValueError(f"a grid workspace must be a JSON object, not {type(spec).__name__}")
    spec = dict(spec)
    meta = spec.pop("_meta", None)
    return spec, meta


def _point_what(mh) -> str:
    """What a grid point's budget says when it runs out (sbh.budget.budget's message starts with it)."""
    return f"the grid point at mH {float(mh):g}"


@contextlib.contextmanager
def _point_budget(mh, on=True):
    """The grid point at ``mh`` within what is left of GRID_DEADLINE (none without a deadline, or ``on`` False)."""
    if not on or GRID_DEADLINE is None:
        yield
        return
    import time
    with budget(max(GRID_DEADLINE - time.monotonic(), 1e-3), _point_what(mh)):
        yield


def _point_ran_out(e, mh) -> bool:
    """Whether BudgetExceeded ``e`` is the grid point's own budget (_point_budget), not an outer one."""
    return str(e).startswith(_point_what(mh) + " ran out")


def mass_grid_check(points, reported_mh, engine="pyhf", model_spec=None, mass_range=None, yield_curve=None,
                    prior_curve=False, scan=True, point_budget=True, shipped=False):
    """Mass profile and per-point mu scans from the fixed-mH workspaces.  Every point in ``mass_range`` (all, when
    None) is profile-fitted (mu, nuisances free; the Hessian width of mu), then scanned conditionally in mu
    (scan_point: mu_scan, the single-mass likelihood the marginal rule reads); out-of-range points are listed, not
    fitted.  The 2*NLL curve of the fitted points gives 2*DeltalnL, the parabolic minimum -> mh_model and the
    2*DeltalnL = 1 crossings -> sigma_lo/sigma_hi, compared report-only with RESULT.json's mH.  Fits walk outward
    from the reported mH, warm-started from the neighbour.  engine="jax": fits on the twin (_GridTwin).  With
    ``model_spec`` (MODEL.json without _meta) every point is compared with it (_grid_signature); a mismatch or a
    fixed POI sets grid_uniform False.  ``yield_curve`` (Task.grid_yield_curve()) drives the content checks
    (grid_content: yield band, centroid, smoothness, dead points), None skips them; ``prior_curve`` adds sigma0_ref
    per point (the reference run).  ``scan`` False (a task not scored on the marginal, _grid_knobs) skips the mu
    scans and sigma0_ref and leaves scan_version out.  Under GRID_DEADLINE no point is started after it and, with
    ``point_budget`` (every task but a shipped-rule one), each point's fit and scan run within what is left of it
    (_point_budget): a point stopped there is a hole.  A point no fit can minimise is a failed point, the model's,
    before it is fitted (a constraint width that is not a positive finite number, model.constraint_width_problem, as
    on MODEL.json), and so is a point whose fit ends at a -2 ln L that is not a finite number; ``shipped`` (a task kept
    on the rule it shipped with): neither, the twin as that grader built it (_GridTwin) and its tie set
    (grid_coverage).  Output: the older keys (curve, mu_hat, twice_nll_abs, point_engines, aligned with the fitted
    points) plus scan_version, mass_range, points, coverage, content, scan_s."""
    import time
    t0 = time.perf_counter()
    twin = _GridTwin(shipped=shipped) if engine == "jax" else None
    n_by = {"jax": 0, "pyhf": 0}
    mh_rep = sig_rep = None
    if isinstance(reported_mh, dict) and reported_mh.get("value") is not None:
        mh_rep = float(reported_mh["value"])
        sig_rep = float(reported_mh.get("total") or 0.0) or None
    mh_sorted = sorted(points)
    rng = None if mass_range is None else (float(mass_range[0]), float(mass_range[1]))
    in_range = [rng is None or (rng[0] - 1e-9 <= mh <= rng[1] + 1e-9) for mh in mh_sorted]
    fit_idx = [k for k, ok in enumerate(in_range) if ok]
    rows = [{"mass": float(mh), "in_range": bool(ok), "status": "out_of_range" if not ok else "failed",
             "mu_hat": None, "twice_nll_abs": None, "nll_min": None, "sigma_hess": None, "bounds": None,
             "mu_scan": None, "yield_mu1": None, "centroid": None, "engine": None}
            for mh, ok in zip(mh_sorted, in_range)]
    order = []
    start = None
    if fit_idx:
        centre = mh_rep if mh_rep is not None else float(np.median([mh_sorted[k] for k in fit_idx]))
        s = int(np.argmin([abs(mh_sorted[k] - centre) for k in fit_idx]))
        start = fit_idx[s]
        up, dn = fit_idx[s + 1:], fit_idx[:s][::-1]
        order = [start] + up + dn
        if GRID_DEADLINE is not None:
            order = [start] + [k for pair in zip(up, dn) for k in pair] + up[len(dn):] + dn[len(up):]
    results, n_failed, mu_at, eng_at = {}, 0, {}, {}
    n_budget_cut, scan_s = 0, 0.0
    uniform, mismatch, sig_ref, sig_ref_mh = True, None, None, None
    if model_spec is not None:
        sig_ref, sig_ref_mh = _grid_signature(model_spec), "MODEL.json"
    specs, trusted = {}, set()
    for k, mh in enumerate(mh_sorted):
        try:
            spec, meta = _load_point(points[mh])
            sig = _grid_signature(spec)
            facts = template_facts(spec, meta, mass_range=rng)
        except Exception as e:  # noqa: BLE001
            rows[k]["why"] = f"unreadable: {_err(e)}"
            continue
        if model_spec is not None and sig == sig_ref:
            trusted.add(k)
        if sig_ref is None:
            sig_ref, sig_ref_mh = sig, mh
        elif uniform:
            for key in sig_ref:
                if sig[key] != sig_ref[key]:
                    uniform = False
                    mismatch = (f"mH_{mh:.2f}: {key} differ from "
                                + (sig_ref_mh if isinstance(sig_ref_mh, str) else f"mH_{sig_ref_mh:.2f}"))
                    break
        if uniform and _fixed_poi(spec):
            uniform = False
            mismatch = f"mH_{mh:.2f}: the POI {_fixed_poi(spec)!r} is fixed (every point is fitted with it free)"
        rows[k].update(facts)
        try:
            cfg = _measurement_cfg(spec)
            poi = cfg.get("poi")
            b = next((p.get("bounds") for p in cfg.get("parameters") or [] if isinstance(p, dict)
                      and p.get("name") == poi), None)
            rows[k]["bounds"] = [float(b[0][0]), float(b[0][1])] if b else None
        except Exception:  # noqa: BLE001
            pass
        if in_range[k]:
            specs[k] = spec
    init = {}

    def fit_point(k, spec):
        """((model, the fit, its engine, the scan's fields, the scan's seconds), None), or (None, why) where the point's
        fit fails; nothing recorded here, so a point cut by its budget leaves no trace but its hole."""
        try:
            ws = pyhf.Workspace(spec, validate=k not in trusted)
            model = ws.model(validate=k not in trusted)
            data = ws.data(model)
            refused = None if shipped else constraint_width_problem(model.config)
            if refused:
                return None, f"fit failed: {refused}"
            init_pars = init.get(k)
            if init_pars is not None and len(init_pars) != len(model.config.suggested_init()):
                init_pars = None
            got = twin.fit_point(spec, model, data, init_pars) if twin is not None else None
            eng_k = "jax" if got is not None else "pyhf"
            if got is None:
                got = _pyhf_point(model, data, init_pars)
        except Exception as e:  # noqa: BLE001
            return None, f"fit failed: {fit_error(e) if _PYHF_HESSE_BUG in str(e) else _err(e)}"
        if not shipped and not np.isfinite(float(got["twice_nll_abs"])):
            return None, f"fit failed: its -2 ln L at the fit is {float(got['twice_nll_abs']):g}, not a finite number"
        ts = time.perf_counter()
        try:
            upd = scan_point(model, data, got, ev=got.get("em"), prior_curve=prior_curve) if scan else {}
        except Exception as e:  # noqa: BLE001
            upd = {"mu_scan": None, "scan_error": _err(e)}
        return (model, got, eng_k, upd, time.perf_counter() - ts), None

    for k in order:
        if GRID_DEADLINE is not None and time.monotonic() > GRID_DEADLINE:
            n_budget_cut += 1
            rows[k]["why"] = "wall budget ran out"
            continue
        mh = mh_sorted[k]
        spec = specs.get(k)
        if spec is None:
            n_failed += 1
            continue
        try:
            with _point_budget(mh, point_budget):
                done, why = fit_point(k, spec)
        except BudgetExceeded as e:
            if not _point_ran_out(e, mh):
                raise
            n_budget_cut += 1
            rows[k]["why"] = "wall budget ran out"
            continue
        if done is None:
            n_failed += 1
            rows[k]["why"] = why
            continue
        model, got, eng_k, upd, dt = done
        n_by[eng_k] += 1
        eng_at[float(mh)] = eng_k
        pars, tnll = got["x"], got["twice_nll_abs"]
        results[mh] = float(tnll)
        poi = model.config.poi_index
        try:
            mu_at[float(mh)] = float(np.asarray(pars, dtype=float)[poi])
        except Exception:  # noqa: BLE001
            mu_at[float(mh)] = None
        rows[k].update(status="ok", mu_hat=mu_at[float(mh)], twice_nll_abs=results[mh], nll_min=got["nll_min"],
                       sigma_hess=got["sigma_hess"], engine=eng_at[float(mh)],
                       bounds=[float(b) for b in model.config.suggested_bounds()[poi]])
        rows[k].update(upd)
        scan_s += dt
        pos = fit_idx.index(k)
        if k >= start:
            nxt = fit_idx[pos + 1] if pos + 1 < len(fit_idx) else None
        else:
            nxt = fit_idx[pos - 1] if pos > 0 else None
        warm = [float(x) for x in np.asarray(pars, dtype=float)]
        if nxt is not None:
            init[nxt] = warm
        if k == start and pos > 0:
            init[fit_idx[pos - 1]] = warm
    curve = [[float(mh), results[mh]] for mh in mh_sorted if mh in results]
    eng = {"engine": ("pyhf" if not n_by["jax"] else "jax" if not n_by["pyhf"] else "mixed")}
    if n_budget_cut:
        eng["n_budget_cut"] = n_budget_cut
    if twin is not None:
        eng.update(engine_points=n_by, n_compiled=twin.n_compiled, n_shared=twin.n_shared,
                   n_floor=twin.n_floor)
        if twin.fallbacks:
            eng["engine_fallback"] = sorted(set(twin.fallbacks))[:3]
    model_yield = template_facts(model_spec)["yield_mu1"] if isinstance(model_spec, dict) else None
    content = grid_content(rows, rng, yield_curve, model_mass=mh_rep, model_yield=model_yield)
    for r in rows:
        r.pop("lineshape", None)
    coverage, off = grid_coverage(rows, rng, model_mass=mh_rep, shipped=shipped)
    content["model_off_minimum"] = off
    grid_blk = {"scan_version": SCAN_VERSION, "mass_range": list(rng) if rng else None, "points": rows,
                "coverage": coverage, "content": content, "scan_s": round(scan_s, 2),
                "n_out_of_range": int(len(mh_sorted) - len(fit_idx))}
    if not scan:
        grid_blk.pop("scan_version")
    if len(curve) < 3:
        return {"n_points": len(curve), "n_failed": n_failed,
                "grid_uniform": bool(uniform), "grid_mismatch": mismatch,
                "error": "fewer than 3 usable grid points", "usable": False,
                "usable_reason": "fewer than 3 fitted points", **eng, **grid_blk}
    out = {**_interval_summary(curve, mh_rep, sig_rep, uniform=uniform, mismatch=mismatch,
                               n_failed=n_failed),
           "fit_s": round(time.perf_counter() - t0, 2), **eng}
    out["mu_hat"] = [mu_at.get(float(m)) for m, *_ in out["curve"]]
    out["twice_nll_abs"] = [results.get(float(m)) for m, *_ in out["curve"]]
    out["point_engines"] = [eng_at.get(float(m)) for m, *_ in out["curve"]]
    out["engine_points"] = n_by
    out.update(grid_blk)
    try:
        from sbh.kinds.binned_likelihood.scoring import grid_profile
        gp = grid_profile(out)
        out.update(usable=gp["usable"], usable_reason=gp["reason"], sigma_root=gp.get("sigma_root"),
                   sigma_lo_root=gp.get("sigma_lo_root"), sigma_hi_root=gp.get("sigma_hi_root"))
    except Exception as e:  # noqa: BLE001
        out["usable_error"] = f"{type(e).__name__}: {e}"
    return out
COMBINE_BATTERY = os.environ.get("SBH_COMBINE_BATTERY", "1").strip().lower() not in ("0", "off", "no", "false")


GOF_WINDOW = None


def gof_window():
    """The GOF window in force (GOF_WINDOW, set from the task); a verification without one is an evaluator error."""
    if GOF_WINDOW is None:
        raise ValueError("no GOF window: the task states none (Task.gof_window)")
    return tuple(GOF_WINDOW)


def _window_mask(model, meta, window=None):
    """(mask over main bins in pyhf channel order, info): bin centre from `_meta.binning[<ch>].edges` in [lo, hi),
    the window given or gof_window().  A channel without usable edges contributes no bin (noted)."""
    lo, hi = window if window is not None else gof_window()
    binning = meta.get("binning") if isinstance(meta, dict) else None
    binning = binning if isinstance(binning, dict) else {}
    mask, notes = [], []
    for ch in model.config.channels:
        nb = int(model.config.channel_nbins[ch])
        b = binning.get(ch)
        try:
            e = np.asarray(b.get("edges") if isinstance(b, dict) else None, dtype=float)
        except (TypeError, ValueError):
            e = np.asarray([])
        if e.ndim != 1 or len(e) != nb + 1 or not np.all(np.isfinite(e)):
            mask += [False] * nb
            notes.append(f"{ch}: no usable _meta.binning edges")
            continue
        c = 0.5 * (e[1:] + e[:-1])
        mask += [bool(x) for x in ((c >= lo) & (c < hi))]
    mask = np.asarray(mask, dtype=bool)
    return mask, {"window": [float(lo), float(hi)], "n_bins": int(mask.sum()), "notes": notes[:5]}


GOF_AUX = os.environ.get("SBH_GOF_AUX", "held").strip().lower()


def _gof_pvalues(q_obs: dict, q_toys: dict, dof, window_info: Optional[dict]):
    """Scored GOF p's of one ensemble, each (k + 1) / (N + 1): p_glob (``saturated``), p_sr (``window``) and
    p_comb = rank of the observed min(p_glob, p_sr) among every toy's own min.  None (with a note) when
    dof <= 0 (all three), no window bin (p_sr, p_comb) or no valid toy."""
    n = len(q_toys.get("saturated", ()))
    out = {"gof_p_glob": None, "gof_p_sr": None, "gof_p_comb": None, "gof_p_convention": "(k+1)/(N+1)",
           "gof_aux": GOF_AUX, "gof_window": (window_info or {}).get("window"),
           "gof_n_sr_bins": int((window_info or {}).get("n_bins") or 0), "gof_p_note": None,
           "gof_p_missing": None}
    if dof is not None and dof <= 0:
        out["gof_p_note"], out["gof_p_missing"] = _gof_undefined(dof), "dof"
        return out
    if not n:
        out["gof_p_note"], out["gof_p_missing"] = "no valid toy fit", "no_toys"
        return out
    pg = _pooled_p(q_toys["saturated"], q_obs["saturated"])
    out["gof_p_glob"] = float(pg[0])
    if "window" not in q_obs or len(q_toys.get("window", ())) != n:
        lo, hi = (window_info or {}).get("window") or gof_window()
        out["gof_p_note"] = (f"no main bin centre in [{lo:g}, {hi:g}) GeV "
                             "(MODEL.json _meta.binning): no window p, no combined p")
        out["gof_p_missing"] = "no_window"
        return out
    ps = _pooled_p(q_toys["window"], q_obs["window"])
    out["gof_p_sr"] = float(ps[0])
    m = np.minimum(pg, ps)
    out["gof_p_comb"] = float(np.sum(m <= m[0] + 0.25 / len(m)) / len(m))
    out["gof_q_obs_sr"] = float(q_obs["window"])
    return out


def _pyhf_toy_pdf(model, pars, floor=GEN_RATE_FLOOR):
    """(pdf, n_floored): the toy pdf of the pyhf engine at ``pars``, pyhf's ``model.make_pdf(pars)`` with the main-bin
    Poisson rates below ``floor`` raised to it (GEN_RATE_FLOOR, as the twin's toy_pdf), the constraint terms pyhf's;
    n_floored the main bins raised.  Where none is (or ``floor`` is None: the shipped grader) it is pyhf's own pdf, so
    the draws are pyhf's draw for draw.  A rate that is not a number is left as it is."""
    tl = pyhf.tensorlib
    p = tl.astensor(np.asarray(pars, dtype=float))
    if floor is None:
        return model.make_pdf(p), 0
    rates = np.asarray(model.expected_actualdata(p), dtype=float)
    low = np.isfinite(rates) & (rates < floor)
    n = int(np.sum(low))
    if not n:
        return model.make_pdf(p), 0
    from pyhf import probability as prob
    parts = [prob.Independent(prob.Poisson(tl.astensor(np.where(low, float(floor), rates))))]
    cons = model.constraint_model.make_pdf(p)
    if cons:
        parts.append(cons)
    return prob.Simultaneous(parts, model.fullpdf_tv, model.batch_size), n


def _drawn_main_rates(model, pars, ev=None, floor=GEN_RATE_FLOOR):
    """The main-bin rates a toy at ``pars`` is drawn at: the twin's floored rates (``ev``, JaxPyhfModel.toy_pdf), else
    pyhf's with a finite rate below ``floor`` raised to it (_pyhf_toy_pdf)."""
    if ev is not None:
        return np.asarray(ev.expected_main(pars), dtype=float)
    rates = np.asarray(model.expected_actualdata(pyhf.tensorlib.astensor(np.asarray(pars, dtype=float))), dtype=float)
    if floor is None:
        return rates
    return np.where(np.isfinite(rates) & (rates < floor), float(floor), rates)


def toy_ensembles(model, data, res, mu_hat, ev=None, obs_fit=None, dof=None, window=None, window_info=None,
                  gen_floor=GEN_RATE_FLOOR, check_rates=True):
    """Toy ensemble at the best fit: GOF p's (fallback for the scored stage, same recipe as freq_coverage) and
    bias / spread of mu-hat.  pyhf: toys drawn at the best fit's rates floored at ``gen_floor`` (_pyhf_toy_pdf; the
    bins it raised in n_gen_floor; None: pyhf's own rates, the shipped grader), pyhf fits and deviance (negative-rate
    fits counted).  jax (``ev``, ``obs_fit``): one likelihood, combine's -- toys from the twin's floored rates at its
    best fit, twin refits, combine's saturated q (``gof_p_toys_deviance`` diagnostic).  Aux data per GOF_AUX.  Floor /
    kink fits are counted, not dropped.  gof_p_toys = gof_p_glob (older key).  ``check_rates`` (every task but a
    shipped-rule one): a generation point whose main or Poisson-constraint rates no Poisson draw takes raises
    ModelFitError before any draw (check_toy_rates: the model's failure)."""
    import time
    t0 = time.perf_counter()
    n_toys = N_TOYS_FIT if N_TOYS_FIT is not None else N_TOYS
    if dof is None:
        dof = _gof_dof(model, [res.bestfit, model.config.suggested_init()])
    if window is not None and window_info is None:
        window_info = {"window": list(gof_window()), "n_bins": int(np.sum(window))}
    np.random.seed(20260923)
    poi = model.config.poi_index
    n_main = model.config.nmaindata
    d = np.asarray(data, dtype=float)
    n_gen_floor = None
    if ev is not None:
        pars_gen = np.asarray(obs_fit["x"], dtype=float)
        mu_gen = float(pars_gen[poi])
        q_obs = _twin_gof(ev, pars_gen, d[:n_main], d[n_main:], window)
        pdf = ev.toy_pdf(pars_gen)
    else:
        pars_gen = np.asarray(res.bestfit, dtype=float)
        mu_gen = float(mu_hat)
        pars_obs = np.asarray(model_fit(pyhf.infer.mle.fit, data, model))
        q_obs = _pyhf_gof(model, pars_obs, d[:n_main], window)
        pdf, n_gen_floor = _pyhf_toy_pdf(model, pars_gen, gen_floor)
    if check_rates:
        check_toy_rates(model, pars_gen, _drawn_main_rates(model, pars_gen, ev, gen_floor))
    q_toys = {k: [] for k in q_obs}
    mus, flags = [], {"n_floor": 0, "n_sample_floor": 0, "n_kink": 0, "n_negative_rate": 0}
    for _ in range(n_toys):
        toy = np.asarray(pdf.sample((1,))[0], dtype=float)
        if GOF_AUX == "held":
            toy[n_main:] = d[n_main:]
        try:
            if ev is not None:
                r = ev.fit(toy[:n_main], toy[n_main:], init=pars_gen, want_sigma=False)
                if not r["valid"]:
                    continue
                fitted = r["x"]
                q = _twin_gof(ev, fitted, toy[:n_main], toy[n_main:], window)
                flags["n_floor"] += int(r["n_floor"] > 0)
                flags["n_sample_floor"] += int(r["n_sample_floor"] > 0)
                flags["n_kink"] += int(r["kink"])
            else:
                fitted = np.asarray(pyhf.infer.mle.fit(toy, model, init_pars=pars_gen.tolist()))
                exp = np.asarray(model.expected_actualdata(fitted), dtype=float)
                flags["n_negative_rate"] += int(exp.min() < 0)
                q = _pyhf_gof(model, fitted, toy[:n_main], window)
        except Exception:
            continue
        for k, v in q.items():
            q_toys[k].append(v)
        mus.append(float(fitted[poi]))
    mus = np.asarray(mus)
    unc = float(res.uncertainty[poi])
    ps = _gof_pvalues(q_obs, q_toys, dof, window_info)
    out = {"n_toys": int(len(mus)), "n_toys_requested": int(n_toys),
           "gof_p_toys": ps["gof_p_glob"], **ps,
           "gof_statistic": "saturated_combine" if ev is not None else "saturated_deviance",
           "gof_q_obs": float(q_obs["saturated"]),
           "gof_dof": int(dof)}
    if ev is not None:
        out["gof_p_toys_deviance"] = (float(_pooled_p(q_toys["saturated_deviance"],
                                                      q_obs["saturated_deviance"])[0])
                                      if dof > 0 and len(q_toys["saturated_deviance"]) else None)
        out["gof_q_obs_deviance"] = float(q_obs["saturated_deviance"])
    if dof <= 0:
        out["gof_p_undefined"] = _gof_undefined(dof)
    keep = ("n_floor", "n_sample_floor", "n_kink") if ev is not None else ("n_negative_rate",)
    if ev is None and gen_floor is not None:
        out["n_gen_floor"] = n_gen_floor
    return {**out,
            "bias_mu_injected": mu_gen,
            "bias_mean_mu_hat": float(mus.mean()) if len(mus) else None,
            "bias": float(mus.mean() - mu_gen) if len(mus) else None,
            "toy_spread": float(mus.std()) if len(mus) else None,
            "unc_coverage_ratio": float(mus.std() / unc)
            if len(mus) and unc > 0 else None,
            **{k: int(flags[k]) for k in keep},
            "wall_s": round(time.perf_counter() - t0, 2)}


def freq_coverage(model, data, mu_truth, seed=20260924, ev=None, obs_fit=None, dof=None, window=None,
                  window_info=None, extra_pins=None, n_toys=None, gen_floor=GEN_RATE_FLOOR, check_rates=True):
    """The SCORED GOF stage and coverage at truth.  Toys are generated at the truth-pinned conditional fit (POI =
    mu_truth, nuisances profiled on the observations; recorded in ``cond_fit``), main counts Poisson-drawn, aux
    per GOF_AUX (held: coverage is conditional on the aux measurements), each toy refit.  GOF p's from
    _gof_pvalues; ``gof_p_saturated`` = global p (older key), ``gof_p_saturated_deviance`` diagnostic.
    pyhf: toys drawn at the conditional fit's rates floored at ``gen_floor`` (_pyhf_toy_pdf; the bins it raised in
    cond_fit's n_gen_floor; None: pyhf's own rates, the shipped grader), pyhf fits (HESSE sigmas), deviance.  jax
    (``ev``, ``obs_fit``): everything on the twin (floored
    rates, autodiff-Hessian sigmas, combine's q); raises _TwinFitError if the conditional fit fails (caller
    reruns on pyhf).  ``extra_pins`` ({parameter index: value}, twin only): further parameters held at their values
    (clipped into their bounds) in the conditional fit the toys are drawn at, recorded as ``pins`` {name: value}.
    ``n_toys``: the number of toys (default N_TOYS).  ``check_rates`` (every task but a shipped-rule one): a
    generation point whose main or Poisson-constraint rates no Poisson draw takes raises ModelFitError before any draw
    (check_toy_rates: the model's failure, its GOF charged).

    Every valid toy enters the GOF and the coverage denominator.  A toy without a POI sigma (e.g. POI on its
    bound with zero curvature) is covered iff |mu-hat - truth| <= _bound_window; dropping it would bias
    coverage in a truth-dependent way.  Pulls use toys with a sigma (n_pull_toys); coverage_68 is None only
    when no toy has one."""
    import time
    t0 = time.perf_counter()
    n_toys = N_TOYS if n_toys is None else int(n_toys)
    np.random.seed(seed)
    poi = model.config.poi_index
    n_main = model.config.nmaindata
    lo, hi = model.config.suggested_bounds()[poi]
    win = _bound_window(lo, hi, mu_truth)
    mu_pin = float(np.clip(mu_truth, lo, hi))
    pins = {int(i): float(np.clip(v, *model.config.suggested_bounds()[i])) for i, v in (extra_pins or {}).items()}
    d = np.asarray(data, dtype=float)
    if ev is not None:
        x0 = ev.init.copy()
        x0[poi] = mu_pin
        fx = ev.fixed.copy()
        fx[poi] = True
        alt = np.asarray(obs_fit["x"], dtype=float).copy()
        alt[poi] = mu_pin
        for i, v in pins.items():
            x0[i] = alt[i] = v
            fx[i] = True
        rc = ev.fit(d[:n_main], d[n_main:], init=x0, fixed=fx, want_sigma=False, starts=(alt,))
        if not rc["valid"]:
            raise _TwinFitError("twin conditional (truth-pinned) fit invalid")
        pars_cond = rc["x"]
        cond = {"engine": "jax", "n_floor": int(rc["n_floor"]), "n_sample_floor": int(rc["n_sample_floor"]),
                "kink": bool(rc["kink"]), "min_expected_rate": float(rc["min_rate"]),
                "start": int(rc["start"])}
        obs = _twin_gof(ev, obs_fit["x"], d[:n_main], d[n_main:], window)
    elif pins:
        raise ValueError("extra_pins need the twin")
    else:
        pars_cond = np.asarray(model_fit(pyhf.infer.mle.fixed_poi_fit, mu_pin, data, model), dtype=float)
        nu_c = np.asarray(model.expected_actualdata(pyhf.tensorlib.astensor(pars_cond)), dtype=float)
        cond = {"engine": "pyhf", "min_expected_rate": float(nu_c.min()),
                "negative_expected_rate": bool(nu_c.min() < 0)}
        pars_hat = np.asarray(model_fit(pyhf.infer.mle.fit, data, model), dtype=float)
        obs = _pyhf_gof(model, pars_hat, d[:n_main], window)
    if dof is None:
        dof = _gof_dof(model, [pars_cond, model.config.suggested_init()])
    if window is not None and window_info is None:
        window_info = {"window": list(gof_window()), "n_bins": int(np.sum(window))}
    if ev is not None:
        pdf = ev.toy_pdf(pars_cond)
    else:
        pdf, n_gen_floor = _pyhf_toy_pdf(model, pars_cond, gen_floor)
        if gen_floor is not None:
            cond["n_gen_floor"] = n_gen_floor
    if check_rates:
        check_toy_rates(model, pars_cond, _drawn_main_rates(model, pars_cond, ev, gen_floor))
    toys = np.array(pdf.sample((n_toys,)), dtype=float)
    if GOF_AUX == "held":
        toys[:, n_main:] = d[n_main:]
    mu_hats, sigmas, covered, stats = [], [], [], {k: [] for k in obs}
    n_sig_undef, n_sig_undef_poi_bound, n_sig_undef_cov = 0, 0, 0
    flags = {"n_floor": 0, "n_sample_floor": 0, "n_kink": 0, "n_negative_rate": 0}
    undef_bound_dist = []
    for toy in toys:
        try:
            if ev is not None:
                r = ev.fit(np.asarray(toy)[:n_main], np.asarray(toy)[n_main:],
                           init=pars_cond)
                if not r["valid"]:
                    continue
                x_fit = r["x"]
                sigma_fit = None if r["sigma"] is None else float(r["sigma"][poi])
                gof = _twin_gof(ev, x_fit, np.asarray(toy)[:n_main], np.asarray(toy)[n_main:], window)
                flag = {"n_floor": r["n_floor"] > 0, "n_sample_floor": r["n_sample_floor"] > 0,
                        "n_kink": r["kink"]}
            else:
                fitted = pyhf.infer.mle.fit(toy, model, init_pars=pars_cond.tolist(),
                                            return_uncertainties=True)
                fitted = np.asarray(fitted, dtype=float)
                if fitted.ndim == 2:
                    x_fit, sigma_fit = fitted[:, 0], float(fitted[poi, 1])
                else:
                    x_fit, sigma_fit = fitted, None
                exp = np.asarray(model.expected_actualdata(x_fit), dtype=float)
                flag = {"n_negative_rate": exp.min() < 0}
                gof = _gof_stats(exp, np.asarray(toy)[:n_main], window)
        except Exception:
            continue
        for k, v in flag.items():
            flags[k] += int(v)
        mu_fit = float(x_fit[poi])
        if sigma_fit is None or not np.isfinite(sigma_fit) or sigma_fit <= 0:
            dist = min(abs(mu_fit - lo), abs(mu_fit - hi))
            n_sig_undef += 1
            n_sig_undef_poi_bound += int(dist <= win)
            undef_bound_dist.append(dist)
            cov = bool(abs(mu_fit - mu_truth) <= win)
            n_sig_undef_cov += int(cov)
            sigma_fit = float("nan")
        else:
            cov = bool(abs(mu_fit - mu_truth) <= sigma_fit)
        mu_hats.append(mu_fit)
        sigmas.append(sigma_fit)
        covered.append(cov)
        for k, v in gof.items():
            stats[k].append(v)
    mu_hats, sigmas = np.asarray(mu_hats), np.asarray(sigmas)
    okm = np.isfinite(sigmas)
    pulls = (mu_hats[okm] - mu_truth) / sigmas[okm]
    has_cov = bool(okm.any())
    out = {"mu_truth": float(mu_truth), "mu_pinned": mu_pin,
           **({"pins": {model.config.par_names[i]: v for i, v in pins.items()}} if pins else {}),
           "n_toys": int(len(mu_hats)), "n_failed": int(n_toys - len(mu_hats)),
           "n_coverage_toys": int(len(mu_hats)) if has_cov else 0,
           "n_pull_toys": int(okm.sum()),
           "coverage_68": float(np.mean(covered)) if has_cov else None,
           "pull_mean": float(pulls.mean()) if has_cov else None,
           "pull_width": float(pulls.std()) if has_cov else None,
           "at_boundary": bool(mu_pin != float(mu_truth) or abs(mu_pin - lo) < 1e-9),
           "n_sigma_undefined": n_sig_undef,
           "n_sigma_undefined_covered": n_sig_undef_cov,
           "n_sigma_undefined_poi_at_bound": n_sig_undef_poi_bound,
           "sigma_undefined_max_poi_bound_dist": float(max(undef_bound_dist)) if undef_bound_dist else None,
           "poi_bound_window": float(win),
           "cond_fit": cond,
           **{k: int(flags[k]) for k in (("n_floor", "n_sample_floor", "n_kink") if ev is not None
                                         else ("n_negative_rate",))},
           "gof_statistic": "saturated_combine" if ev is not None else "saturated_deviance",
           "gof_q_obs": float(obs["saturated"]),
           "gof_dof": int(dof)}
    ps = _gof_pvalues(obs, stats, dof, window_info)
    out.update(ps)
    out["gof_p_saturated"] = ps["gof_p_glob"]
    if "saturated_deviance" in stats:
        out["gof_p_saturated_deviance"] = (float(_pooled_p(stats["saturated_deviance"],
                                                           obs["saturated_deviance"])[0])
                                           if dof > 0 and len(stats["saturated_deviance"]) else None)
    if dof <= 0:
        out["gof_p_undefined"] = _gof_undefined(dof)
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    return out


def _verify_one(job):
    """Pool worker: one seed, isolated.  The backend is re-applied per seed: cabinetry restores its previous
    optimizer only after a successful fit, so results would otherwise depend on --jobs.
    ``cap`` (post.py verify_cap, or None): with a cap the seed runs under grader.deadlines and a wall budget
    (sbh/budget.py) whose end fails the dataset as the grader stage does ("fit failed", charged to the model,
    ran_out).  verify_wall_s is always recorded; the record is stamped (mark_errors)."""
    seed_dir, result_name, mu_truth, n_toys, optimizer, n_toys_fit, gof_aux, task_name, cap, window, prior = job
    global N_TOYS, N_TOYS_FIT, GOF_AUX, STAGE_DEADLINE, GRID_DEADLINE, GOF_WINDOW, TASK_NAME, PRIOR_CURVE
    N_TOYS, N_TOYS_FIT, GOF_AUX, GOF_WINDOW = n_toys, n_toys_fit, gof_aux, window
    TASK_NAME, PRIOR_CURVE = task_name, bool(prior)
    _set_backend(optimizer)
    if cap is None:
        return mark_errors(_verify_isolated(Path(seed_dir), result_name, mu_truth, task_name),
                           _shipped(Path(seed_dir)))
    import time
    t0 = time.monotonic()
    cap_s = cap.get("cap_s")
    if cap_s is None:
        rec = _verify_isolated(Path(seed_dir), result_name, mu_truth, task_name)
    else:
        _repo_on_path()
        from sbh.budget import BudgetExceeded, budget
        from sbh.kinds.binned_likelihood.grader import deadlines
        STAGE_DEADLINE, GRID_DEADLINE = deadlines(t0, cap_s)
        try:
            with budget(cap_s, f"the model verification of {Path(seed_dir).name}"):
                rec = _verify_isolated(Path(seed_dir), result_name, mu_truth, task_name)
        except BudgetExceeded:
            rec = {"seed": Path(seed_dir).name, "ran_out": True,
                   "status": f"fit failed: the model verification's cap of {cap_s:g} s per dataset ran out before "
                             f"this model was verified ({time.monotonic() - t0:.0f} s)"}
        finally:
            STAGE_DEADLINE = GRID_DEADLINE = None
    rec["verify_wall_s"] = round(time.monotonic() - t0, 2)
    return mark_errors(rec, _shipped(Path(seed_dir)))


def _verify_isolated(seed_dir, result_name, mu_truth, task_name):
    """The size caps (_size_gate) and the event-list gate, then verify_seed; an exception of the grader's own code is
    this seed's "verify error" only, recorded as the evaluator's ("evaluator_error")."""
    try:
        over = _size_gate(seed_dir)
        if over is not None:
            return over
        bad, ev_info = _event_list_gate(seed_dir, task_name)
        if bad is not None:
            return bad
        rec = verify_seed(seed_dir, result_name, mu_truth=mu_truth)
        if ev_info:
            rec["event_list"] = ev_info
        return rec
    except Exception as e:
        return {"seed": seed_dir.name, "status": f"verify error: {type(e).__name__}: {e}", "evaluator_error": _err(e)}


SCORED_BLOCKS = ("mu_fit", "mass_grid", "toys", "freq_toys", "z_exp_asimov", "quality_crosscheck")
REPORT_ONLY_BLOCKS = ("impacts", "gof_seq", "pyhf_crosscheck", "combine_model")


def evaluator_errors(rec) -> list:
    """The evaluator-side errors one grader record carries that leave a scored number out: the record's own ("verify
    error"), its SCORED_BLOCKS' and, on the mass grid's link, the profile at MODEL.json (link.model_nll: the lower-2NLL
    rule not applied) and a fit comparison that raised only where it left the grid unlinked (_grid_link then marks the
    mass_grid block itself; where the identical-workspace route linked the grid anyway nothing is left out), as
    "<where>: <message>"."""
    out = []
    if not isinstance(rec, dict):
        return out
    if rec.get("evaluator_error"):
        out.append(f"record: {rec['evaluator_error']}")
    for k in SCORED_BLOCKS:
        blk = rec.get(k)
        if isinstance(blk, dict) and blk.get("evaluator_error"):
            out.append(f"{k}: {blk['evaluator_error']}")
    link = (rec.get("mass_grid") or {}).get("link") if isinstance(rec.get("mass_grid"), dict) else None
    if isinstance(link, dict):
        if (link.get("evaluator_error") and link.get("linked") is not True
                and not rec["mass_grid"].get("evaluator_error")):
            out.append(f"mass_grid.link: {link['evaluator_error']}")
        mn = link.get("model_nll")
        if isinstance(mn, dict) and mn.get("evaluator_error"):
            out.append(f"mass_grid.link.model_nll: {mn['evaluator_error']}")
    return out


def fault_places(rec) -> list:
    """Where a grader record carries an evaluator error on what the score reads (evaluator_errors), by name: "record",
    a SCORED_BLOCKS name, "mass_grid.link" or "mass_grid.link.model_nll"."""
    return [e.split(":", 1)[0] for e in evaluator_errors(rec)]


def report_only_errors(rec) -> list:
    """The evaluator-side errors of one grader record's REPORT_ONLY_BLOCKS, as "<where>: <message>": recorded, never
    counted in grader_attempts (the score reads none of these blocks)."""
    if not isinstance(rec, dict):
        return []
    return [f"{k}: {rec[k]['evaluator_error']}" for k in REPORT_ONLY_BLOCKS
            if isinstance(rec.get(k), dict) and rec[k].get("evaluator_error")]


def attempts_of(rec) -> dict:
    """The grader attempts a stored record counts: its grader_attempts, else (a record written before them) one
    attempt, which ended with an evaluator error only where this grader marked one: a record stamped ERRORS_MARKED with
    an evaluator error on what the score reads (evaluator_errors).  A record without the stamp was graded before the
    grader marked its own errors, by an earlier grader that may since have been fixed: its error counts as none, so a
    fault in the dataset's first regrade is a first fault, retried before it can be reproduced."""
    att = rec.get("grader_attempts") if isinstance(rec, dict) else None
    if isinstance(att, dict):
        return att
    stamped = isinstance(rec, dict) and rec.get(ERRORS_MARKED_KEY) == ERRORS_MARKED
    return {"n": 1, "n_evaluator_error": int(stamped and bool(evaluator_errors(rec)))}


def _previous_attempts(path) -> dict:
    """{seed: its grader_attempts} as the verification file this verification replaces recorded them: ``path`` itself
    (a re-run over it), else ``path``.prev (where the post stage moves it first); attempts_of each record."""
    out = {}
    for p in (Path(path).with_name(Path(path).name + ".prev"), Path(path)):
        try:
            rows = json.loads(p.read_text()).get("per_seed") or []
        except (OSError, ValueError, AttributeError):
            continue
        for r in rows:
            if not isinstance(r, dict) or not r.get("seed"):
                continue
            out[r["seed"]] = attempts_of(r)
    return out


def to_retry(rec) -> bool:
    """Whether the post stage verifies a dataset once more before it is scored (--stage retry): its record carries an
    evaluator error on what the score reads (evaluator_errors) that has not recurred yet (grader_attempts
    n_evaluator_error below 2).  Its fault then either clears (a one-off) or is reproduced before the run is first
    scored, never shown with the faulted cases left out as if a rerun could clear them."""
    return bool(evaluator_errors(rec)) and int(attempts_of(rec).get("n_evaluator_error") or 0) < 2


def count_attempt(rec, before) -> dict:
    """``rec`` (one dataset's new grader record) with "grader_attempts" = {"n": verifications of this dataset so far,
    this one included, "n_evaluator_error": those that ended with an evaluator-side error (evaluator_errors), and, once
    there was one, "n_systemic": those whose error was part of a systemic fault (mark_systemic, never counted in
    n_evaluator_error)}, counted on from ``before`` (_previous_attempts): an evaluator error seen twice is reproduced,
    not a one-off."""
    prev = before.get(rec.get("seed")) or {}
    att = {"n": int(prev.get("n") or 0) + 1,
           "n_evaluator_error": int(prev.get("n_evaluator_error") or 0) + int(bool(evaluator_errors(rec)))}
    if prev.get("n_systemic"):
        att["n_systemic"] = int(prev["n_systemic"])
    rec["grader_attempts"] = att
    return rec


SYSTEMIC_FAULT = "systemic evaluator fault"


def systemic_places(rows, completed) -> dict:
    """{place: number of datasets} for each place (fault_places) where the records ``rows`` ({seed: record}; a dataset
    without one counts as no fault) of the run's completed datasets ``completed`` (seed names) carry an evaluator error
    on more than half of them: a systemic fault (SYSTEMIC_FAULT); {} where none is."""
    seeds = sorted(set(completed))
    counts = {}
    for s in seeds:
        for p in set(fault_places((rows or {}).get(s))):
            counts[p] = counts.get(p, 0) + 1
    return {p: n for p, n in sorted(counts.items()) if 2 * n > len(seeds)}


def is_systemic(rec, places) -> bool:
    """Whether ``rec``'s fault is part of a systemic one: it carries an evaluator error on what the score reads and
    every place it does (fault_places) lies where a systemic fault sits (``places``, systemic_places).  A dataset with
    a fault of its own beside the systemic one is not: that fault is retried and can be reproduced."""
    mine = set(fault_places(rec))
    return bool(mine) and mine <= set(places or ())


RETRIED_KEY = "retried"


def mark_systemic(rec, places) -> bool:
    """Count ``rec``'s latest attempts, whose evaluator error is part of a systemic fault (is_systemic over ``places``,
    systemic_places), as part of it: those attempts move from grader_attempts' "n_evaluator_error" to "n_systemic"
    (two where the record is its post stage's retry, RETRIED_KEY, both attempts having ended with the fault; else one),
    the places named in "systemic_fault".  A record already so marked, or whose fault is not systemic, is left as it
    is.  Returns whether it marked it."""
    if not is_systemic(rec, places) or rec.get("systemic_fault"):
        return False
    att = dict(attempts_of(rec))
    k = min(2 if rec.get(RETRIED_KEY) else 1, int(att.get("n_evaluator_error") or 0))
    att["n_evaluator_error"] = int(att.get("n_evaluator_error") or 0) - k
    att["n_systemic"] = int(att.get("n_systemic") or 0) + k
    rec["grader_attempts"], rec["systemic_fault"] = att, sorted(fault_places(rec))
    return True


def settle_systemic(rows, completed) -> list:
    """Mark every record of ``rows`` ({seed: record}) whose fault is systemic over those records as they now stand
    (systemic_places over the run's completed datasets ``completed``; mark_systemic), so the marks depend on the run's
    final records alone, never on the order its datasets were graded in.  Returns the seeds marked now."""
    places = systemic_places(rows, completed)
    return [s for s in sorted(set(completed)) if isinstance((rows or {}).get(s), dict)
            and mark_systemic(rows[s], places)]


GRADING_UNFINISHED = "verify error: grading did not finish"


def unfinished_record(seed, before, **extra) -> dict:
    """The stamped placeholder of one dataset's grading, written before it starts (GRADING_UNFINISHED): status and
    evaluator_error say grading did not finish, its attempt counted on from ``before`` (count_attempt), ``extra`` (a
    kind's own fields) beside them."""
    why = ("grading did not finish: the grader's process ended before it wrote this dataset's record (killed, or out of "
           "memory)")
    rec = {"seed": seed, "status": GRADING_UNFINISHED, "evaluator_error": why, **extra}
    return count_attempt(mark_errors(rec), before)


def _repo_on_path():
    """Put the repo on sys.path for sbh imports (a script run adds only its own directory)."""
    import sys
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.append(root)


def _verify_cap(run_dir, suite, task_name, seed_dirs):
    """Per-dataset cap for this verification (post.py verify_cap: the closure's, or None with why); None for a
    task without model_verify_cap."""
    if not task_name:
        return None
    _repo_on_path()
    from sbh.kinds.binned_likelihood.post import verify_cap
    from sbh.task import load_task
    return verify_cap(run_dir, suite, load_task(task_name), N_TOYS, ENGINE,
                      combine=[Path(s).name for s in seed_dirs if _graded_in_combine(Path(s))])


def _graded_in_combine(seed_dir) -> bool:
    """Whether verify_seed grades this dataset on combine (MODEL_COMBINE/ declared, battery on, combine installed)."""
    return (seed_dir / "out" / "MODEL_COMBINE").is_dir() and COMBINE_BATTERY and _combine_bindir() is not None


def _task_of_store_run(run_dir):
    """The task a stored run names (the run.json beside its evaluation directory), or None: a stored run is re-linked
    without its suite."""
    for p in (Path(run_dir) / "run.json", Path(run_dir).parent / "run.json"):
        try:
            t = json.loads(p.read_text()).get("task")
        except (OSError, ValueError, AttributeError):
            continue
        if t:
            from sbh.task import current_name
            return current_name(t)
    return None


def _task_of_run(run_dir):
    """The run's task from its suite's manifest (datasets live in <suite>/<public|sealed>/<seed>), or None."""
    for rp in sorted(Path(run_dir).glob("runs/seed_*/record.json")):
        try:
            ds = json.loads(rp.read_text()).get("dataset")
            if ds:
                from sbh.suite import load_manifest
                return load_manifest(Path(ds).parents[1]).get("task")
        except (OSError, ValueError, IndexError):
            continue
    return None


def _event_list_gate(seed_dir, task_name=None):
    """Event-list requirement on one dev run (eventlist.py, shared with the grader stage) -> (failure record or None,
    checks record or None).  Only for runs stamped with contract >= the task's event_list_from; unknown
    ``task_name`` = not checked.  A failure is charged to the model ("invalid workspace: EVENTS.json ...")."""
    try:
        rec = json.loads((seed_dir / "record.json").read_text())
    except (OSError, ValueError):
        return None, None
    if not (isinstance(rec, dict) and rec.get("contract")):
        return None, None
    _repo_on_path()
    from sbh.kinds.binned_likelihood.eventlist import STATUS_PREFIX, check_dir, required
    from sbh.task import load_task
    if task_name is None or not required(load_task(task_name), rec):
        return None, None
    if not (seed_dir / "out" / "MODEL.json").exists():
        return None, None
    try:
        problem, info = check_dir(seed_dir / "out", dataset_dir=rec.get("dataset"))
    except Exception as e:  # noqa: BLE001
        problem, info = f"cannot be checked ({type(e).__name__}: {str(e)[:160]})", {}
    if problem:
        return {"seed": seed_dir.name, "status": f"{STATUS_PREFIX}{problem}"[:600], "event_list": info}, info
    return None, info


def _set_backend(optimizer: str) -> None:
    """numpy + Minuit when per-toy uncertainties are needed (coverage stage), else numpy + scipy."""
    pyhf.set_backend("numpy", "minuit" if optimizer == "minuit" else pyhf.optimize.scipy_optimizer())


def _init_worker(optimizer: str, n_toys: int) -> None:
    """Pool initializer: re-apply main()'s state (forkserver workers inherit none; scipy's optimizer would drop
    every coverage toy).  Must not import jax: a raising initializer makes the Pool respawn forever."""
    global N_TOYS
    N_TOYS = n_toys
    _set_backend(optimizer)
    _pin_for_jax()


def _pin_for_jax(slot=None):
    """jax engine: pin each process to one CPU (the only way to keep XLA:CPU single-threaded; otherwise every
    worker sizes its pools to the whole allocation).  Must run before any jax import; failures are logged."""
    if ENGINE != "jax":
        return None
    try:
        from sbh.kinds.binned_likelihood.cpu_pin import pin_single_cpu
        return pin_single_cpu(slot)
    except Exception as e:  # noqa: BLE001
        logging.getLogger("h4l_model_verify").warning(
            "jax engine: CPU pinning failed (%s); XLA may oversubscribe the CPUs", _err(e))
        return None


def _worker_state(_=None):
    """(optimizer class, N_TOYS) as a worker sees them (initializer probe)."""
    return type(pyhf.get_backend()[1]).__name__, N_TOYS


def make_pool(workers: int, optimizer: str, n_toys: int):
    """Seed-parallel Pool with main()'s state re-applied per worker."""
    return Pool(workers, initializer=_init_worker, initargs=(optimizer, n_toys))


def main():
    global N_TOYS, N_TOYS_FIT, GOF_AUX, ENGINE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True,
                    help="a work/<suite>/<label> directory with runs/seed_*")
    ap.add_argument("--result-name", default="RESULT.json")
    ap.add_argument("--out", default="model_verify.json")
    ap.add_argument("--toys", type=int, default=0,
                    help="toys per seed of the truth-pinned (scored) GOF / coverage stage and, unless "
                         "--toys-fit says otherwise, of the stage at the fit (0 = skip)")
    ap.add_argument("--toys-fit", type=int, default=None,
                    help="toys of the stage at the fit (the scored stage's fallback; default: --toys)")
    ap.add_argument("--gof-window", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="the signal-region GOF window [LO, HI) (default: the run's task's gof_window)")
    ap.add_argument("--gof-aux", choices=("held", "redrawn"), default=GOF_AUX,
                    help="auxiliary data in the GOF toys: held at the observed values (the scored recipe, "
                         "default; env SBH_GOF_AUX) or re-drawn (combine --toysFrequentist)")
    ap.add_argument("--suite", default=None,
                    help="sealed suite dir: enables the truth-pinned "
                         "frequentist coverage stage (evaluator-side only)")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seed_* names to verify (default: every seed); the output then holds "
                         "only those -- merge per-seed outputs with --merge")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    help="seeds verified in parallel processes (default: the Slurm allocation)")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="merge per-seed outputs (from --seeds splits) into --out; no fitting")
    ap.add_argument("--stage", choices=("full", "grid", "link", "retry"), default="full",
                    help="grid: reload --out, recompute every ok dataset's mass_grid (the mH profile, the per-point "
                         "mu scans and template checks) from the stored MODEL_GRID where the task is scored on the "
                         "mass-marginal mu, and every combine-graded dataset's on any task (MODEL.json measured as a "
                         "fresh verification measures it where the record predates that, the grid by "
                         "COMBINE_GRID_RULE), keep the toys / GOF / coverage / mu_fit (adding MODEL.json's own mu scan "
                         "where the stored mu_fit predates it), stamp scan_version and write --out (the previous file "
                         "to --out.prev); engine jax unless SBH_TOY_ENGINE says otherwise.  link: reload --out and "
                         "re-link every stored mass_grid to MODEL.json with no fit (the grid's files, MODEL.json and "
                         "the reported mH as stored; the task's Task.grid_link_identical), the rest kept.  Both need "
                         "no suite (--task names the task where the run directory does not).  retry: reload --out and "
                         "verify once more, as the full stage does (same flags), every completed dataset whose record "
                         "carries an evaluator error on a scored block that has not recurred (to_retry), the attempts "
                         "counted on, unless the fault is systemic (the same place on more than half of the run's "
                         "completed datasets: nothing verified again, those records marked systemic_fault); the post "
                         "stage runs it before the score.  A task kept on the rule it shipped "
                         "with has neither the grid nor the retry stage")
    ap.add_argument("--task", default=None,
                    help="the run's task by name (default: the --suite manifest's, else the run.json beside the run "
                         "directory, else the datasets' suite)")
    ap.add_argument("--prior-curve", action="store_true",
                    help="also compute sigma0_ref per in-range grid point (the marginal mu rule's prior scale; the "
                         "closure's reference run)")
    a = ap.parse_args()
    if a.merge:
        rows, cap = {}, None
        for f in a.merge:
            part = json.load(open(f))
            for r in part.get("per_seed") or []:
                rows[r["seed"]] = r
            if cap is None and isinstance(part.get("verify_cap"), dict):
                cap = {k: v for k, v in part["verify_cap"].items() if k not in ("n_ran_out", "ran_out")}
        run_dir = Path(a.run_dir)
        summary = _write(run_dir / a.out, run_dir.name, [rows[k] for k in sorted(rows)], cap)
        print(f"merged {len(rows)} seeds from {len(a.merge)} files into {run_dir / a.out}")
        return
    if ENGINE is None:
        ap.error(f"SBH_TOY_ENGINE={_ENGINE_RAW!r}: expected pyhf or jax (alias evermore)")
    N_TOYS, N_TOYS_FIT, GOF_AUX = a.toys, a.toys_fit, a.gof_aux
    truths = {}
    if a.suite:
        from sbh.score import read_json_maybe_shielded
        man = read_json_maybe_shielded(Path(a.suite) / "manifest.json")
        truths = {s["seed"]: s["truth"] for s in man["seeds"]}
    optimizer = "minuit" if (a.suite and a.toys) else "scipy"
    _set_backend(optimizer)
    run_dir = Path(a.run_dir)
    from sbh.task import current_name
    task_name = (current_name(a.task) if a.task else current_name(man.get("task")) if a.suite
                 else _task_of_store_run(run_dir) or _task_of_run(run_dir))
    seed_dirs = sorted(run_dir.glob("runs/seed_*"))
    if a.seeds:
        want = {x.strip() for x in a.seeds.split(",") if x.strip()}
        seed_dirs = [s for s in seed_dirs if s.name in want]
    if a.stage == "link":
        _relink(run_dir, run_dir / a.out, a.result_name, seed_dirs, task_name)
        return
    if a.stage in ("grid", "retry") and task_name and _task_shipped(task_name):
        raise SystemExit(f"--stage {a.stage}: {task_name} keeps the rule it shipped with (Task.score_rule), whose "
                         "grader has no such stage: verify it whole")
    if a.stage == "grid":
        if not task_name:
            raise SystemExit("--stage grid: the run's task is unknown here; name it with --task")
        if not os.environ.get("SBH_TOY_ENGINE"):
            os.environ["SBH_TOY_ENGINE"] = ENGINE = "jax"
        _regrid(run_dir, run_dir / a.out, a.result_name, seed_dirs, task_name, a.prior_curve, a.jobs)
        return
    cap = _verify_cap(run_dir, a.suite, task_name, seed_dirs) if a.suite else None
    cap_blk = None if cap is None else {**cap, "toys": N_TOYS, "engine": ENGINE}
    exempt = set((cap or {}).get("combine_exempt") or ())

    def cap_of(seed_dir):
        """No cap for a combine-graded dataset (not the path the closure measured)."""
        if cap is None or seed_dir.name not in exempt:
            return cap
        return dict(cap, cap_s=None, why="no cap: graded through combine's battery, not the path the closure measured")
    window = tuple(a.gof_window) if a.gof_window else None
    if window is None and task_name:
        from sbh.task import load_task
        window = getattr(load_task(task_name), "gof_window", None)
    if window is None and seed_dirs:
        ap.error("no GOF window: give --suite (its task states gof_window) or --gof-window LO HI")
    jobs = [(str(s), a.result_name, (truths.get(s.name, {}) or {}).get("mu"), a.toys, optimizer, a.toys_fit,
             a.gof_aux, task_name, cap_of(s), window, a.prior_curve) for s in seed_dirs]
    if a.stage == "retry":
        _retry(run_dir, run_dir / a.out, jobs, optimizer, a.toys, a.jobs, cap_blk)
        return
    before = _previous_attempts(run_dir / a.out)
    rows = {} if task_name and _task_shipped(task_name) else {
        Path(j[0]).name: unfinished_record(Path(j[0]).name, before) for j in jobs}
    order = [Path(j[0]).name for j in jobs]

    def checkpoint():
        return _write(run_dir / a.out, run_dir.name, [rows[s] for s in order if s in rows], cap_blk)
    if rows:
        checkpoint()
    workers = max(1, min(a.jobs, len(jobs)))
    if workers > 1:
        with make_pool(workers, optimizer, a.toys) as pool:
            for rec in pool.imap(_verify_one, jobs):
                rows[rec["seed"]] = count_attempt(rec, before)
                checkpoint()
    else:
        _pin_for_jax()
        for job in jobs:
            rec = _verify_one(job)
            rows[rec["seed"]] = count_attempt(rec, before)
            checkpoint()
    summary = checkpoint()
    print(json.dumps({k: summary[k] for k in ("label", "n_with_model",
                                              "consistency_pass",
                                              "gof_p_values")}, indent=1))


def _task_shipped(task_name) -> bool:
    """Whether the named task keeps the rule it shipped with (Task.score_rule SHIPPED_RULE)."""
    return bool(_grid_knobs_of(task_name).get("shipped"))


def _grid_knobs_of(task_name) -> dict:
    """_grid_knobs of the named task (TASK_NAME set while it is read)."""
    global TASK_NAME
    keep = TASK_NAME
    TASK_NAME = task_name
    try:
        return _grid_knobs()
    finally:
        TASK_NAME = keep


def _completed(seed_dir) -> bool:
    """Whether the run completed this dataset (its record.json passes sbh.score.record_ok): the only datasets the score
    reads a grader record of."""
    try:
        rec = json.loads((Path(seed_dir) / "record.json").read_text())
        _repo_on_path()
        from sbh.score import record_ok
        return bool(record_ok(rec))
    except (OSError, ValueError, AttributeError, TypeError, KeyError):
        return False


def completed_seeds(run_dir) -> list:
    """The run's completed datasets (_completed), by seed name: the datasets a systemic fault is counted over."""
    return [s.name for s in sorted(Path(run_dir).glob("runs/seed_*")) if _completed(s)]


def seed_records(run_dir, name, new=None) -> dict:
    """{seed: its grader record} of a kind that writes one per dataset (runs/<seed>/<name>), as they stand on disk,
    ``new`` ({seed: record}) in place of what is there; an unreadable file is no record."""
    out = {}
    for f in sorted(Path(run_dir).glob(f"runs/seed_*/{name}")):
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(r, dict):
            out[f.parent.name] = r
    return {**out, **(new or {})}


def _retry(run_dir, path, jobs, optimizer, n_toys, workers, cap_blk):
    """--stage retry: reload ``path`` and verify once more (``jobs``: the full stage's, one per dataset) every completed
    dataset (_completed) whose record to_retry selects, counting its attempts on from the file; the new records replace
    theirs and the file is written in place (its .prev, the verification before this post stage, kept).  A dataset
    whose fault is systemic (is_systemic: every place it faults lies where a fault sits on more than half of the run's
    completed datasets, systemic_places over the whole file, whatever --seeds names) is not verified again: its record
    is marked (mark_systemic) and the file written.  The verification before it graded every dataset, so the decision
    is over the run's records as a whole, whatever order they were graded in."""
    if not Path(path).exists():
        raise SystemExit(f"--stage retry: no verification file to reload at {path}")
    prev = json.load(open(path))
    rows = {r["seed"]: r for r in prev.get("per_seed") or [] if isinstance(r, dict) and r.get("seed")}
    label = prev.get("label") or Path(run_dir).name
    done = completed_seeds(run_dir)
    systemic = systemic_places(rows, done)
    if systemic:
        marked = [s for s in done if isinstance(rows.get(s), dict) and mark_systemic(rows[s], systemic)]
        _write(path, label, [rows[k] for k in sorted(rows)], cap_blk)
        where = ", ".join(f"{p} on {n} of {len(done)}" for p, n in systemic.items())
        print(f"--stage retry: {SYSTEMIC_FAULT} ({where} completed datasets): those datasets not verified again "
              f"({len(marked)} record(s) marked now); the run stays provisional until the evaluator is fixed and the "
              "run verified again")
    todo = [j for j in jobs if to_retry(rows.get(Path(j[0]).name)) and _completed(j[0])
            and not (rows.get(Path(j[0]).name) or {}).get("systemic_fault")]
    if not todo:
        print(f"--stage retry: no completed dataset of {label} carries an evaluator error on a scored block that has "
              "not recurred: nothing to verify again")
        return
    print(f"--stage retry: verifying {len(todo)} dataset(s) once more ({', '.join(Path(j[0]).name for j in todo)})")
    before = _previous_attempts(path)
    n = max(1, min(workers, len(todo)))
    if n > 1:
        with make_pool(n, optimizer, n_toys) as pool:
            for rec in pool.imap(_verify_one, todo):
                rows[rec["seed"]] = count_attempt(rec, before)
    else:
        _pin_for_jax()
        for job in todo:
            rec = _verify_one(job)
            rows[rec["seed"]] = count_attempt(rec, before)
    summary = _write(path, label, [rows[k] for k in sorted(rows)], cap_blk)
    again = {Path(j[0]).name for j in todo}
    print(json.dumps({"verified again": sorted(again),
                      "cleared": sorted(s for s in again if not evaluator_errors(rows[s])),
                      "reproduced": sorted(s for s in (summary.get("evaluator_error_reproduced") or []) if s in again)},
                     indent=1))


def _regrid_one(job):
    """Pool worker of --stage grid: one stored record -> (the record, why its grid was kept as stored or None).  A
    MODEL.json-path record gets its mass_grid recomputed from MODEL_GRID (fits, scans, template checks, the link to
    MODEL.json), a stored mass_grid kept where the grid's files are not in the directory (a published run), and
    MODEL.json's own mu scan where its stored mu_fit predates it, everything else kept; a combine-graded one goes
    through _regrid_combine.  Never raises: a failure here is the evaluator's, on mass_grid."""
    seed_dir, result_name, rec, task_name, prior, optimizer, engine = job
    global TASK_NAME, PRIOR_CURVE, ENGINE
    TASK_NAME, PRIOR_CURVE, ENGINE = task_name, bool(prior), engine
    import time
    t0 = time.monotonic()
    _set_backend(optimizer)
    out_dir = Path(seed_dir) / "out"
    spec, kept = None, None
    try:
        spec = load_output_json(out_dir / "MODEL.json", max_bytes=MODEL_MAX_BYTES)
        spec.pop("_meta", None)
        reported_mh = _reported_mh_of(seed_dir, result_name)
        if rec.get("engine") == "combine":
            rec, kept = _regrid_combine(seed_dir, rec, reported_mh, spec)
        elif grid_bytes(out_dir) is None and isinstance(rec.get("mass_grid"), dict):
            kept = "no MODEL_GRID files in this evaluation directory: the stored mass_grid kept"
        else:
            rec.pop("mass_grid", None)
            _grid_stage(rec, out_dir, reported_mh, model_spec=spec)
    except Exception as e:  # noqa: BLE001
        rec["mass_grid"] = {"error": f"regrid: {_err(e)}", "evaluator_error": f"regrid: {_err(e)}",
                            "scan_version": SCAN_VERSION}
    mf = rec.get("mu_fit")
    if spec is not None and isinstance(mf, dict) and not mf.get("error") and not mf.get("mu_scan"):
        mf.update(_model_scan(spec))
    rec["regrid"] = {"scan_version": SCAN_VERSION, "wall_s": round(time.monotonic() - t0, 2), "engine": ENGINE,
                     **({"kept": kept} if kept else {})}
    return rec, kept


def _outputs_whole(seed_dir) -> bool:
    """Whether a dataset's out/ is as its run left it (RESULT.json beside MODEL.json), so that a grid not there was not
    declared.  A stored run that keeps MODEL.json alone (Kind.score_files) cannot tell."""
    return os.path.lexists(Path(seed_dir) / "out" / "RESULT.json")


def _regrid_combine(seed_dir, rec, reported_mh, spec):
    """--stage grid on a combine-graded record -> (the record, why its grid was kept as stored or None): verify_seed's
    _combine_record without the battery, whose fields stay as stored.  (1) MODEL.json measured where the record
    predates that (no mu_fit): its nominal fit, the twin's observed-data fit, mu_fit and pyhf_fit, as a fresh
    verification measures it; unfittable is the "fit failed" record a fresh verification gives, combine's block kept.
    (2) The grid by COMBINE_GRID_RULE (_combine_grid), a mass_grid of engine "combine" read as combine's own MH scan and
    any other as the declared grid's block, redone.  Kept as stored, unstamped: a dataset whose out/ holds no grid files
    and is not whole (_outputs_whole: a published run, so whether a grid was declared is unknown here), or holds none
    beside a stored declared grid's block."""
    out_dir = Path(seed_dir) / "out"
    if not isinstance(rec.get("mu_fit"), dict):
        ws = pyhf.Workspace(spec)
        model = ws.model()
        data = ws.data(model)
        refused = _fit_refusal(rec["seed"], model)
        if refused is not None:
            return {**refused, "combine_model": rec.get("combine_model")}, None
        res, attempts, nominal_fallback, nominal_twin, built = _nominal_fit(spec, model, data)
        if res is None:
            return {**_fit_failed(rec["seed"], attempts), "combine_model": rec.get("combine_model")}, None
        rec.update(**({"nominal_fit_fallback": nominal_fallback} if nominal_fallback else {}),
                   **({"nominal_fit_twin": nominal_twin} if nominal_twin else {}), fit_attempts=attempts)
        tn = getattr(res, "best_twice_nll", None)
        if tn is not None and np.isfinite(float(tn)):
            rec["pyhf_fit"] = {"twice_nll": float(tn)}
        ev, ev_reason, obs_fit, twin_fit = _twin_measurement(spec, model, data, res, built)
        if twin_fit is not None:
            rec["twin_fit"] = twin_fit
        rec["mu_fit"] = _mu_measurement(model, data, res, ev=ev, obs_fit=obs_fit)
        if ev_reason:
            rec["mu_fit"]["engine_fallback"] = ev_reason
        if twin_fit is not None:
            twin_fit["sigma_hess"] = rec["mu_fit"].get("sigma_hess")
        rec["pyhf_crosscheck"] = _pyhf_crosscheck(model, res, rec.get("mu_model"))
    mg = rec.get("mass_grid")
    scan = isinstance(mg, dict) and mg.get("engine") == "combine"
    if grid_bytes(out_dir) is not None:
        if isinstance(mg, dict) and not scan:
            rec.pop("mass_grid")
    elif not _outputs_whole(seed_dir):
        return rec, ("no MODEL_GRID files and no RESULT.json in this evaluation directory (a published run keeps "
                     "MODEL.json alone): whether a grid was declared is unknown here")
    elif isinstance(mg, dict) and not scan:
        return rec, "a declared grid's block but no MODEL_GRID files here"
    _combine_grid(rec, out_dir, reported_mh, spec)
    return rec, None


def _model_scan(spec):
    """MODEL.json's own conditional mu scan for a stored mu_fit that predates it (--stage grid; mu_fit_block
    records it on a fresh verification): {"mu_scan", "n_scan_failed", "scan_nll_min", "scan_version",
    "scan_source"}, the free fit and the scan on the twin when built (pyhf otherwise), the stored mu_fit's other keys
    (sigma, z_obs, ...) untouched.  On failure {"scan_error"}."""
    try:
        ws = pyhf.Workspace(spec)
        model = ws.model()
        data = ws.data(model)
        twin = _GridTwin() if ENGINE == "jax" else None
        got = twin.fit_point(spec, model, data, None) if twin is not None else None
        if got is None:
            got = _pyhf_point(model, data, None)
        sc = scan_point(model, data, got, ev=got.get("em"))
        return {"mu_scan": sc["mu_scan"], "n_scan_failed": sc["n_scan_failed"], "scan_nll_min": got["nll_min"],
                "scan_version": SCAN_VERSION, "scan_source": "regrid"}
    except Exception as e:  # noqa: BLE001
        return {"scan_error": _err(e)}


def _regrid(run_dir, path, result_name, seed_dirs, task_name, prior_curve, jobs):
    """--stage grid: reload ``path`` and recompute, among ``seed_dirs``, every ok record's mass_grid where the task is
    scored on the mass-marginal mu (Task.mu_marginal_over_mass: the per-point scans), and every combine-graded one's on
    any task (_regrid_combine: MODEL.json's own measurement where the record predates it, the grid by
    COMBINE_GRID_RULE); write the previous file to ``path``.prev and the new summary (scan_version stamped) to ``path``.
    A failed record, or one without MODEL.json here, is kept as stored.  Needs the run's task and no suite."""
    import shutil
    from sbh.task import load_task
    if not Path(path).exists():
        raise SystemExit(f"--stage grid: no verification file to reload at {path}")
    gated = bool(getattr(load_task(task_name), "mu_marginal_over_mass", False))
    prev = json.load(open(path))
    rows = {r["seed"]: r for r in prev.get("per_seed") or []}
    want = {Path(s).name for s in seed_dirs}
    cap = None
    if isinstance(prev.get("verify_cap"), dict):
        cap = {k: v for k, v in prev["verify_cap"].items() if k not in ("n_ran_out", "ran_out")}
    todo = []
    for s in seed_dirs:
        r = rows.get(Path(s).name)
        if r is None or r.get("status") != "ok" or not (gated or r.get("engine") == "combine"):
            continue
        if not (Path(s) / "out" / "MODEL.json").exists():
            continue
        todo.append((str(s), result_name, r, task_name, bool(prior_curve), "minuit", ENGINE))
    if not todo and not gated:
        raise SystemExit(f"--stage grid: the run's task ({task_name}) is not scored on the mass-marginal mu "
                         f"(Task.mu_marginal_over_mass) and no dataset was graded in combine: nothing to redo")
    skipped = sorted(want - {Path(j[0]).name for j in todo})
    print(f"--stage grid: {len(todo)} dataset(s) to redo from {path}"
          + ("" if gated else " (combine-graded only: the task is not scored on the mass-marginal mu)")
          + (f", {len(skipped)} kept as stored ({', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''})"
             if skipped else "") + (", with sigma0_ref (--prior-curve)" if prior_curve else ""))
    shutil.copy2(path, str(path) + ".prev")
    workers = max(1, min(jobs, len(todo)))
    label = prev.get("label") or Path(run_dir).name
    done, kept = 0, {}

    def take(out):
        nonlocal done
        rec, why = out
        rows[rec["seed"]] = rec
        done += 1
        if why:
            kept[rec["seed"]] = why
        _write(path, label, [rows[k] for k in sorted(rows)], cap)
    if workers > 1 and todo:
        with make_pool(workers, "minuit", 0) as pool:
            for out in pool.imap(_regrid_one, todo):
                take(out)
    else:
        _pin_for_jax()
        for job in todo:
            take(_regrid_one(job))
    summary = _write(path, label, [rows[k] for k in sorted(rows)], cap)
    walls = {r["seed"]: (r.get("regrid") or {}).get("wall_s") for r in summary["per_seed"] if r.get("regrid")}
    scans = {r["seed"]: (r.get("mass_grid") or {}).get("scan_s") for r in summary["per_seed"] if r.get("regrid")}
    print(json.dumps({"redone": done, "grid kept as stored": kept or None, "scan_version": SCAN_VERSION,
                      "wall_s": walls, "scan_s": scans, "mass_grid": summary.get("mass_grid")}, indent=1))


def _reported_mh_of(seed_dir, result_name="RESULT.json"):
    """The reported mH as the grader reads it: out/<result_name> (verify_seed's read), else the result the runner
    recorded in record.json (a published run keeps that, not RESULT.json); None when neither reports one."""
    rj = Path(seed_dir) / "out" / result_name
    if os.path.lexists(rj):
        return _read_result(rj)[1]
    try:
        return result_fields(json.loads((Path(seed_dir) / "record.json").read_text()).get("result"))[1]
    except (OSError, ValueError, AttributeError):
        return None


RELINKED = "relinked"


def relink_record(rec, out_dir, reported_mh, model_spec, identical=True) -> str:
    """--stage link on one stored grader record: its declared grid's mass_grid re-linked to MODEL.json with no fit.
    The link's products go first (mass_grid linked, link, usable, usable_reason, the link's evaluator_error, and the
    profile's MODEL.json 2NLL, restore_fitted_profile, the profile then rebuilt from its own fits), then _grid_link runs
    on the stored fits with the grid's files (load_grid on ``out_dir``; ``identical`` False: the fit route alone,
    Task.grid_link_identical) and MODEL.json (``model_spec``, without _meta); the report's usable flag follows.
    Returns RELINKED, or why the record is kept as stored (no fitted declared grid; no grid files here, which the
    identical-workspace route needs: a published run keeps MODEL.json only)."""
    mg = rec.get("mass_grid")
    if not isinstance(mg, dict) or not mg.get("curve"):
        return "kept: no fitted grid"
    if mg.get("engine") == "combine":
        return "kept: combine's own MH profile, not a declared grid"
    points = None
    if identical:
        points = _load_grid(out_dir)
        if not points:
            return "kept: no MODEL_GRID files in this evaluation directory (the identical-workspace route needs them)"
    for k in ("linked", "link", "usable", "usable_reason", "evaluator_error"):
        mg.pop(k, None)
    if restore_fitted_profile(mg):
        refresh_profile(mg, reported_mh)
    _grid_link(rec, reported_mh, points=points, model_spec=model_spec)
    if "usable" not in mg:
        refresh_profile(mg, reported_mh)
    return RELINKED


def _relink(run_dir, path, result_name, seed_dirs, task_name):
    """--stage link: reload ``path``, re-link every ok record's declared grid among ``seed_dirs`` to MODEL.json with no
    fit (relink_record), honouring the task's Task.grid_link_identical; write the previous file to ``path``.prev and
    the new one to ``path``, the file's engine and aux setting as stored.  Needs the run's task (--task, the run.json a
    published run carries, or its suite) and no suite otherwise."""
    import shutil
    from collections import Counter
    global ENGINE, GOF_AUX
    if not Path(path).exists():
        raise SystemExit(f"--stage link: no verification file to reload at {path}")
    if not task_name:
        raise SystemExit("--stage link: the run's task is unknown here; name it with --task (its "
                         "Task.grid_link_identical decides whether the identical-workspace route applies)")
    from sbh.task import load_task
    identical = bool(getattr(load_task(task_name), "grid_link_identical", True))
    prev = json.load(open(path))
    ENGINE = prev.get("engine") or ENGINE
    GOF_AUX = prev.get("gof_aux") or GOF_AUX
    rows = {r["seed"]: r for r in prev.get("per_seed") or []}
    cap = None
    if isinstance(prev.get("verify_cap"), dict):
        cap = {k: v for k, v in prev["verify_cap"].items() if k not in ("n_ran_out", "ran_out")}
    done, changed = Counter(), []
    for s in seed_dirs:
        r = rows.get(Path(s).name)
        if r is None or r.get("status") != "ok":
            continue
        before = ((r.get("mass_grid") or {}).get("link") or {}) if isinstance(r.get("mass_grid"), dict) else {}
        before = (before.get("linked"), before.get("by"))
        out = Path(s) / "out"
        try:
            spec = load_output_json(out / "MODEL.json", max_bytes=MODEL_MAX_BYTES)
            spec.pop("_meta", None)
        except Exception as e:  # noqa: BLE001
            what = f"kept: MODEL.json unreadable here ({type(e).__name__})"
        else:
            what = relink_record(r, out, _reported_mh_of(s, result_name), spec, identical=identical)
        done[what] += 1
        if what == RELINKED:
            link = r["mass_grid"].get("link") or {}
            r["relink"] = {"link_identical": identical, "linked_before": before[0], "by_before": before[1]}
            after = (link.get("linked"), link.get("by"))
            if after != before:
                changed.append(f"{r['seed']}: {before[0]}/{before[1]} -> {after[0]}/{after[1]}")
    shutil.copy2(path, str(path) + ".prev")
    label = prev.get("label") or Path(run_dir).name
    summary = _write(path, label, [rows[k] for k in sorted(rows)], cap)
    print(json.dumps({"task": task_name, "link_identical": identical, "records": dict(done),
                      "link changed (linked/by)": changed,
                      "model_nll used": sorted(r["seed"] for r in summary["per_seed"]
                                               if (((r.get("mass_grid") or {}).get("link") or {}).get("model_nll")
                                                   or {}).get("used"))}, indent=1))


def _write(path, label, results, cap=None):
    """Write the verification file.  ``cap`` (or None) is recorded as "verify_cap" with the datasets that ran out.  The
    reproduced and systemic faults are listed as these records stand, the systemic ones over the run's completed
    datasets (systemic_places)."""
    ok = [r for r in results if r["status"] == "ok"]
    systemic = systemic_places({r["seed"]: r for r in results if isinstance(r, dict) and r.get("seed")},
                               completed_seeds(Path(path).parent))
    summary = {"label": label, "n_seeds": len(results),
               "scan_version": SCAN_VERSION,
               "n_with_model": len(ok),
               "consistency_pass": sum(1 for r in ok if r["consistency"]
                                       and r["consistency"]["pass"]),
               "gof_p_values": [r["gof_p_saturated"] for r in ok],
               "coverage_68": {r["seed"]: r["freq_toys"].get("coverage_68")
                               for r in ok if isinstance(r.get("freq_toys"), dict)
                               and "error" not in r["freq_toys"]} or None,
               "z_exp_asimov": {r["seed"]: r["z_exp_asimov"] for r in ok
                                if isinstance(r.get("z_exp_asimov"), float)} or None,
               "engine": ENGINE,
               "engines": {e: sum(1 for r in ok if r.get("engine") == e)
                           for e in sorted({r.get("engine") or "?" for r in ok})} or None,
               "engine_fallback": {r["seed"]: r["engine_fallback"] for r in ok
                                   if r.get("engine_fallback")} or None,
               "negative_expected_rate": sorted(
                   r["seed"] for r in ok
                   if r.get("negative_expected_rate")
                   or (r.get("pyhf_crosscheck") or {}).get("negative_expected_rate")) or None,
               "mu_fit": {r["seed"]: {k: r["mu_fit"].get(k) for k in ("value", "sigma", "sigma_source", "z_obs")}
                          for r in ok if isinstance(r.get("mu_fit"), dict) and "error" not in r["mu_fit"]} or None,
               "gof_p_comb": {r["seed"]: (r.get("freq_toys") or {}).get("gof_p_comb") for r in ok
                              if isinstance(r.get("freq_toys"), dict)} or None,
               "gof_aux": GOF_AUX,
               "evaluator_error": {r["seed"]: evaluator_errors(r) for r in results if evaluator_errors(r)} or None,
               "evaluator_error_reproduced": sorted(
                   r["seed"] for r in results if evaluator_errors(r) and not is_systemic(r, systemic)
                   and int((r.get("grader_attempts") or {}).get("n_evaluator_error") or 0) >= 2) or None,
               "evaluator_error_systemic": sorted(r["seed"] for r in results if is_systemic(r, systemic)) or None,
               "evaluator_error_report_only": {r["seed"]: report_only_errors(r) for r in results
                                               if report_only_errors(r)} or None,
               "per_seed": results}
    mg = [r["mass_grid"] for r in ok
          if isinstance(r.get("mass_grid"), dict) and "error" not in r["mass_grid"]]
    summary["mass_grid"] = {
        "n_seeds_with_grid": len(mg),
        "n_agree_mh": sum(1 for g in mg if g.get("agree_mh")),
        "n_agree_sigma": sum(1 for g in mg if g.get("agree_sigma")),
        "n_scanned": sum(1 for g in mg if g.get("scan_version") == SCAN_VERSION),
        "n_coverage_ok": sum(1 for g in mg if (g.get("coverage") or {}).get("ok")),
        "n_content_usable": sum(1 for g in mg if (g.get("content") or {}).get("usable"))} if mg else None
    cm = [r["combine_model"] for r in ok if isinstance(r.get("combine_model"), dict)]
    fit = [c for c in cm if c.get("r_fit") is not None]
    summary["combine_model"] = {
        "n_seeds_declared": len(cm),
        "n_graded_on_combine": sum(1 for r in ok if r.get("engine") == "combine"),
        "n_skipped": sum(1 for c in cm if c.get("skipped")),
        "n_fit_ok": len(fit),
        "n_agree_mu": sum(1 for c in fit if c.get("agree_mu")),
        "n_mh_floating": sum(1 for c in fit if c.get("mh_floating")),
        "n_agree_mh": sum(1 for c in fit if c.get("agree_mh")),
        "n_data_mismatch": sum(1 for c in cm if c.get("data_mismatch"))} if cm else None
    if cap is not None:
        ran_out = [r["seed"] for r in results if r.get("ran_out")]
        summary["verify_cap"] = {**cap, "n_ran_out": len(ran_out), "ran_out": ran_out}
    tmp = Path(path).with_name(f".{Path(path).name}.tmp")
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=1)
    os.replace(tmp, path)
    return summary


if __name__ == "__main__":
    main()

