"""The scan_likelihood kind's grader -- EVALUATOR-ONLY: one dataset's declared model measured on its JAX twin.

Per completed dataset (post.py, the per-dataset post stage):
  0. the event list (binned_likelihood/eventlist.py), as the binned_likelihood grader checks it;
  1. the twin (``twin``): MODEL.json with `_meta` removed, its measurement's POI set to the coverage POI
     (coverage_poi) and every task POI's bounds to the task's range (Task.poi_bounds; a declared bound is read only
     for an open side), built by jax_engine with `_meta.poi_scaling`.  Every fit below runs on it: the kind fits no
     model with pyhf, which would read scaled samples as a plain product of normfactors;
  2. the global fit, the best of starts spread over the scanned POI's range (START_FRACS) and the other POIs at 1
     or their init;
  3. the scanned POI's profile over its range (profile_scan): a coarse scan denser near the lower end, holding the
     global fit, refined between the neighbours of every local minimum and in every q = 1 bracket, each point the
     lowest of three warm-started sweeps (upward, downward, outward from the global fit).  A point more than
     RESTART_DNLL below the global fit restarts it there: the free fit's minimum joins the scan, which is refined
     again.  The best fit is the scan's lowest point and q = 2 (nll - its nll), so the best fit, the piece holding it
     and its errors are one minimum's; every q <= 1 piece (scan_pieces);
  4. every POI's best fit, q = 1 interval (the scanned one's piece, the others' profile crossings, a side without a
     crossing ending at the range's end) and Hessian width, and its stat-only interval (nuisances fixed at the best
     fit, the other POIs profiled);
  5. the GOF and coverage at the truth (binned_likelihood verify.freq_coverage): the coverage POI pinned at its
     truth and every other POI with a truth too (extra_pins), toys from the twin, the GOF dof on the twin's rates,
     the GOF window the task's.  A truth-pinned fit that does not converge, or a generation point whose rates no
     Poisson draw takes (verify_fit.check_toy_rates: ModelFitError), is the model's: its GOF is failed ("gof_failed",
     charged as a failed dataset's);
  6. the self-reported quality cross-checks (verify_fit._crosschecks) and, report-only, whether RESULT.json's
     values are the model's ("consistency"; the scored comparison is scoring.py's truthfulness).

The record's "status" is "ok" or why the dataset has no measurement.  What the MODEL causes is its failure ("no
MODEL.json", "invalid workspace", "fit failed": a spec the twin does not read, a parameter whose init lies outside
its bounds, a constraint width that is not a positive finite number, rates the twin does not reproduce within the cross-check's tolerance on the gross per-bin rate, a global
fit that does not converge, a verification that runs out of the task's cap, post.py); only the evaluator's own failure
("verify error": no jax, an exception in its code; the record marked "evaluator_error") leaves the dataset pending.  Which it is follows from where the refusal
is raised (twin, _verify), never from an exception's text.  The self-reported claims are read through
verify_fit.finite_number: nothing a submission reports there raises.  Every record is stamped "errors_marked"
(binned_likelihood verify.mark_errors): its unmarked errors are the model's.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import VERIFY_FILE  # noqa: F401

SCAN_VERSION = "scan-v2"
SCAN_COARSE = 100
SCAN_REFINE = 8
START_FRACS = (0.005, 0.01, 0.03, 0.1, 0.3, 0.6)
RESTART_DNLL = 1e-3
SWEEP_TOL = 1e-3
PIECE_KEYS = ("lo", "hi", "x_min", "q_min", "lo_boundary", "hi_boundary")


class Refused(Exception):
    """The dataset cannot be graded; the message is its record's status.  ``evaluator`` True: the evaluator's own
    failure (its record carries "evaluator_error": pending, never charged), else the model's."""

    def __init__(self, status: str, evaluator: bool = False):
        super().__init__(status)
        self.evaluator = bool(evaluator)


def coverage_poi(task) -> str:
    """The POI whose truth-pinned coverage the grader measures (the measurement's POI): the task's second when it has
    one (the scanned first's coverage is its scan density's business), else the first."""
    return task.pois[1] if len(task.pois) > 1 else task.pois[0]


def build_model(spec: dict, task):
    """(pyhf model, data, the spec built, its `_meta`, {POI: parameter index}) of a MODEL.json spec: `_meta` removed,
    the measurement's POI coverage_poi(task), every task POI floating within the task's range (a declared bound only
    for an open side) and started at its declared init clipped into it."""
    import pyhf
    from . import poi_bounds
    spec = json.loads(json.dumps(spec))
    meta = spec.pop("_meta", None) or {}
    declared = pyhf.Workspace(spec).model().config
    cfg = spec["measurements"][0]["config"]
    cfg["poi"] = coverage_poi(task)
    pars = cfg.setdefault("parameters", [])
    for p in task.pois:
        i = declared.par_slice(p).start
        (dlo, dhi), init = declared.suggested_bounds()[i], declared.suggested_init()[i]
        lo, hi = poi_bounds(task, p)
        lo, hi = float(dlo if lo is None else lo), float(dhi if hi is None else hi)
        ent = next((x for x in pars if x.get("name") == p), None)
        if ent is None:
            ent = {"name": p}
            pars.append(ent)
        ent.pop("fixed", None)
        ent.update(bounds=[[lo, hi]], inits=[min(max(float(init), lo), hi)])
    ws = pyhf.Workspace(spec)
    model = ws.model()
    return model, np.asarray(ws.data(model), float), spec, meta, {p: model.config.par_slice(p).start for p in task.pois}


def twin(spec: dict, model, table: dict):
    """The model's JAX twin with its POI scaling, the kind's one engine (module docstring).  Refused, the model's: a
    parameter whose init lies outside its bounds ("fit failed", jax_engine.init_outside_bounds: pyhf's fit refuses
    such a model, so the binned_likelihood grader fails it alike), a constraint width that is not a positive finite
    number ("fit failed", binned_likelihood model.constraint_width_problem, as the binned grader refuses it); "invalid
    workspace": a modifier the engine does not
    support, a POI scaling outside the grammar, a spec its constructor refuses (jax_engine.SpecError), rates that miss
    pyhf's beyond the cross-check's tolerance on the gross per-bin rate.  Refused, the evaluator's ("verify error"): no
    jax.  Any other exception is the evaluator's too (verify_dataset)."""
    from ..binned_likelihood.model import poi_scaling_entries
    from ..binned_likelihood.verify_fit import _evt
    try:
        eng = _evt()
    except Exception as e:  # noqa: BLE001
        raise Refused(f"verify error: jax engine unavailable: {type(e).__name__}: {e}", evaluator=True)
    why = eng.init_outside_bounds(model.config)
    if why:
        raise Refused(f"fit failed: {why} (a fit cannot start there)")
    why = eng.constraint_width_problem(model.config)
    if why:
        raise Refused(f"fit failed: {why} (pyhf's fit fails on it, so no fit is made)")
    kinds = {m["type"] for c in spec["channels"] for s in c["samples"] for m in s["modifiers"]}
    if kinds - eng.SUPPORTED:
        raise Refused(f"invalid workspace: unsupported modifiers: {sorted(kinds - eng.SUPPORTED)}")
    try:
        poi_scaling_entries(table, spec)
    except ValueError as e:
        raise Refused(f"invalid workspace: {e}")
    try:
        ev = eng.JaxPyhfModel(spec, model, floor=eng.FLOOR, poi_scaling=table)
    except eng.SpecError as e:
        raise Refused(f"invalid workspace: {e}")
    why = eng.pyhf_mismatch(model, ev, gross=True)
    if why:
        raise Refused(f"invalid workspace: the evaluator's engine does not reproduce the model: {why}")
    return ev


def global_fit(ev, main, aux, idx: Dict[str, int], scanned: str) -> Optional[dict]:
    """The lowest valid fit from the model's init and starts with the scanned POI at START_FRACS of its range and the
    other POIs at 1 or their init (clipped); None when none converges."""
    i_s = idx[scanned]
    lo, hi = ev.bounds[i_s]
    cands = [ev.init.copy()]
    for f in START_FRACS:
        for other in (1.0, None):
            x = ev.init.copy()
            x[i_s] = lo + f * (hi - lo)
            for p, i in idx.items():
                if p != scanned and other is not None:
                    x[i] = min(max(other, ev.bounds[i][0]), ev.bounds[i][1])
            if not any(np.array_equal(x, c) for c in cands):
                cands.append(x)
    best = None
    for c in cands:
        r = ev.fit(main, aux, init=c, want_sigma=False)
        if r["valid"] and (best is None or r["nll"] < best["nll"]):
            best = r
    return best


def closed(piece: dict, xs: List[float]) -> dict:
    """A piece of the sorted scan points ``xs`` with both ends: an end with neither a q = 1 crossing nor the range's
    bound -- the q <= 1 run reaches a scan end off the bound, where the fit at the bound failed -- is that scan
    point, since the piece extends at least to it."""
    return {**piece, "lo": xs[0] if piece["lo"] is None else piece["lo"],
            "hi": xs[-1] if piece["hi"] is None else piece["hi"]}


def scan_pieces(xs: List[float], qs: List[float], lo: float, hi: float, x_hat: float) -> Tuple[List[dict], dict]:
    """(every q <= 1 piece of a scan, the one holding its lowest point ``x_hat``) in the record's form (PIECE_KEYS,
    ends closed)."""
    from . import scan as SC
    pieces = [closed({k: p[k] for k in PIECE_KEYS}, xs) for p in SC.pieces_of(xs, qs, lo, hi)]
    return pieces, next(p for p in pieces if p["lo"] <= x_hat <= p["hi"])


def profile_scan(ev, main, aux, fit: dict, i_s: int) -> dict:
    """The scanned POI's profile over its range (module docstring, step 3) -> {"points": [[x, q], ...], "x_hat" (the
    lowest point), "nll_min" (its nll), "pieces", "interval" (the piece holding x_hat), "n_points", "n_failed",
    "n_sweep_disagree", "n_restarts", "x_best" (the parameters at x_hat), "scan_version"}."""
    lo, hi = ev.bounds[i_s]
    fx = ev.fixed.copy()
    fx[i_s] = True
    x0 = np.asarray(fit["x"], float)
    vals: Dict[float, list] = {}
    xs_at: Dict[float, np.ndarray] = {}
    tried = set()

    def record(v, r):
        if v not in vals or r["nll"] < min(vals[v]):
            xs_at[v] = np.asarray(r["x"], float)
        vals.setdefault(v, []).append(float(r["nll"]))
        tried.add(v)

    def fit_at(v, init, alts=()):
        """The conditional fit at v from ``init`` (``alts`` tried where it does not converge), recorded."""
        tried.add(v)
        x_init = np.asarray(init, float).copy()
        x_init[i_s] = v
        r = ev.fit(main, aux, init=x_init, fixed=fx, want_sigma=False, starts=tuple(alts))
        if not (r["valid"] and np.isfinite(r["nll"])):
            return None
        record(v, r)
        return r

    def sweep(order):
        prev = x0
        for v in order:
            r = fit_at(v, prev, (x0,))
            if r is not None:
                prev = r["x"]

    def lowest():
        return min(vals, key=lambda v: (min(vals[v]), v))

    def refine():
        """Points between the neighbours of every local minimum and in every q = 1 bracket of the scan so far, each
        fitted from its neighbours' fits (the lower is kept)."""
        xs = sorted(vals)
        nll_min = min(vals[lowest()])
        qs = [2.0 * (min(vals[v]) - nll_min) for v in xs]
        extra = set()
        for k in range(len(xs)):
            a, b = max(k - 1, 0), min(k + 1, len(xs) - 1)
            if qs[k] <= qs[a] and qs[k] <= qs[b]:
                extra |= {float(v) for v in np.linspace(xs[a], xs[b], SCAN_REFINE + 2)[1:-1]}
            if k + 1 < len(xs) and (qs[k] - 1.0) * (qs[k + 1] - 1.0) < 0:
                extra |= {float(v) for v in np.linspace(xs[k], xs[k + 1], SCAN_REFINE // 2 + 2)[1:-1]}
        for v in sorted(extra - tried):
            for u in (max((u for u in xs if u < v), default=None), min((u for u in xs if u > v), default=None)):
                if u is not None:
                    fit_at(v, xs_at[u])

    x_fit = float(x0[i_s])
    record(x_fit, fit)
    coarse = sorted({lo + (hi - lo) * (k / SCAN_COARSE) ** 2 for k in range(SCAN_COARSE + 1)} | {x_fit})
    sweep(coarse)
    sweep(coarse[::-1])
    sweep([v for v in coarse if v > x_fit])
    sweep([v for v in coarse if v <= x_fit][::-1])
    refine()
    n_restarts = 0
    v_low = lowest()
    if min(vals[v_low]) < float(fit["nll"]) - RESTART_DNLL:
        n_restarts = 1
        r = ev.fit(main, aux, init=xs_at[v_low], want_sigma=False)
        if r["valid"] and r["nll"] <= min(vals[v_low]):
            record(float(r["x"][i_s]), r)
        refine()
    x_hat = lowest()
    nll_min = min(vals[x_hat])
    xs = sorted(vals)
    qs = [2.0 * (min(vals[v]) - nll_min) for v in xs]
    pieces, main_piece = scan_pieces(xs, qs, lo, hi, x_hat)
    return {"points": [[x, q] for x, q in zip(xs, qs)], "x_hat": x_hat, "nll_min": nll_min, "pieces": pieces,
            "interval": main_piece, "n_points": len(xs), "n_failed": len(tried) - len(xs),
            "n_sweep_disagree": sum(max(n) - min(n) > SWEEP_TOL for n in vals.values()),
            "n_restarts": n_restarts, "x_best": xs_at[x_hat], "scan_version": SCAN_VERSION}


def _stat_cond(ev, main, aux, x_hat, i_p: int, pois: List[int]):
    """cond(v) -> nll at POI i_p = v with every nuisance fixed at the best fit x_hat and the other POIs profiled."""
    fx = np.ones(ev.npars, bool)
    fx[[i for i in pois if i != i_p]] = False

    def cond(v):
        x_init = np.asarray(x_hat, float).copy()
        x_init[i_p] = v
        r = ev.fit(main, aux, init=x_init, fixed=fx, want_sigma=False)
        return float(r["nll"]) if r["valid"] else None
    return cond


def poi_fit(ev, model, data, x_hat, nll_min: float, i_p: int, pois: List[int], scan: Optional[dict]) -> dict:
    """One POI's best fit, q = 1 interval (``scan``: the scanned POI's piece holding the best fit; else profile
    crossings, a side reaching the range's end without one ending there), Hessian width and stat-only interval.  A
    side whose profile gives no result is None: the scorer fails a POI without both err_minus and err_plus."""
    from ..binned_likelihood.verify_fit import _conditional_fitters, _profile_side
    main, aux = data[:ev.n_main], data[ev.n_main:]
    v = float(x_hat[i_p])
    lo_b, hi_b = ev.bounds[i_p]
    sg = ev._sigma(x_hat, main, aux)
    s_h = float(sg[i_p]) if sg is not None and np.isfinite(sg[i_p]) and sg[i_p] > 0 else None
    out = {"value": v, "sigma_hess": s_h}
    s0 = s_h or 0.05 * (hi_b - lo_b)
    if scan is not None:
        out.update(err_minus=v - scan["interval"]["lo"], err_plus=scan["interval"]["hi"] - v)
    else:
        cond, reset, _state = _conditional_fitters(model, data, ev, x_hat, poi=i_p)
        for side, sgn, bound in (("lo", -1.0, lo_b), ("hi", +1.0, hi_b)):
            reset()
            d, st, _n = _profile_side(cond, nll_min, v, bound, sgn, s0)
            out[f"err_{'minus' if side == 'lo' else 'plus'}"] = abs(bound - v) if st == "bound" else d
            out[f"status_{side}"] = st
    cond = _stat_cond(ev, main, aux, x_hat, i_p, pois)
    for side, sgn, bound in (("minus", -1.0, lo_b), ("plus", +1.0, hi_b)):
        d, st, _n = _profile_side(cond, nll_min, v, bound, sgn, s0)
        out[f"stat_{side}"] = abs(bound - v) if st == "bound" else d
    return out


def _agrees(reported, fit: dict) -> bool:
    """Whether a reported POI entry's value is the model's: within 0.25 x the model's error on the value's side +
    0.05 of its best fit.  False where the model's fit is no interval (a profile side without a result: the scorer
    fails that POI)."""
    from . import scan as SC
    a, m = SC.interval(reported), SC.interval(fit)
    if a is None or m is None:
        return False
    return abs(a[0] - m[0]) <= 0.25 * SC.side_sigma(*m, a[0]) + 0.05


def verify_dataset(seed_dir: Path, task, truth: dict, n_toys: int) -> dict:
    """The grader on one completed dataset (module docstring) -> its record ("status" "ok" or why not), stamped
    (mark_errors).  Both of the evaluator's own exits (a Refused of the evaluator's: no jax; any other exception of its
    code) mark the record "evaluator_error" with the error text: the dataset is pending on an evaluator fault, never
    failed."""
    from ..binned_likelihood.verify import mark_errors
    from ..binned_likelihood.verify_fit import _err
    try:
        return mark_errors(_verify(Path(seed_dir), task, truth, n_toys))
    except Refused as e:
        rec = {"seed": Path(seed_dir).name, "status": str(e)[:600], "scan_version": SCAN_VERSION}
        if e.evaluator:
            rec["evaluator_error"] = str(e)[:600]
        return mark_errors(rec)
    except Exception as e:  # noqa: BLE001
        return mark_errors({"seed": Path(seed_dir).name, "status": f"verify error: {type(e).__name__}: {e}"[:600],
                            "evaluator_error": _err(e)[:600], "scan_version": SCAN_VERSION})


def _verify(seed_dir: Path, task, truth: dict, n_toys: int) -> dict:
    from ...contract import load_output_json
    from ..binned_likelihood import verify as MV
    from ..binned_likelihood.grader import MAX_JSON_BYTES
    from ..binned_likelihood.verify_fit import _crosschecks, _read_result
    rec = {"seed": seed_dir.name, "scan_version": SCAN_VERSION, "engine": "jax"}
    bad, ev_info = MV._event_list_gate(seed_dir, task.name)
    if bad is not None:
        return {**bad, "scan_version": SCAN_VERSION}
    if ev_info:
        rec["event_list"] = ev_info
    mp = seed_dir / "out" / "MODEL.json"
    if not mp.exists():
        raise Refused("no MODEL.json")
    if mp.stat().st_size > MAX_JSON_BYTES:
        raise Refused(f"invalid workspace: MODEL.json larger than {MAX_JSON_BYTES >> 20} MiB")
    try:
        declared = load_output_json(mp)
        model, data, spec, meta, idx = build_model(declared, task)
    except Exception as e:  # noqa: BLE001
        raise Refused(f"invalid workspace: {type(e).__name__}: {e}")
    ev = twin(spec, model, meta.get("poi_scaling") or {})
    main, aux = data[:ev.n_main], data[ev.n_main:]
    scanned = task.pois[0]
    fit = global_fit(ev, main, aux, idx, scanned)
    if fit is None:
        raise Refused("fit failed: the model's global fit does not converge on the evaluator's engine")
    scan = profile_scan(ev, main, aux, fit, idx[scanned])
    x_hat = scan.pop("x_best")
    rec["scan"] = scan
    pois = list(idx.values())
    rec["poi_fits"] = {p: poi_fit(ev, model, data, x_hat, scan["nll_min"], i, pois, scan if p == scanned else None)
                       for p, i in idx.items()}
    mask, winfo = MV._window_mask(model, meta, tuple(task.gof_window))
    dof = MV._gof_dof(model, [x_hat, model.config.suggested_init()], rates=ev.expected_main_unfloored)
    rec["gof_dof"] = int(dof)
    cp = coverage_poi(task)
    if n_toys and cp in truth:
        pins = {idx[p]: float(truth[p]) for p in task.pois if p != cp and p in truth}
        try:
            rec["freq_toys"] = MV.freq_coverage(model, data, float(truth[cp]), ev=ev,
                                                obs_fit={"x": x_hat, "nll": scan["nll_min"]}, dof=dof, window=mask,
                                                window_info=winfo, extra_pins=pins, n_toys=n_toys)
        except MV._TwinFitError as e:
            rec["freq_toys"], rec["gof_failed"] = {"error": f"{type(e).__name__}: {e}"}, True
        except MV.ModelFitError as e:
            rec["freq_toys"], rec["gof_failed"] = {"error": f"fit failed: {e}"}, True
    _, _, self_quality, _ = _read_result(seed_dir / "out" / "RESULT.json")
    _crosschecks(rec, self_quality, None)
    result = json.loads((seed_dir / "record.json").read_text()).get("result") or {}
    reported = {p.get("name"): p for p in result.get("pois") or [] if isinstance(p, dict)}
    rec["consistency"] = {"pass": all(_agrees(reported.get(p), f) for p, f in rec["poi_fits"].items())}
    rec["status"] = "ok"
    return rec
