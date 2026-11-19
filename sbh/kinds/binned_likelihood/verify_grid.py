"""The model verifier's mass-grid helpers (verify.py; the grid stage itself, mass_grid_check, stays there with
GRID_DEADLINE): signatures, the link to MODEL.json, the twin grid fitter and the 2 DeltalnL(mH) interval; loading
MODEL_GRID and the same-workspace test are model.py's (load_grid, grid_identity), shipped with the participant's
self-check.  Evaluator-only (sbh/harbor_pack.py EVALUATOR_ONLY); verify.py re-exports every name."""
import json
import math
import sys

import numpy as np
import pyhf

from .model import grid_identity, load_grid, nearest_point, read_point
from .verify_fit import (_TwinFitError, _err, _evt, MU_SUPPORT, SCAN_VERSION, SCAN_DEDUP, _conditional_fitters,
                         conditional_scan, mu_scan_set)


def _const(name, default):
    try:
        from . import scoring
    except Exception:  # noqa: BLE001
        return default
    return getattr(scoring, name, default)


REACH_TOL = _const("REACH_TOL", 0.5)
MAX_GAP_IN_RANGE = _const("MAX_GAP_IN_RANGE", 1.0)
YIELD_BAND_NEAR = _const("YIELD_BAND_NEAR", math.log(1.2))
YIELD_BAND_FAR = _const("YIELD_BAND_FAR", math.log(1.5))
YIELD_BAND_NEAR_GEV = _const("YIELD_BAND_NEAR_GEV", 5.0)
CENTROID_CONST_TOL = _const("CENTROID_CONST_TOL", 1.0)
CENTROID_SLOPE = _const("CENTROID_SLOPE", (0.8, 1.2))
CENTROID_MIN_SPAN = _const("CENTROID_MIN_SPAN", 4.0)
SLOPE_EXTRA = _const("SLOPE_EXTRA", 0.1)
DEAD_YIELD_FRAC = _const("DEAD_YIELD_FRAC", 0.05)
DEAD_MAX_FRAC = _const("DEAD_MAX_FRAC", 0.10)
WIDTH_TOL = _const("WIDTH_TOL", (0.08, 0.04))
WIDTH_TOL_LOCAL = _const("WIDTH_TOL_LOCAL", (0.04, 0.06))
WIDTH_LOCAL_GEV = _const("WIDTH_LOCAL_GEV", 1.0)
WIDTH_SCALING = _const("WIDTH_SCALING", (0.0, 0.25, 0.5, 0.75, 1.0))
SHARE_TOL = _const("SHARE_TOL", 0.15)
DEAD_BOUND_REL = 1e-6
CENTROID_HALF_WINDOW = 4.0
CENTROID_MIN_HALF = 1.5
PEAK_SMOOTH = 0.5
FINE_STEP = 0.05
FINE_MARGIN = 30.0
WINDOW_FRAC_FLOOR = 0.2
WINDOW_CORE = 10.0
LN_FLOAT_MAX = math.log(sys.float_info.max)


def _ln_ratio(a, b):
    """ln(a / b), or None ("no ratio") where a / b is not a positive finite float: either number not a number, zero,
    negative or infinite, or the ratio under- or overflowing the float range.  Never raises; where the ratio is a
    positive finite float the value is exactly math.log(a / b)."""
    try:
        r = float(a) / float(b)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return math.log(r) if math.isfinite(r) and r > 0 else None


LINK_MU_REL, LINK_MU_ABS = 0.05, 0.01
LINK_TWICE_NLL_TOL = _const("LINK_TWICE_NLL_TOL", 0.1)


def _model_twice_nll(rec):
    """{engine: MODEL.json's own 2NLL at its best fit} of a verification record: "jax" the twin's observed-data fit,
    "pyhf" the nominal fit's (cabinetry or pyhf's MLE); None where that engine did not fit it."""
    return {"jax": (rec.get("twin_fit") or {}).get("twice_nll"), "pyhf": (rec.get("pyhf_fit") or {}).get("twice_nll")}


def _grid_link(rec, reported_mh, points=None, model_spec=None):
    """Whether the grid is MODEL.json's model at the reported mH, by the first route that holds:

      1. identical workspace (``points`` {mass: path-or-spec} as _load_grid gives them and ``model_spec`` MODEL.json
         both given; None skips the route, Task.grid_link_identical False): the grid's workspace nearest the reported
         mH within the local spacing, at any mass the grid declares, in or out of the mass range, is MODEL.json's
         (model.grid_identity);
      2. fit: the fitted grid point nearest the reported mH (within the local spacing, model.nearest_point) reproduces
         MODEL.json's best fit (mu and 2*NLL within the LINK_* tolerances, 2*NLL on the engine that fitted that point,
         else cross-engine).  A comparison that cannot be made is not a pass.

    mass_grid["link"]: "linked", "by" ("identical workspace" or "fit") when linked, else "why"; the fit comparison's
    fields (point, distance, local_spacing, mu_grid, mu_model, mu_tol, twice_nll_grid, twice_nll_model, engine) as
    far as it got; with route 1 tried also its fields (identical, identical_point, identical_file, identical_distance,
    identical_spacing, identical_rel, identical_diff or identical_why) and "fit_linked", route 2's own verdict.
    Unlinked sets mass_grid["linked"] = False and the grader's usable flag False (scoring.grid_profile reads `linked`
    itself).  An exception in the fit route's own code is the evaluator's: link["evaluator_error"], and, when no route
    links the grid, mass_grid["evaluator_error"] too (the verdict text kept).  Linked by route 1, the profile takes
    MODEL.json's 2NLL at that point where it is lower (_profile_at_model)."""
    mg, mf = rec.get("mass_grid"), rec.get("mu_fit")
    if not isinstance(mg, dict) or not mg.get("curve"):
        return
    link = {"linked": None}
    mh_rep = reported_mh.get("value") if isinstance(reported_mh, dict) else None
    fit_ok, why, grader_err = False, None, None
    try:
        mhs = [float(c[0]) for c in mg["curve"]]
        if mh_rep is None or not np.isfinite(float(mh_rep)):
            why = "no reported mH to locate MODEL.json on the grid"
        else:
            j, dist, spacing, near = nearest_point(mhs, float(mh_rep))
            link.update(point=mhs[j], distance=dist, local_spacing=spacing)
            mu_g = (mg.get("mu_hat") or [None] * len(mhs))[j]
            mu_m = mf.get("value") if isinstance(mf, dict) else None
            sig = (mf.get("sigma") or mf.get("sigma_hess")) if isinstance(mf, dict) else None
            ok_mu = None
            if mu_g is not None and mu_m is not None:
                tol = LINK_MU_REL * max(float(sig or 0.3), 0.05) + LINK_MU_ABS
                ok_mu = bool(abs(float(mu_g) - float(mu_m)) <= tol)
                link.update(mu_grid=float(mu_g), mu_model=float(mu_m), mu_tol=tol)
            ok_nll = None
            t_g = (mg.get("twice_nll_abs") or [None] * len(mhs))[j]
            eng = (mg.get("point_engines") or [None] * len(mhs))[j]
            if eng is None:
                eng = "jax" if (mg.get("engine_points") or {}).get("pyhf", 0) == 0 else "pyhf"
            by_eng = _model_twice_nll(rec)
            t_m = by_eng.get(eng)
            cross = t_m is None and any(v is not None for v in by_eng.values())
            if cross:
                t_m = next(v for v in by_eng.values() if v is not None)
            if t_g is not None and t_m is not None:
                ok_nll = bool(abs(float(t_g) - float(t_m)) <= LINK_TWICE_NLL_TOL)
                link.update(twice_nll_grid=float(t_g), twice_nll_model=float(t_m), engine=eng,
                            **({"cross_engine": True} if cross else {}))
            fit_ok = bool(near and ok_mu and ok_nll)
            if not fit_ok:
                rng = mg.get("mass_range")
                outside = (isinstance(rng, (list, tuple)) and len(rng) == 2
                           and not (float(rng[0]) <= float(mh_rep) <= float(rng[1])))
                why = (f"the reported mH {float(mh_rep):g} lies outside the {float(rng[0]):g}-"
                       f"{float(rng[1]):g} GeV mass range (only in-range grid points are fitted)"
                       if not near and outside
                       else "no grid point within the local grid spacing of the reported mH" if not near
                       else "MODEL.json's fit cannot be compared with the grid point at the reported mH "
                            "(no fitted mu or 2NLL)" if ok_mu is None or ok_nll is None
                       else "the grid point at the reported mH does not reproduce MODEL.json's fit")
    except Exception as e:  # noqa: BLE001
        grader_err = f"grid link: {_err(e)}"
        fit_ok, why = False, f"linkage check failed: {_err(e)}"
        link["evaluator_error"] = grader_err
    ident = None
    if points is not None and model_spec is not None:
        ident = grid_identity(points, mh_rep, model_spec)
        link.update(ident, fit_linked=fit_ok)
    if ident is not None and ident["identical"]:
        link.update(linked=True, by="identical workspace")
    elif fit_ok:
        link.update(linked=True, by="fit")
    else:
        note = ""
        if ident is not None and ident["identical"] is False:
            note = (f"; nor is the grid's workspace at {ident['identical_point']:g} GeV ({ident['identical_file']}) "
                    f"MODEL.json's: {ident.get('identical_diff')}")
        elif ident is not None and ident.get("identical_why") not in (None, "no reported mass"):
            note = f"; nor does the grid hold MODEL.json at the reported mH: {ident['identical_why']}"
        link.update(linked=False, why=f"{why}{note}")
    mg["link"] = link
    if link["linked"] is False:
        mg["linked"] = False
        mg["usable"], mg["usable_reason"] = False, f"grid is not MODEL.json's model: {link.get('why')}"
        if grader_err:
            mg["evaluator_error"] = grader_err
    elif link.get("by") == "identical workspace":
        _profile_at_model(rec, reported_mh)


def _reported_numbers(reported_mh):
    """(mH, its total) as mass_grid_check reads the reported mH: floats, None where absent (a total of 0 is none)."""
    if isinstance(reported_mh, dict) and reported_mh.get("value") is not None:
        return float(reported_mh["value"]), float(reported_mh.get("total") or 0.0) or None
    return None, None


def _profile_at_model(rec, reported_mh):
    """The scored profile at a grid linked by the identical workspace (module comment): at the identical point, when it
    is a fitted point, the lower of its own 2NLL and MODEL.json's on the same engine (_model_twice_nll; skipped when
    MODEL.json has none on that engine), the curve and its summary then rebuilt (refresh_profile).  Recorded as
    link["model_nll"] = {"point", "engine", "twice_nll_grid", "twice_nll_model", "used"} (or {"used": False, "why"});
    the point's own fit stays in its row as "twice_nll_fit" when it is replaced.  Never raises: a failure is
    recorded there as the evaluator's and the profile left as fitted."""
    mg = rec.get("mass_grid")
    link = mg.get("link") if isinstance(mg, dict) else None
    if not isinstance(link, dict) or link.get("by") != "identical workspace":
        return
    try:
        m = link.get("identical_point")
        mhs = [float(c[0]) for c in mg.get("curve") or []]
        j = next((i for i, x in enumerate(mhs) if m is not None and x == float(m)), None)
        if j is None:
            link["model_nll"] = {"used": False, "why": "the identical workspace is not a fitted grid point (outside "
                                                       "the mass range)"}
            return
        tabs = mg.get("twice_nll_abs") or []
        eng = (mg.get("point_engines") or [None] * len(mhs))[j]
        t_m = _model_twice_nll(rec).get(eng)
        t_g = tabs[j] if j < len(tabs) else None
        if t_m is None or t_g is None:
            link["model_nll"] = {"used": False, "point": mhs[j], "engine": eng,
                                 "why": f"MODEL.json has no 2NLL on the point's engine ({eng})" if t_m is None
                                 else "the point has no 2NLL"}
            return
        use = bool(float(t_m) < float(t_g))
        if use:
            new = dict(mg, twice_nll_abs=[float(t) for t in tabs],
                       points=[dict(r) for r in mg.get("points") or [] if isinstance(r, dict)])
            if isinstance(mg.get("content"), dict):
                new["content"] = dict(mg["content"])
            new["twice_nll_abs"][j] = float(t_m)
            row = next((r for r in new["points"] if r.get("mass") == mhs[j]), None)
            if row is not None:
                row["twice_nll_fit"], row["twice_nll_abs"] = float(t_g), float(t_m)
            refresh_profile(new, reported_mh)
            mg.update(new)
        link["model_nll"] = {"point": mhs[j], "engine": eng, "twice_nll_grid": float(t_g),
                             "twice_nll_model": float(t_m), "used": use}
    except Exception as e:  # noqa: BLE001
        link["model_nll"] = {"used": False, "evaluator_error": f"profile at MODEL.json: {_err(e)}"}


def restore_fitted_profile(mg):
    """Undo _profile_at_model on a stored mass_grid: every row's "twice_nll_fit" back into twice_nll_abs (row and
    curve).  Returns whether anything changed (the caller rebuilds the profile, refresh_profile)."""
    changed = False
    rows = {r.get("mass"): r for r in mg.get("points") or [] if isinstance(r, dict)}
    mhs = [float(c[0]) for c in mg.get("curve") or []]
    tabs = mg.get("twice_nll_abs") or []
    for i, m in enumerate(mhs):
        r = rows.get(m)
        if r is not None and r.get("twice_nll_fit") is not None:
            r["twice_nll_abs"] = r.pop("twice_nll_fit")
            if i < len(tabs):
                tabs[i] = r["twice_nll_abs"]
            changed = True
    return changed


def refresh_profile(mg, reported_mh):
    """Rebuild a mass_grid's profile from its per-point 2NLL (twice_nll_abs, aligned with the curve): the curve and its
    summary (_interval_summary), the coverage facts and MODEL.json's offset from the minimum (grid_coverage), and the
    report's usable flag and root widths (scoring.grid_profile, as mass_grid_check sets them).  No fit."""
    mh_rep, sig_rep = _reported_numbers(reported_mh)
    mhs = [float(c[0]) for c in mg["curve"]]
    tabs = [float(t) for t in mg["twice_nll_abs"]]
    mg.update(_interval_summary([[m, t] for m, t in zip(mhs, tabs)], mh_rep, sig_rep,
                                uniform=bool(mg.get("grid_uniform", True)), mismatch=mg.get("grid_mismatch"),
                                n_failed=int(mg.get("n_failed") or 0)))
    rows = mg.get("points")
    if isinstance(rows, list):
        rng = mg.get("mass_range")
        cov, off = grid_coverage(rows, tuple(rng) if rng else None, model_mass=mh_rep)
        mg["coverage"] = cov
        if isinstance(mg.get("content"), dict):
            mg["content"]["model_off_minimum"] = off
    try:
        from sbh.kinds.binned_likelihood.scoring import grid_profile
        gp = grid_profile(mg)
        mg.update(usable=gp["usable"], usable_reason=gp["reason"], sigma_root=gp.get("sigma_root"),
                  sigma_lo_root=gp.get("sigma_lo_root"), sigma_hi_root=gp.get("sigma_hi_root"))
    except Exception as e:  # noqa: BLE001
        mg["usable_error"] = f"{type(e).__name__}: {e}"


_load_grid = load_grid


def _grid_signature(spec):
    """Structural identity of a grid workspace: channels, bins, observations, modifier set, backgrounds in full,
    measurement block.  Must be identical across the grid and to MODEL.json (only signal templates move with
    mH), so the 2*NLL values form one profile likelihood."""
    channels = spec.get("channels", []) or []
    names = sorted(c.get("name", "?") for c in channels)
    bins = {c.get("name", "?"): sorted({len(s.get("data", []) or [])
                                        for s in c.get("samples", []) or []})
            for c in channels}
    obs = {o.get("name", "?"): tuple(float(x) for x in o.get("data", []) or [])
           for o in spec.get("observations", []) or []}
    mods = sorted({(m.get("name", "?"), m.get("type", "?"))
                   for c in channels for s in c.get("samples", []) or []
                   for m in s.get("modifiers", []) or []})
    poi = next((m.get("config", {}).get("poi") for m in spec.get("measurements", []) or []), "mu")
    bkg = {c.get("name", "?"): json.dumps(sorted(
        [s.get("name", "?"), s.get("data"), sorted((json.dumps(m, sort_keys=True) for m in s.get("modifiers", []) or []))]
        for s in c.get("samples", []) or []
        if not any(m.get("type") == "normfactor" and m.get("name") == poi for m in s.get("modifiers", []) or [])),
        sort_keys=True, default=str) for c in channels}
    return {"channel names": names, "bin counts": bins,
            "observations": obs, "modifier set": mods, "backgrounds": bkg, "measurement": _measurement_sig(spec)}


def _measurement_cfg(spec):
    """Config of the first measurement (Workspace.model()'s default)."""
    ms = spec.get("measurements") or []
    m = ms[0] if isinstance(ms, list) and ms and isinstance(ms[0], dict) else {}
    return m.get("config") if isinstance(m.get("config"), dict) else {}


def _measurement_sig(spec):
    """The POI and every parameter's settings, as model identity (else a grid point could fix or re-bound one)."""
    cfg = _measurement_cfg(spec)
    pars = {str(p.get("name")): {k: v for k, v in p.items() if k != "name"}
            for p in cfg.get("parameters") or [] if isinstance(p, dict)}
    return json.dumps({"poi": cfg.get("poi"), "parameters": pars}, sort_keys=True, default=str)


def _fixed_poi(spec):
    """The POI's name when the measurement fixes it, else None."""
    cfg = _measurement_cfg(spec)
    poi = cfg.get("poi")
    for p in cfg.get("parameters") or []:
        if isinstance(p, dict) and p.get("name") == poi and p.get("fixed"):
            return str(poi)
    return None


class _GridTwin:
    """Per-point grid fitter on the JAX twin, compiled once: later structure-identical points reuse the first
    point's graph with their own arrays (avoids a ~1 s compile per point).  A different structure gets its own
    graph; an unbuildable point returns None -> pyhf.  2*NLL is the full -2 log L, so twin and pyhf points
    share one curve.  ``shipped``: each point's twin as the shipped grader built it (jax_engine.build)."""

    def __init__(self, shipped=False):
        self.shipped = bool(shipped)
        self.template = None
        self.n_compiled = self.n_shared = self.n_floor = 0
        self.fallbacks = []

    def fit_point(self, spec, model, data, init_pars):
        """The free fit of one grid workspace: {"x", "twice_nll_abs", "nll_min" (full -ln L), "sigma_hess" (POI
        Hessian width or None), "em", "main", "aux", "const"}; None when the twin cannot build it (-> pyhf)."""
        try:
            em, why = _evt().build(spec, model, share=self.template, shareable=True, shipped=self.shipped)
        except Exception as e:  # noqa: BLE001
            em, why = None, f"jax engine unavailable: {type(e).__name__}: {e}"
        if em is None:
            self.fallbacks.append(why)
            return None
        if self.template is None:
            self.template = em
        if em.shared:
            self.n_shared += 1
        else:
            self.n_compiled += 1
        d = np.asarray(data, dtype=float)
        main, aux = d[:em.n_main], d[em.n_main:]
        r = em.fit(main, aux, init=init_pars, want_sigma=True)
        if not r["valid"]:
            raise _TwinFitError("twin grid fit invalid")
        self.n_floor += int(r["n_floor"] > 0)
        const = float(em.full_nll_const(main, aux))
        poi = em.poi_index
        sg = r["sigma"]
        sig = float(sg[poi]) if sg is not None and np.isfinite(sg[poi]) and sg[poi] > 0 else None
        return {"x": np.asarray(r["x"], dtype=float), "nll_min": float(r["nll"]) + const,
                "twice_nll_abs": 2.0 * (float(r["nll"]) + const), "sigma_hess": sig,
                "em": em, "main": main, "aux": aux, "const": const}

    def fit(self, spec, model, data, init_pars):
        """(x, 2*NLL) of the free fit (the older interface), None -> pyhf."""
        got = self.fit_point(spec, model, data, init_pars)
        return None if got is None else (got["x"], got["twice_nll_abs"])


def _pyhf_point(model, data, init_pars):
    """The free fit of one grid workspace on pyhf: {"x", "twice_nll_abs", "nll_min", "sigma_hess"} (the Hessian width
    only when the backend's optimizer returns uncertainties, i.e. minuit)."""
    poi = model.config.poi_index
    sig = None
    try:
        fitted, tnll = pyhf.infer.mle.fit(data, model, init_pars=init_pars, return_uncertainties=True,
                                          return_fitted_val=True)
        fitted = np.asarray(fitted, dtype=float)
        if fitted.ndim == 2:
            x, u = fitted[:, 0], float(fitted[poi, 1])
            sig = u if np.isfinite(u) and u > 0 else None
        else:
            x = fitted
    except Exception:  # noqa: BLE001
        pars, tnll = pyhf.infer.mle.fit(data, model, init_pars=init_pars, return_fitted_val=True)
        x = np.asarray(pars, dtype=float)
    tnll = float(tnll)
    return {"x": x, "twice_nll_abs": tnll, "nll_min": 0.5 * tnll, "sigma_hess": sig}


def scan_point(model, data, fit, ev=None, prior_curve=False, support=MU_SUPPORT):
    """The conditional mu scan of one fitted grid point (``fit`` from _GridTwin.fit_point or _pyhf_point): mu_scan
    over mu_scan_set, walked outward from the free fit with the warm start (conditional_scan); with ``prior_curve``
    also sigma0_ref (sigma0_width).  Returns {"mu_scan", "n_scan_failed"} and, with prior_curve, "sigma0_ref" (None
    when the width fails on both engines, "sigma0_ref_error" saying why) and "sigma0_ref_engine"."""
    poi = model.config.poi_index
    lo_b, hi_b = (float(b) for b in model.config.suggested_bounds()[poi])
    x = np.asarray(fit["x"], dtype=float)
    mu_hat = float(x[poi])
    cond, reset, state = _conditional_fitters(model, data, ev, x, nll_offset=fit.get("const", 0.0))
    mus = mu_scan_set(mu_hat, fit.get("sigma_hess"), lo_b, hi_b, support=support)
    want = (0.0,) if prior_curve else ()
    scan, n_fail, xs = conditional_scan(cond, reset, mu_hat, fit.get("nll_min"), mus, want_x=want, state=state)
    out = {"mu_scan": [[round(m, 6), round(v, 6)] for m, v in scan], "n_scan_failed": int(n_fail)}
    if prior_curve:
        x0 = next((v for m, v in xs.items() if abs(m) <= SCAN_DEDUP), None)
        if x0 is None:
            reset()
            x0 = np.asarray(state["x"], dtype=float).copy() if cond(0.0) is not None else None
        out.update(sigma0_width(model, data, x0, ev))
    return out


def sigma0_width(model, data, x0, ev=None):
    """The marginal mu rule's prior scale at one grid point (the reference run, --prior-curve): the background-only
    Fisher width of mu, i.e. the POI's Hessian width on the mu = 0 Asimov expectation of the main bins at x0 (the
    observed-data mu = 0 conditional fit), auxiliary data as observed, every parameter free.  Twin: the exact
    autodiff Hessian at x0 (jax_engine._sigma).  pyhf (no twin, or the twin's Hessian not positive definite): Minuit's
    HESSE after a fit to the same Asimov data started at x0 (the minuit backend; the fit moves the nuisances by the
    pull of the observed aux, a second-order difference).  {"sigma0_ref", "sigma0_ref_engine"[, "sigma0_ref_error"]}."""
    out = {"sigma0_ref": None, "sigma0_ref_engine": None}
    if x0 is None:
        out["sigma0_ref_error"] = "no mu = 0 conditional fit"
        return out
    poi = model.config.poi_index
    d = np.asarray(data, dtype=float)
    n_main = model.config.nmaindata
    errs = []
    if ev is not None:
        try:
            asimov = np.asarray(ev.expected_main(x0), dtype=float)
            sg = ev._sigma(x0, asimov, d[n_main:])
            if sg is not None and np.isfinite(sg[poi]) and sg[poi] > 0:
                return {"sigma0_ref": float(sg[poi]), "sigma0_ref_engine": "jax"}
            errs.append("twin Hessian not positive definite")
        except Exception as e:  # noqa: BLE001
            errs.append(f"twin: {_err(e)}")
    try:
        main = np.asarray(model.expected_actualdata(pyhf.tensorlib.astensor(np.asarray(x0, dtype=float))), dtype=float)
        asimov = np.concatenate([main, d[n_main:]])
        fitted = np.asarray(pyhf.infer.mle.fit(asimov, model, init_pars=[float(v) for v in x0],
                                               return_uncertainties=True), dtype=float)
        u = float(fitted[poi, 1]) if fitted.ndim == 2 else float("nan")
        if np.isfinite(u) and u > 0:
            return {"sigma0_ref": u, "sigma0_ref_engine": "pyhf"}
        errs.append("pyhf HESSE gave no width")
    except Exception as e:  # noqa: BLE001
        errs.append(f"pyhf: {_err(e)}")
    out["sigma0_ref_error"] = "; ".join(errs)
    return out



def _poi_name(spec):
    return str(_measurement_cfg(spec).get("poi") or "mu")


def template_facts(spec, meta=None, mass_range=None):
    """Per-point facts of a grid workspace from its JSON alone: yield_mu1 = the summed nominal of every POI-scaled
    sample over ALL channels (mu = 1, nuisances nominal); over the signal-window channels (usable
    ``meta["binning"][<channel>]["edges"]``, i.e. len = bins + 1, whose span overlaps ``mass_range`` when given) the
    POI-scaled template as a cumulative yield on the merged edges ("lineshape": {"edges", "cdf"}, for grid_content's
    binning-window correction; verify.py drops it before storing), the binning "window" [first edge, last edge], the
    template's "peak" (density maximum, averaged over +-PEAK_SMOOTH) and its core "centroid" (within
    +-CENTROID_HALF_WINDOW of the peak, clipped symmetrically at the window; None when the peak is closer than
    CENTROID_MIN_HALF to an edge or there is no usable binning).  n_signal_samples counts the POI-scaled samples."""
    poi = _poi_name(spec)
    binning = (meta or {}).get("binning") if isinstance(meta, dict) else None
    binning = binning if isinstance(binning, dict) else {}
    total, n_sig = 0.0, 0
    parts = []
    by_channel = {}
    for ch in spec.get("channels", []) or []:
        edges = None
        b = binning.get(ch.get("name"))
        if isinstance(b, dict):
            try:
                e = np.asarray(b.get("edges"), dtype=float)
                if e.ndim == 1 and len(e) >= 2 and np.all(np.isfinite(e)) and np.all(np.diff(e) > 0):
                    edges = e
            except (TypeError, ValueError):
                edges = None
        if edges is not None and mass_range is not None and (edges[-1] < float(mass_range[0])
                                                             or edges[0] > float(mass_range[1])):
            edges = None
        tmpl = None
        for s in ch.get("samples", []) or []:
            if not any(m.get("type") == "normfactor" and m.get("name") == poi for m in s.get("modifiers", []) or []):
                continue
            try:
                nom = np.asarray(s.get("data"), dtype=float)
            except (TypeError, ValueError):
                continue
            if nom.ndim != 1 or not np.all(np.isfinite(nom)):
                continue
            n_sig += 1
            total += float(nom.sum())
            by_channel[str(ch.get("name"))] = by_channel.get(str(ch.get("name")), 0.0) + float(nom.sum())
            if edges is not None and len(edges) == len(nom) + 1:
                tmpl = nom.copy() if tmpl is None else tmpl + nom
        if tmpl is not None:
            parts.append((edges, np.concatenate([[0.0], np.cumsum(tmpl)])))
    out = {"yield_mu1": float(total) if n_sig else None, "centroid": None, "peak": None, "window": None,
           "n_signal_samples": int(n_sig), "core_width": None, "eff_width": None, "core_half": None,
           "channel_shares": ({k: round(v / total, 6) for k, v in sorted(by_channel.items())}
                              if n_sig and total > 0 else None)}
    if not parts:
        return out
    grid = np.unique(np.concatenate([e for e, _ in parts]))
    cdf = sum(np.interp(grid, e, c, left=0.0, right=float(c[-1])) for e, c in parts)
    out["window"] = [float(grid[0]), float(grid[-1])]
    out["lineshape"] = {"edges": grid.tolist(), "cdf": cdf.tolist()}
    if not cdf[-1] > 0:
        return out
    fine_ax = _fine_template(grid, cdf, mass_range)
    if fine_ax is None:
        return out
    mid, fine, step = fine_ax
    k = max(int(round(PEAK_SMOOTH / step)), 0)
    dens = np.convolve(fine, np.ones(2 * k + 1), mode="same") if k else fine
    if not dens.max() > 0:
        return out
    peak = float(np.mean(mid[dens >= dens.max() * (1.0 - 1e-9)]))
    out["peak"] = peak
    h = min(CENTROID_HALF_WINDOW, peak - float(grid[0]), float(grid[-1]) - peak)
    if h >= CENTROID_MIN_HALF:
        core = _core_shape(mid, fine, step, peak, h)
        if core is not None:
            out.update(centroid=core[0], core_width=core[1], eff_width=core[2], core_half=float(h))
    return out


def _core_bin_width(p):
    """The median bin width (GeV) of a point's template within its core window (its lineshape's merged edges), None
    without them."""
    ls, pk, h = p.get("lineshape"), p.get("peak"), p.get("core_half")
    if not isinstance(ls, dict) or pk is None or not h:
        return None
    try:
        e = np.asarray(ls.get("edges"), dtype=float)
    except (TypeError, ValueError):
        return None
    w = np.diff(e)[(e[1:] > float(pk) - float(h)) & (e[:-1] < float(pk) + float(h))]
    return float(np.median(w)) if len(w) else None


def _fine_template(edges, cdf, mass_range=None):
    """(mid, yield per fine cell, step) of a template given as a cumulative yield on its bin edges, spread uniformly
    within each bin onto a FINE_STEP axis over the binning window (cut to the mass range +- FINE_MARGIN); None when
    there is no usable span or yield."""
    e, c = np.asarray(edges, dtype=float), np.asarray(cdf, dtype=float)
    if e.ndim != 1 or e.shape != c.shape or len(e) < 2 or not c[-1] > 0:
        return None
    lo, hi = float(e[0]), float(e[-1])
    if mass_range is not None:
        lo, hi = max(lo, float(mass_range[0]) - FINE_MARGIN), min(hi, float(mass_range[1]) + FINE_MARGIN)
    if not (math.isfinite(hi - lo) and hi - lo > 2 * FINE_STEP):
        return None
    n = int(min(math.ceil((hi - lo) / FINE_STEP), 20000))
    ax = np.linspace(lo, hi, n + 1)
    fine = np.clip(np.diff(np.interp(ax, e, c)), 0.0, None)
    return 0.5 * (ax[1:] + ax[:-1]), fine, (hi - lo) / n


def _core_shape(mid, fine, step, peak, h):
    """(centroid, rms width, sensitivity width) of a fine template's core within +-h of its peak, or None without
    yield there.  The sensitivity width is 1 / int f^2 with f the core normalised to 1: under a locally flat
    background the Fisher information of mu is ~ yield^2 int f^2 / b, so a template narrowed or peaked at a fixed
    yield shows here (2 sqrt(pi) sigma for a Gaussian core)."""
    w = (mid >= peak - h) & (mid <= peak + h)
    yc = fine[w]
    if not yc.sum() > 0:
        return None
    c = float((yc * mid[w]).sum() / yc.sum())
    rms = float(math.sqrt(max((yc * (mid[w] - c) ** 2).sum() / yc.sum(), 0.0)))
    return c, rms, float(step * yc.sum() ** 2 / (yc * yc).sum())


def _curve_fn(yield_curve):
    """ln Y(m) from the task's yield curve {"masses": [...], "yield": [...]} (log-linear in m, clamped at the ends),
    or None when the curve is unusable."""
    if not isinstance(yield_curve, dict):
        return None
    try:
        m = np.asarray(yield_curve.get("masses"), dtype=float)
        y = np.asarray(yield_curve.get("yield"), dtype=float)
    except (TypeError, ValueError):
        return None
    if m.ndim != 1 or m.shape != y.shape or len(m) < 2 or not np.all(np.isfinite(m)) or not np.all(y > 0):
        return None
    o = np.argsort(m)
    m, ly = m[o], np.log(y[o])
    return lambda v: float(np.interp(float(v), m, ly))


def _band(dm, side_reach):
    """The yield band half-width b(|m_j - m_0|): ln YIELD_BAND_NEAR within YIELD_BAND_NEAR_GEV, rising linearly to
    ln YIELD_BAND_FAR at the range end on that side (``side_reach`` = that end's distance from m_0)."""
    dm = abs(float(dm))
    if dm <= YIELD_BAND_NEAR_GEV or side_reach <= YIELD_BAND_NEAR_GEV:
        return float(YIELD_BAND_NEAR)
    f = min((dm - YIELD_BAND_NEAR_GEV) / (side_reach - YIELD_BAND_NEAR_GEV), 1.0)
    return float(YIELD_BAND_NEAR + f * (YIELD_BAND_FAR - YIELD_BAND_NEAR))


def _window_fraction_fn(fitted, mass_range):
    """(f, lineshape point): f(m) = the fraction of the model's own lineshape CORE inside its binning window at mass
    m, relative to the lineshape point L (the fitted in-range point with a template nearest the middle of the range;
    f(m_L) = 1).  The core is L's template within +-WINDOW_CORE of its peak: the core moves with m_H, a far tail
    need not (a continuum-like low tail, a renormalised template), and the core is what a window edge at the range
    end cuts.  The template at m is L's scaled by m / m_L, so f(m) = the part of L's core inside the window scaled by
    m_L / m (what lies outside L's own window counts as nothing, so f <= 1).  (None, None) without a usable
    lineshape."""
    rows = [p for p in fitted if isinstance(p.get("lineshape"), dict)]
    if not rows:
        return None, None
    ms = [float(p["mass"]) for p in rows]
    mid = 0.5 * (float(mass_range[0]) + float(mass_range[1])) if mass_range is not None else float(np.median(ms))
    lp = min(rows, key=lambda p: abs(float(p["mass"]) - mid))
    try:
        e = np.asarray(lp["lineshape"]["edges"], dtype=float)
        c = np.asarray(lp["lineshape"]["cdf"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None, None
    if e.ndim != 1 or e.shape != c.shape or len(e) < 2 or not c[-1] > 0:
        return None, None
    m_l = float(lp["mass"])
    pk = lp.get("peak")
    lo_c, hi_c = (float(c[0]), float(c[-1])) if pk is None or WINDOW_CORE is None else \
        (float(np.interp(float(pk) - WINDOW_CORE, e, c)), float(np.interp(float(pk) + WINDOW_CORE, e, c)))
    if not hi_c > lo_c:
        return None, None

    def core_cdf(x):
        return min(max(float(np.interp(x, e, c)), lo_c), hi_c)

    def f(m):
        m = float(m)
        if not (math.isfinite(m) and m > 0):
            return 1.0
        s = m_l / m
        return (core_cdf(e[-1] * s) - core_cdf(e[0] * s)) / (hi_c - lo_c)
    return f, lp


def grid_content(points, mass_range, yield_curve, model_mass=None, model_yield=None):
    """The per-point template checks (module header) over the in-range fitted points of ``points`` (the per_seed
    mass_grid "points" rows: mass, in_range, status, mu_hat, bounds and template_facts' yield_mu1, centroid,
    lineshape).  Marks dead points IN PLACE (status "dead"; mu_hat on hi_b, or yield below DEAD_YIELD_FRAC of the
    curve) and adds "window_frac" (f_j, the binning-window fraction the yields are divided by), "ln_yield_ratio"
    (ln r_j) and, above the yield band, "mu_stretch" (the band clip the score applies).  m_0 / S_0 (``model_mass`` / ``model_yield``: MODEL.json's mass and raw yield, else the argmin point's)
    anchor the band.  Without a usable curve every check is skipped (usable, reason "no yield curve").  Returns the
    "content" block: yield_band_ok, centroid_ok, smooth_ok, n_dead, dead_ok, usable (every check that ran passed),
    reasons, anchor, window_correction, model_off_minimum (filled by grid_coverage), and the template-shape checks
    width_ok / share_ok (_shape_checks: a narrowed, peaked or channel-shifted template at a fixed yield)."""
    ln_y = _curve_fn(yield_curve)
    reasons = []
    fitted = [p for p in points if p.get("in_range") and p.get("status") in ("ok", "dead")]
    blk = {"yield_band_ok": None, "centroid_ok": None, "smooth_ok": None, "n_dead": 0, "dead_ok": None,
           "width_ok": None, "share_ok": None, "usable": True, "reasons": reasons, "anchor": None,
           "n_yield_clipped": 0, "centroid_line": None,
           "window_correction": None, "model_off_minimum": None}
    if ln_y is None:
        reasons.append("no yield curve")
        return blk
    if not fitted:
        reasons.append("no fitted in-range point")
        return {**blk, "usable": False}
    outside = (model_mass is not None and mass_range is not None and np.isfinite(float(model_mass))
               and not float(mass_range[0]) <= float(model_mass) <= float(mass_range[1]))
    if outside:
        reasons.append(f"MODEL.json's mass {float(model_mass):g} GeV is outside the range: the band is anchored on "
                       "the grid's in-range minimum")
        model_yield = None
    if model_mass is None or not np.isfinite(float(model_mass)) or outside:
        anchor = min((p for p in fitted if p.get("twice_nll_abs") is not None), key=lambda p: p["twice_nll_abs"],
                     default=None)
        if anchor is None:
            mid = (0.5 * (float(mass_range[0]) + float(mass_range[1])) if mass_range is not None
                   else float(np.median([float(p["mass"]) for p in fitted])))
            anchor = min(fitted, key=lambda p: abs(float(p["mass"]) - mid))
        model_mass = float(anchor["mass"])
        if model_yield is None:
            model_yield = anchor.get("yield_mu1")
    elif model_yield is None:
        near = min(fitted, key=lambda p: abs(float(p["mass"]) - float(model_mass)))
        model_yield = near.get("yield_mu1")
    m0 = float(model_mass)
    s0 = float(model_yield) if model_yield is not None and np.isfinite(float(model_yield)) and float(model_yield) > 0 \
        else None
    lo, hi = (float(mass_range[0]), float(mass_range[1])) if mass_range is not None else (-np.inf, np.inf)
    wf, lp = _window_fraction_fn(fitted, mass_range)
    if wf is None:
        reasons.append("no lineshape (no _meta.binning edges in the rows): yields not corrected for the binning window")
    else:
        blk["window_correction"] = {"lineshape_mass": float(lp["mass"]), "window": lp.get("window")}

    def frac(m):
        return 1.0 if wf is None else max(wf(m), WINDOW_FRAC_FLOOR)

    s0c = None if s0 is None else s0 / frac(m0)
    blk["anchor"] = {"mass": m0, "yield_mu1": s0, "window_frac": None if wf is None else round(wf(m0), 6)}
    if s0c is None:
        reasons.append("MODEL.json has no POI-scaled signal yield: yield checks skipped")
    n_dead = 0
    for p in fitted:
        dead = None
        mu, b = p.get("mu_hat"), p.get("bounds")
        if mu is not None and isinstance(b, (list, tuple)) and len(b) == 2 and b[1] is not None \
                and np.isfinite(float(b[1])) and float(mu) >= float(b[1]) - DEAD_BOUND_REL * max(1.0, abs(float(b[1]))):
            dead = "mu_hat on the upper POI bound"
        s = p.get("yield_mu1")
        if wf is not None:
            p["window_frac"] = round(wf(p["mass"]), 6)
        if s0c is not None and s is not None:
            sc = float(s) / frac(p["mass"])
            expect = s0c * math.exp(ln_y(p["mass"]) - ln_y(m0))
            if sc < DEAD_YIELD_FRAC * expect:
                dead = dead or f"yield {sc:.3g} below {DEAD_YIELD_FRAC:g} of the curve ({expect:.3g})"
            ln_r = _ln_ratio(sc, s0c)
            p["ln_yield_ratio"] = (ln_r - (ln_y(p["mass"]) - ln_y(m0))) if ln_r is not None else None
        if dead:
            p["status"], p["why"] = "dead", dead
            n_dead += 1
    dead_ok = bool(n_dead <= DEAD_MAX_FRAC * len(fitted))
    if not dead_ok:
        reasons.append(f"{n_dead} of {len(fitted)} in-range points dead (> {DEAD_MAX_FRAC:.0%})")
    live = [p for p in fitted if p["status"] == "ok"]
    yield_band_ok = None
    for p in fitted:
        p.pop("mu_stretch", None)
    if s0c is not None:
        above, below = [], []
        for p in live:
            r = p.get("ln_yield_ratio")
            if r is None:
                continue
            m = float(p["mass"])
            reach = (hi - m0) if m >= m0 else (m0 - lo)
            b = _band(abs(m - m0), reach if np.isfinite(reach) else YIELD_BAND_NEAR_GEV)
            if r > b + 1e-9:
                p["mu_stretch"] = round(math.exp(min(r - b, LN_FLOAT_MAX)), 6)
                above.append(m)
            elif r < -b - 1e-9:
                below.append(m)
        yield_band_ok = not above and not below
        blk["n_yield_clipped"] = len(above)
        if above:
            mx = max(float(p["mu_stretch"]) for p in live if p.get("mu_stretch"))
            reasons.append(f"yield band: {len(above)} point(s) above it, clipped to the band edge (e.g. "
                           f"{above[0]:g} GeV; largest factor {mx:.3g})")
        if below:
            reasons.append(f"yield band: {len(below)} point(s) below it, kept (e.g. {below[0]:g} GeV)")
    cen = [(float(p["mass"]), float(p["centroid"])) for p in live if p.get("centroid") is not None]
    centroid_ok = None
    if cen:
        ms, cs = np.array([m for m, _ in cen]), np.array([c for _, c in cen])
        if len(cen) >= 3 and float(ms.max() - ms.min()) >= CENTROID_MIN_SPAN:
            try:
                slope, icpt = (float(v) for v in np.polyfit(ms, cs, 1))
            except (np.linalg.LinAlgError, ValueError):
                slope = icpt = float("nan")
        else:
            slope = 1.0
            icpt = float(np.median(cs - ms))
        res = cs - (icpt + slope * ms)
        far = [float(m) for m, d in zip(ms, res) if not abs(d) <= CENTROID_CONST_TOL]
        steep = not (CENTROID_SLOPE[0] - 1e-9 <= slope <= CENTROID_SLOPE[1] + 1e-9)
        centroid_ok = not far and not steep
        blk["centroid_line"] = {"slope": round(slope, 4), "offset_at_125": round(icpt + (slope - 1.0) * 125.0, 3),
                                "max_residual": round(float(np.max(np.abs(res))), 3)}
        if steep:
            reasons.append(f"centroid: does not track the label (slope {slope:.3f} outside "
                           f"[{CENTROID_SLOPE[0]:g}, {CENTROID_SLOPE[1]:g}])")
        if far:
            reasons.append(f"centroid: {len(far)} point(s) more than {CENTROID_CONST_TOL:g} GeV off the line through "
                           f"the others (e.g. {far[0]:g} GeV)")
    elif not live:
        reasons.append("no live point: centroid unchecked")
    else:
        reasons.append("no _meta.binning edges: centroid unchecked")
    smooth_ok = None
    if s0c is not None:
        seq = sorted((float(p["mass"]), float(p["yield_mu1"]) / frac(p["mass"])) for p in live
                     if p.get("yield_mu1") is not None and float(p["yield_mu1"]) > 0)
        kinks = []
        for (ma, sa), (mb, sb) in zip(seq, seq[1:]):
            if abs(math.log(sb) - math.log(sa)) > abs(ln_y(mb) - ln_y(ma)) + SLOPE_EXTRA * (mb - ma):
                kinks.append(ma)
        smooth_ok = not kinks
        if kinks:
            reasons.append(f"smoothness: {len(kinks)} step(s) steeper than the curve (e.g. after {kinks[0]:g} GeV)")
    width_ok, share_ok = _shape_checks(live, lp, mass_range, reasons, blk)
    usable = all(v is not False for v in (centroid_ok, smooth_ok, dead_ok, width_ok, share_ok))
    blk.update(yield_band_ok=yield_band_ok, centroid_ok=centroid_ok, smooth_ok=smooth_ok, n_dead=int(n_dead),
               dead_ok=dead_ok, width_ok=width_ok, share_ok=share_ok, usable=bool(usable))
    return blk


def _shape_checks(live, lp, mass_range, reasons, blk):
    """(width_ok, share_ok) of the live points (module header: width, shares), reasons appended, the largest
    deviations into ``blk`` ("width_dev", "width_dev_local", "width_bw", "share_dev").  The widths are compared with the
    lineshape point L's core read in the SAME half-window (a core clipped at a binning edge is narrower for that
    reason alone); None = not evaluable (no lineshape, fewer than 3 points with widths, no channel shares)."""
    width_ok = share_ok = None
    fine_l = None
    if lp is not None and lp.get("peak") is not None and isinstance(lp.get("lineshape"), dict):
        fine_l = _fine_template(lp["lineshape"].get("edges"), lp["lineshape"].get("cdf"), mass_range)
    if fine_l is not None:
        mid, fine, step = fine_l
        pk = float(lp["peak"])
        h_l = min(CENTROID_HALF_WINDOW, pk - float(mid[0]), float(mid[-1]) - pk)
        m_l = float(lp["mass"])
        rows = []
        for p in live:
            h = p.get("core_half")
            if None in (p.get("core_width"), p.get("eff_width"), h) or not h > 0:
                continue
            ref = _core_shape(mid, fine, step, pk, min(float(h), h_l))
            if ref is None or not (ref[1] > 0 and ref[2] > 0 and p["core_width"] > 0 and p["eff_width"] > 0):
                continue
            rho_rms, rho_eff = _ln_ratio(p["core_width"], ref[1]), _ln_ratio(p["eff_width"], ref[2])
            if rho_rms is None or rho_eff is None:
                continue
            rows.append((float(p["mass"]), rho_rms, rho_eff, _core_bin_width(p)))
        if len(rows) >= 3:
            ms = np.array([r[0] for r in rows])
            bws = [r[3] for r in rows if r[3] is not None]
            bw = float(np.median(bws)) if bws else 0.0
            tol_g = WIDTH_TOL[0] + WIDTH_TOL[1] * bw
            tol_l = WIDTH_TOL_LOCAL[0] + WIDTH_TOL_LOCAL[1] * bw
            worst_g, worst_l, bad_g, bad_l = 0.0, 0.0, None, None
            for col, label in ((1, "rms"), (2, "sensitivity")):
                rho = np.array([r[col] for r in rows])
                best = None
                for a in WIDTH_SCALING:
                    res = rho - a * np.log(ms / m_l)
                    dev = np.abs(res - np.median(res))
                    if best is None or dev.max() < best[0]:
                        best = (float(dev.max()), dev, a)
                if best[0] > worst_g:
                    worst_g, bad_g = best[0], (label, ms[best[1] > tol_g + 1e-12], best[2])
                loc = np.zeros(len(rho))
                for i in range(len(rho)):
                    nb = (np.abs(ms - ms[i]) <= WIDTH_LOCAL_GEV + 1e-9) & (np.arange(len(rho)) != i)
                    if nb.sum() >= 2:
                        loc[i] = abs(rho[i] - float(np.median(rho[nb])))
                if loc.max() > worst_l:
                    worst_l, bad_l = float(loc.max()), (label, ms[loc > tol_l + 1e-12])
            width_ok = bool(worst_g <= tol_g + 1e-12 and worst_l <= tol_l + 1e-12)
            blk.update(width_dev=round(worst_g, 4), width_dev_local=round(worst_l, 4), width_bw=round(bw, 3))
            if worst_g > tol_g + 1e-12:
                label, bad, a = bad_g
                reasons.append(f"template width: the core's {label} width departs from the lineshape point's by up to "
                               f"{worst_g:.2f} in ln (> {tol_g:.3g} at {bw:g} GeV bins, best common scaling "
                               f"(m / m_L)^{a:g}) at {len(bad)} point(s) (e.g. {bad[0]:g} GeV)")
            if worst_l > tol_l + 1e-12:
                label, bad = bad_l
                reasons.append(f"template width: the core's {label} width departs from its neighbours' (within "
                               f"{WIDTH_LOCAL_GEV:g} GeV) by up to {worst_l:.2f} in ln (> {tol_l:.3g} at {bw:g} GeV "
                               f"bins) at {len(bad)} point(s) (e.g. {bad[0]:g} GeV)")
    sh = [(float(p["mass"]), p["channel_shares"]) for p in live if isinstance(p.get("channel_shares"), dict)]
    if len(sh) >= 3:
        chans = sorted({c for _, d in sh for c in d})
        worst, bad = 0.0, []
        for c in chans:
            v = np.array([float(d.get(c, 0.0)) for _, d in sh])
            dev = np.abs(v - np.median(v))
            worst = max(worst, float(dev.max()))
            bad += [(m, c) for (m, _), x in zip(sh, dev) if x > SHARE_TOL + 1e-12]
        share_ok = not bad
        blk["share_dev"] = round(worst, 4)
        if bad:
            reasons.append(f"channel shares: a channel's share of the signal yield departs from its median by up to "
                           f"{worst:.2f} (> {SHARE_TOL:g}) at {len({m for m, _ in bad})} point(s) (e.g. {bad[0][0]:g} "
                           f"GeV, {bad[0][1]})")
    return width_ok, share_ok


def _twice_nll_on_support(p, support=MU_SUPPORT):
    """2 x the minimum of a point's profile on the mu support within its own bounds (scoring.nll_min_on_support's
    rule, here without the scorer): the lowest scanned value there, or the free fit when it lies there; the free fit's
    twice_nll_abs for a point without a scan (a task that does not scan ranks as before).  A deficit point (mu_hat
    below the support under a declared lo_b < 0) is ranked at its l(0), not at its free fit outside the support."""
    scan = p.get("mu_scan")
    if not scan:
        return p.get("twice_nll_abs")
    b = p.get("bounds") or (-np.inf, np.inf)
    try:
        lo, hi = max(float(support[0]), float(b[0])), min(float(support[1]), float(b[1]))
    except (TypeError, ValueError, IndexError):
        lo, hi = float(support[0]), float(support[1])
    vals = []
    for r in scan:
        try:
            m, v = float(r[0]), float(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(m) and np.isfinite(v) and lo - 1e-9 <= m <= hi + 1e-9:
            vals.append(v)
    mh, nm = p.get("mu_hat"), p.get("nll_min")
    if mh is not None and nm is not None and lo - 1e-9 <= float(mh) <= hi + 1e-9:
        vals.append(float(nm))
    return 2.0 * min(vals) if vals else None


def grid_coverage(points, mass_range, model_mass=None, shipped=False):
    """The coverage facts of the usable (status "ok") in-range points: first/last mass, n_in_range, the largest
    in-range gap (the range ends count as neighbours: a grid starting late leaves a gap from lo), reaches_lo/hi
    within REACH_TOL, ok = both reached and max gap <= MAX_GAP_IN_RANGE (None without a mass range), argmin_point
    (index into ``points`` of the in-range 2NLL minimum on the mu support, _twice_nll_on_support; among a tie set
    within LINK_TWICE_NLL_TOL the member with the largest sigma_hess), tie_set, tie_spread_2nll.  Also
    model_off_minimum = {dq, dm, flag}: MODEL.json's point (nearest fitted to ``model_mass``) against the global
    minimum, flag when dq > LINK_TWICE_NLL_TOL.  Only points whose 2NLL on the support is a finite number rank, so the
    tie set is never empty where one does; ``shipped`` (a task kept on the rule it shipped with): every point with a
    2NLL, as that grader ranked them."""
    idx = [i for i, p in enumerate(points) if p.get("in_range") and p.get("status") == "ok"]
    out = {"first_mass": None, "last_mass": None, "n_in_range": len(idx), "max_gap_in_range": None,
           "reaches_lo": None, "reaches_hi": None, "ok": None, "argmin_point": None, "tie_set": [],
           "tie_spread_2nll": None}
    if not idx:
        out["ok"] = False if mass_range is not None else None
        return out, None
    ms = [float(points[i]["mass"]) for i in idx]
    out["first_mass"], out["last_mass"] = min(ms), max(ms)
    gaps = list(np.diff(sorted(ms)))
    if mass_range is not None:
        lo, hi = float(mass_range[0]), float(mass_range[1])
        gaps += [max(min(ms) - lo, 0.0), max(hi - max(ms), 0.0)]
        out["reaches_lo"] = bool(min(ms) <= lo + REACH_TOL)
        out["reaches_hi"] = bool(max(ms) >= hi - REACH_TOL)
    out["max_gap_in_range"] = float(max(gaps)) if gaps else 0.0
    if mass_range is not None:
        out["ok"] = bool(out["reaches_lo"] and out["reaches_hi"] and out["max_gap_in_range"] <= MAX_GAP_IN_RANGE)
    q2 = {i: _twice_nll_on_support(points[i]) for i in idx}
    with_q = [i for i in idx if q2[i] is not None and (shipped or np.isfinite(float(q2[i])))]
    off = None
    if with_q:
        qmin = min(float(q2[i]) for i in with_q)
        tie = [i for i in with_q if float(q2[i]) - qmin <= LINK_TWICE_NLL_TOL]
        out["tie_set"] = tie
        out["tie_spread_2nll"] = float(max(float(q2[i]) for i in tie) - qmin)
        out["argmin_point"] = max(tie, key=lambda i: (points[i].get("sigma_hess") or -1.0, -float(q2[i])))
        if model_mass is not None and np.isfinite(float(model_mass)):
            j = min(with_q, key=lambda i: abs(float(points[i]["mass"]) - float(model_mass)))
            dq = float(q2[j]) - qmin
            k = min(with_q, key=lambda i: float(q2[i]))
            off = {"dq": dq, "dm": abs(float(points[j]["mass"]) - float(points[k]["mass"])),
                   "flag": bool(dq > LINK_TWICE_NLL_TOL), "model_point": j}
    return out, off


def _envelope_crossings(mhs, q):
    """Copy of sbh.score.envelope_crossings (so this runs without sbh); keep the two in step."""
    below = [j for j in range(len(q)) if q[j] < 1.0]
    if not below:
        return None, None

    def at(a, b):
        f = (1.0 - q[a]) / (q[b] - q[a]) if q[b] != q[a] else 0.0
        return float(mhs[a] + f * (mhs[b] - mhs[a]))
    jl, jr = below[0], below[-1]
    return (at(jl, jl - 1) if jl > 0 else None), (at(jr, jr + 1) if jr < len(q) - 1 else None)


def _interval_summary(curve, mh_rep, sig_rep, uniform=True, mismatch=None, n_failed=0):
    """Summary of a 2*NLL curve [[mH, 2NLL], ...]: parabolic minimum -> mh_model, envelope 2*DeltalnL = 1
    crossings -> sigma_lo / sigma_hi, agreement with the reported mH (tolerances floored at the grid spacing).
    Used by mass_grid_check and the combine MH scan."""
    mhs = np.array([c[0] for c in curve], dtype=float)
    q = np.array([c[1] for c in curve], dtype=float)
    q -= q.min()
    i = int(np.argmin(q))
    gaps = np.diff(mhs)
    spacing = float(max(gaps[max(i - 1, 0)], gaps[min(i, len(gaps) - 1)]))
    mh_model = float(mhs[i])
    if 0 < i < len(mhs) - 1:
        d2 = q[i - 1] - 2.0 * q[i] + q[i + 1]
        if d2 > 1e-9:
            x1, x2 = mhs[i - 1] - mhs[i], mhs[i + 1] - mhs[i]
            y1, y2 = q[i - 1] - q[i], q[i + 1] - q[i]
            den = x1 * x2 * (x1 - x2)
            a = (y1 * x2 - y2 * x1) / den
            b = (y2 * x1 * x1 - y1 * x2 * x2) / den
            if a > 1e-12:
                mh_model = float(mhs[i] - 0.5 * b / a)

    x_lo, x_hi = _envelope_crossings(mhs, q)
    sigma_lo = float(mh_model - x_lo) if x_lo is not None else None
    sigma_hi = float(x_hi - mh_model) if x_hi is not None else None
    agree_mh = agree_sigma = None
    if mh_rep is not None:
        tol_mh = max(spacing, 0.25 * max(sig_rep or 0.0, 0.1) + 0.05)
        agree_mh = bool(abs(mh_model - mh_rep) <= tol_mh)
    sides = [s for s in (sigma_lo, sigma_hi) if s is not None]
    sigma_model = float(np.mean(sides)) if sides else None
    if sig_rep is not None and sigma_model is not None:
        agree_sigma = bool(abs(sigma_model - sig_rep) <= max(0.30 * sig_rep, spacing))
    if not uniform:
        agree_mh = False if agree_mh is not None else None
        agree_sigma = False if agree_sigma is not None else None
    return {"n_points": int(len(curve)), "n_failed": int(n_failed),
            "grid_uniform": bool(uniform), "grid_mismatch": mismatch,
            "grid_spacing": spacing,
            "mh_model": mh_model, "sigma_lo": sigma_lo, "sigma_hi": sigma_hi,
            "sigma_model": sigma_model,
            "mh_reported": mh_rep, "sigma_reported": sig_rep,
            "agree_mh": agree_mh, "agree_sigma": agree_sigma,
            "curve": [[float(m), float(v)] for m, v in zip(mhs, q)]}

