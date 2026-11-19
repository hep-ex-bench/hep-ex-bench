"""The model verifier's fit and engine helpers (verify.py): the JAX twin (build, observed-data fit), pyhf post-fit
impacts, the Asimov Z_exp, the profiled mu measurement (mu_fit_block), RESULT.json and the self-reported cross-
checks. Evaluator-only (sbh/harbor_pack.py EVALUATOR_ONLY); verify.py re-exports every name."""
import json
import math

import numpy as np
import pyhf


class _TwinFitError(RuntimeError):
    """A twin fit a stage cannot proceed without did not converge."""


class ModelFitError(RuntimeError):
    """A fit of the declared model raised (pyhf's MLE or Minuit on its likelihood): the model's failure, its message
    fit_error's.  Any other exception a stage raises is the evaluator's (verify.py records it as "evaluator_error")."""


def model_fit(fn, *args, **kwargs):
    """``fn(*args, **kwargs)``, a fit of the declared model; whatever it raises becomes ModelFitError(fit_error(e))."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        raise ModelFitError(fit_error(e)) from e


def _err(e):
    """'Type: message [at file:line]' for a stage's error record."""
    import traceback
    tb = traceback.extract_tb(e.__traceback__)
    where = f"{tb[-1].filename.rsplit('/', 1)[-1]}:{tb[-1].lineno}" if tb else "?"
    return f"{type(e).__name__}: {e} [at {where}]"


def _evt():
    """sbh/kinds/binned_likelihood/jax_engine, imported lazily (the pyhf path never touches jax)."""
    from sbh.kinds.binned_likelihood import jax_engine
    return jax_engine


_PYHF_HESSE_BUG = "'NoneType' object has no attribute 'correlation'"


def fit_error(e) -> str:
    """One fit attempt's error, short: the cause behind pyhf's HESSE AttributeError (a model whose covariance is
    singular at its minimum: the model's failure), else 'Type: message'."""
    if isinstance(e, AttributeError) and _PYHF_HESSE_BUG in str(e):
        return ("HESSE failed at a valid MIGRAD minimum: the covariance is singular, so the model leaves its "
                "parameters' uncertainties undefined (the model's failure; Minuit gave no covariance and pyhf raised "
                "AttributeError reading it)")
    return f"{type(e).__name__}: {e}"[:300]


def _twin_build(spec, model, poi_scaling=None, shipped=False):
    """(twin, reason): the model's JAX twin (``poi_scaling``: jax_engine's POI scaling; ``shipped``: as the shipped
    grader built it, jax_engine.build), or None and why the seed stays on pyhf."""
    try:
        return _evt().build(spec, model, poi_scaling=poi_scaling, shipped=shipped)
    except Exception as e:  # noqa: BLE001
        return None, f"jax engine unavailable: {type(e).__name__}: {e}"


POISSON_RATE_MAX = float(np.iinfo("l").max - np.sqrt(np.iinfo("l").max) * 10)


def poisson_aux_rates(model, pars) -> dict:
    """{constraint name: its rates} of the model's Poisson constraint terms (shapesys) at ``pars``: each parameter
    times its factor, the rate pyhf's constraint pdf draws its auxiliary count at."""
    cfg = model.config
    x = np.asarray(pars, dtype=float)
    out = {}
    for name in cfg.auxdata_order:
        ps = cfg.param_set(name)
        if ps.pdf_type == "poisson":
            out[name] = x[cfg.par_slice(name)] * np.asarray(getattr(ps, "factors", None) or [], dtype=float)
    return out


def check_toy_rates(model, pars, main_rates) -> None:
    """Raise ModelFitError (the model's failure: "fit failed", its GOF charged) where a toy drawn at ``pars`` would
    draw a count at a rate numpy's Poisson sampler refuses: a main-bin rate (``main_rates``, the rates the toy pdf
    draws at) or a Poisson constraint's (poisson_aux_rates) that is not a finite number in [0, POISSON_RATE_MAX].
    The rates come from the declared model at its own fit: a count of order 1e19 events cannot be drawn, and its
    observed GOF statistic of that order has p ~ 0."""
    def first_bad(v):
        v = np.asarray(v, dtype=float).ravel()
        bad = np.flatnonzero(~(np.isfinite(v) & (v >= 0.0) & (v <= POISSON_RATE_MAX)))
        return (int(bad[0]), float(v[bad[0]])) if len(bad) else None
    hit = first_bad(main_rates)
    where = None if hit is None else f"main bin {hit[0]}"
    if hit is None:
        for name, rates in poisson_aux_rates(model, pars).items():
            hit = first_bad(rates)
            if hit is not None:
                where = f"the Poisson constraint {name}" + (f"[{hit[0]}]" if len(np.ravel(rates)) > 1 else "")
                break
    if hit is not None:
        raise ModelFitError(f"an expected rate of {hit[1]:.4g} events ({where}, at the toys' generation point), "
                            f"beyond any Poisson draw (numpy draws a rate in [0, {POISSON_RATE_MAX:.4g}])")


TWIN_RESTART_DNLL = 1e-3


def _twin_obs_fit(ev, main, aux, pyhf_bestfit):
    """The twin's observed-data fit, started at pyhf's best fit.  Where the yield floor is active there (the
    likelihoods differ, so pyhf's basin need not be the twin's), also fit from init and keep the better one
    (TWIN_RESTART_DNLL; ``from`` = "pyhf_bestfit" / "init")."""
    fit = ev.fit(main, aux, init=np.asarray(pyhf_bestfit, float), want_sigma=False)
    fit["from"] = "pyhf_bestfit"
    fs = ev.floor_state(pyhf_bestfit)
    if fs["n_floor"] or fs["n_sample_floor"]:
        alt = ev.fit(main, aux, want_sigma=False)
        if alt["valid"] and (not fit["valid"] or alt["nll"] < fit["nll"] - TWIN_RESTART_DNLL):
            fit = dict(alt, **{"from": "init"})
    return fit


def _pyhf_postfit_impacts(model, data, res):
    """cabinetry.fit.ranking's post-fit impacts only (same refits, without the unrecorded pre-fit half).
    A failing refit raises, as in cabinetry, and costs the whole impacts block."""
    from cabinetry.fit import _fit_model
    poi = model.config.poi_index
    bestfit = np.asarray(res.bestfit, dtype=float)
    unc = np.asarray(res.uncertainty, dtype=float)
    init = list(model.config.suggested_init())
    fix = list(model.config.suggested_fixed())
    impacts = {}
    for i, lbl in enumerate(model.config.par_names):
        if i == poi:
            continue
        fx = fix.copy()
        fx[i] = True
        out = []
        for v in (bestfit[i] + unc[i], bestfit[i] - unc[i]):
            if v == bestfit[i]:
                out.append(0.0)
                continue
            ip = init.copy()
            ip[i] = v
            r = _fit_model(model, data, init_pars=ip, fix_pars=fx)
            out.append(float(r.bestfit[poi] - bestfit[poi]))
        impacts[lbl] = {"up": out[0], "down": out[1]}
    return impacts


def _z_exp(model, res, ev=None, obs_fit=None):
    """Asimov discovery significance of an SM signal: expected data at (pyhf best fit, mu = 1), q0 by pyhf or,
    with ``ev``, the twin.  On the twin the main counts are its floored rates (``asimov_floored``), and where
    the floor is active at pyhf's best fit the point is the twin's ``obs_fit`` (``asimov_from``), as combine
    builds its Asimov at its own fit.  Returns (z_exp float or {"error"}, info)."""
    import time
    t0 = time.perf_counter()
    poi = model.config.poi_index
    n_main = model.config.nmaindata
    pars_sm = np.asarray(res.bestfit, dtype=float).copy()
    pars_sm[poi] = 1.0
    asimov = np.asarray(model.expected_data(pars_sm), dtype=float)
    info = {"engine": "pyhf", "asimov_negative_bins": int(np.sum(asimov[:n_main] < 0))}
    if ev is not None:
        info["asimov_from"] = "pyhf_bestfit"
        fs = ev.floor_state(res.bestfit)
        if (fs["n_floor"] or fs["n_sample_floor"]) and obs_fit is not None and obs_fit["valid"]:
            pars_sm = np.asarray(obs_fit["x"], dtype=float).copy()
            pars_sm[poi] = 1.0
            asimov = np.asarray(model.expected_data(pars_sm), dtype=float)
            info["asimov_from"] = "twin_bestfit"
        floored = np.asarray(ev.expected_main(pars_sm), dtype=float)
        if not np.array_equal(floored, np.asarray(ev.expected_main_unfloored(pars_sm), dtype=float)):
            asimov[:n_main] = floored
            info["asimov_floored"] = True
        q0, qi = ev.q0(asimov[:n_main], asimov[n_main:])
        if q0 is not None:
            info.update(engine="jax", n_floor=qi["n_floor"], n_kink=qi["n_kink"],
                        wall_s=round(time.perf_counter() - t0, 2))
            return float(np.sqrt(max(q0, 0.0))), info
        info["engine_fallback"] = "twin q0 fits invalid"
    try:
        q0 = float(pyhf.infer.test_statistics.q0(
            0.0, pyhf.tensorlib.astensor(asimov), model, model.config.suggested_init(),
            model.config.suggested_bounds(), model.config.suggested_fixed()))
        z = float(np.sqrt(max(q0, 0.0)))
    except Exception as e:
        z = {"error": f"{type(e).__name__}: {e}"}
    info["wall_s"] = round(time.perf_counter() - t0, 2)
    return z, info


PROFILE_TOL = 0.01
PROFILE_MAX_FITS = 12


def _profile_side(cond, nll_free, mu_hat, bound, sgn, s0):
    """One side of the profile interval: (half-width or None, status, n_fits); cond(mu) -> nll or None.
    status: "ok", "interpolated", "bound" (no crossing before the POI bound), "failed", "not converged"."""
    reach = abs(bound - mu_hat) if np.isfinite(bound) else np.inf
    if reach <= 1e-12:
        return None, "bound", 0
    pts = [(0.0, 0.0)]
    d = float(min(max(s0, 1e-6), reach))
    for n in range(1, PROFILE_MAX_FITS + 1):
        at_bound = d >= reach
        nll = cond(mu_hat + sgn * d)
        if nll is None or not np.isfinite(nll):
            return None, "failed", n
        q = max(2.0 * (nll - nll_free), 0.0)
        pts.append((d, float(np.sqrt(q))))
        if abs(q - 1.0) <= PROFILE_TOL:
            return float(d), "ok", n
        below = [p for p in pts if p[1] < 1.0]
        above = [p for p in pts if p[1] > 1.0]
        if not above:
            if at_bound:
                return None, "bound", n
            rq = pts[-1][1]
            d = min(d / rq if rq > 0.1 else 4.0 * d, reach)
            continue
        b, a = max(below), min(above)
        d = b[0] + (1.0 - b[1]) * (a[0] - b[0]) / (a[1] - b[1])
    if [p for p in pts if p[1] > 1.0]:
        return float(d), "interpolated", PROFILE_MAX_FITS
    return None, "not converged", PROFILE_MAX_FITS


try:
    from .scoring import MU_SUPPORT, SCAN_VERSION
except Exception:  # noqa: BLE001
    MU_SUPPORT, SCAN_VERSION = (0.0, 4.0), "marginal-v1"
SCAN_K = tuple(float(k) for k in np.arange(-4.0, 6.01, 0.5))
SCAN_FIXED = (0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0)
SCAN_DEDUP = 1e-6
SCAN_MIN_STEP = 0.05


def _widened_bounds(lo_b, hi_b):
    """The POI bounds the mu = 0 conditional fit runs under: [min(lo_b, 0), max(hi_b, 0)] (mu = 0 is never clipped
    into the declared bounds; the model is scored on what it claims there)."""
    return float(min(lo_b, 0.0)), float(max(hi_b, 0.0))


def _conditional_fitters(model, data, ev, x, nll_offset=0.0, poi=None, bounds=None):
    """(cond, reset, state) for conditional fits of one workspace at fixed POI values, warm-started along a walk.
    cond(mu) -> the FULL -ln L (pyhf's -log pdf; twin: nll + ``nll_offset`` = full_nll_const) at the fit with the
    POI fixed at mu and the nuisances profiled, started from the previous call's fit (``state["x"]``), the free fit
    ``x`` as the fallback start; None when the fit fails.  reset() restarts the warm start from ``x``.  ``poi``: the
    index of the parameter fixed (default the model's POI; another needs the twin); ``bounds``: the parameters'
    bounds (default the model's).  The fixed parameter's bounds are widened to include 0 (``_widened_bounds``) on
    both engines: the twin clips a fixed POI into its bounds and pyhf's fixed_poi_fit cannot fit outside par_bounds."""
    poi = model.config.poi_index if poi is None else poi
    x = np.asarray(x, dtype=float)
    state = {"x": x.copy()}
    bounds = [tuple(map(float, b)) for b in (model.config.suggested_bounds() if bounds is None else bounds)]
    bounds[poi] = _widened_bounds(*bounds[poi])
    if ev is not None:
        d = np.asarray(data, dtype=float)
        main, aux = d[:ev.n_main], d[ev.n_main:]
        fx = ev.fixed.copy()
        fx[poi] = True

        def cond(mu):
            x0 = state["x"].copy()
            x0[poi] = mu
            alt = x.copy()
            alt[poi] = mu
            r = ev.fit(main, aux, init=x0, fixed=fx, want_sigma=False, starts=(alt,), bounds=bounds)
            if not r["valid"] or not np.isfinite(r["nll"]):
                return None
            state["x"] = np.asarray(r["x"], dtype=float)
            return float(r["nll"]) + float(nll_offset)
    elif poi != model.config.poi_index:
        raise ValueError("a conditional fit of another parameter than the model's POI needs the twin")
    else:
        def cond(mu):
            try:
                pars, tw = pyhf.infer.mle.fixed_poi_fit(float(mu), data, model, init_pars=list(map(float, state["x"])),
                                                         par_bounds=bounds, return_fitted_val=True)
            except Exception:  # noqa: BLE001
                return None
            tw = float(tw)
            if not np.isfinite(tw):
                return None
            state["x"] = np.asarray(pars, dtype=float)
            return 0.5 * tw

    def reset():
        state["x"] = x.copy()
    return cond, reset, state


def mu_scan_set(mu_hat, s, lo_b, hi_b, support=MU_SUPPORT):
    """The conditional-scan values S of a workspace: {0} u {mu_hat + k s} u SCAN_FIXED inside the support and the
    workspace's own bounds, plus that interval's ends (a declared bound inside the support is scanned up to; mu = 0
    whenever the bounds widened to min(lo_b, 0) hold it, i.e. hi_b >= 0), sorted,
    a value within SCAN_MIN_STEP x s of a smaller kept one dropped (mu = 0 always kept; e.g. mu_hat - 2.5 s landing
    4e-4 above 0 adds a fit and nothing to the interpolation).  ``s`` the Hessian width (0.3 when the fit has
    none)."""
    lo, hi = max(float(support[0]), float(lo_b)), min(float(support[1]), float(hi_b))
    s = float(s) if s and np.isfinite(s) and s > 0 else 0.3
    cand = [float(mu_hat) + k * s for k in SCAN_K] + list(SCAN_FIXED) + [lo, hi]
    mus = sorted(m for m in cand if lo - 1e-12 <= m <= hi + 1e-12)
    zero = hi_b >= 0.0 and support[0] <= 0.0 <= support[1]
    out = [0.0] if zero else []
    step = max(SCAN_DEDUP, SCAN_MIN_STEP * s)
    for m in mus:
        if not out or m - out[-1] > step:
            out.append(float(m))
    if mus and out and mus[-1] - out[-1] > SCAN_DEDUP:
        if len(out) > 1:
            out[-1] = float(mus[-1])
        else:
            out.append(float(mus[-1]))
    return out


def conditional_scan(cond, reset, mu_hat, nll_min, mus, want_x=(), state=None):
    """l(mu) at every value of ``mus``: each side walked outward from mu_hat with the warm start (reset() before a
    side, as _profile_side's caller does), a failed value skipped.  The free fit (mu_hat, nll_min) is included when
    it lies within the scanned span.  Returns (scan [[mu, nll], ...] sorted in mu, n_failed, {mu: x} for the values
    in ``want_x`` when ``state`` exposes the fitted parameters)."""
    scan, n_failed, xs = [], 0, {}
    want = [float(v) for v in want_x]
    for side in (+1, -1):
        seq = [m for m in mus if (m > mu_hat if side > 0 else m < mu_hat)]
        if side < 0:
            seq = seq[::-1]
        reset()
        for mu in seq:
            nll = cond(mu)
            if nll is None or not np.isfinite(nll):
                n_failed += 1
                continue
            scan.append([float(mu), float(nll)])
            if state is not None and any(abs(mu - w) <= SCAN_DEDUP for w in want):
                xs[float(mu)] = np.asarray(state["x"], dtype=float).copy()
    if mus and mus[0] - SCAN_DEDUP <= mu_hat <= mus[-1] + SCAN_DEDUP and nll_min is not None \
            and not any(abs(m - mu_hat) <= SCAN_DEDUP for m, _ in scan):
        scan.append([float(mu_hat), float(nll_min)])
    scan.sort()
    return scan, n_failed, xs


def mu_fit_block(model, data, res, ev=None, obs_fit=None):
    """The model's own mu measurement (scored in place of RESULT.json's), on the twin if built, else pyhf.  pyhf's free
    refit raising is the model's failure (ModelFitError); the conditional fits' failures are counted in the profile
    and the scan.
      value        best-fit mu
      sigma        profile half-width 1/2 (s- + s+); one side if the other hits the bound or fails; else
                   sigma_hess (``sigma_source``)
      sigma_hess   twin: exact autodiff Hessian; pyhf: the nominal fit's HESSE
      sigma_min    min(profile, Hessian) (the truthfulness test's yardstick)
      sigma_stat   nuisances fixed at the best fit, 1 / sqrt(d2 NLL / d mu2)
      nll_min      the full -ln L at the free fit
      mu_scan      the conditional scan [[mu, full -ln L], ...] over mu_scan_set (the single-mass density of
                   MODEL.json's own point); scan_version stamps it
      nll0, q0     the mu = 0 conditional fit ALWAYS runs, with the POI bound widened to min(lo_b, 0) on both
                   engines (mu0_outside_bounds says the declared bounds exclude 0); q0 = 2 (nll0 - nll_min)
                   clipped at 0, and 0 when mu-hat < 0
      z_obs        sqrt(q0) on the observed data"""
    import time
    t0 = time.perf_counter()
    poi = model.config.poi_index
    lo_b, hi_b = (float(b) for b in model.config.suggested_bounds()[poi])
    n_main = model.config.nmaindata
    d = np.asarray(data, dtype=float)
    out = {"engine": "jax" if ev is not None else "pyhf", "bounds": [lo_b, hi_b]}
    if ev is not None:
        main, aux = d[:n_main], d[n_main:]
        x = np.asarray(obs_fit["x"], dtype=float)
        const = float(ev.full_nll_const(main, aux))
        nll_free = float(obs_fit["nll"]) + const
        mu_hat = float(x[poi])
        sg = ev._sigma(x, main, aux)
        out["sigma_hess"] = float(sg[poi]) if sg is not None and np.isfinite(sg[poi]) and sg[poi] > 0 else None
        try:
            hmm = float(np.asarray(ev.hess(x, main, aux))[poi, poi])
            out["sigma_stat"] = float(1.0 / np.sqrt(hmm)) if hmm > 0 else None
        except Exception:  # noqa: BLE001
            out["sigma_stat"] = None
        cond, reset, _state = _conditional_fitters(model, data, ev, x, nll_offset=const)
    else:
        pars_free, twice = model_fit(pyhf.infer.mle.fit, data, model, init_pars=list(map(float, res.bestfit)),
                                     return_fitted_val=True)
        nll_free = 0.5 * float(twice)
        x = np.asarray(pars_free, dtype=float)
        mu_hat = float(x[poi])
        out["sigma_hess"] = (float(res.uncertainty[poi]) if np.isfinite(res.uncertainty[poi])
                             and res.uncertainty[poi] > 0 else None)
        h = 1e-3 * (out["sigma_hess"] or 1.0)

        def nll_at(mu):
            xx = x.copy()
            xx[poi] = mu
            return -float(np.asarray(model.logpdf(pyhf.tensorlib.astensor(xx), pyhf.tensorlib.astensor(d)))[0])
        try:
            c2 = (nll_at(mu_hat + h) - 2.0 * nll_at(mu_hat) + nll_at(mu_hat - h)) / (h * h)
            out["sigma_stat"] = float(1.0 / np.sqrt(c2)) if np.isfinite(c2) and c2 > 0 else None
        except Exception:  # noqa: BLE001
            out["sigma_stat"] = None
        cond, reset, _state = _conditional_fitters(model, data, None, x)
    out["value"] = mu_hat
    out["nll_min"] = nll_free
    s0 = out["sigma_hess"] or 0.3
    prof = {}
    for tag, sgn, bound in (("lo", -1.0, lo_b), ("hi", +1.0, hi_b)):
        reset()
        prof[tag], prof[f"status_{tag}"], prof[f"n_fits_{tag}"] = _profile_side(cond, nll_free, mu_hat, bound,
                                                                                   sgn, s0)
    out["profile"] = prof
    sides = [prof[t] for t in ("lo", "hi") if prof[f"status_{t}"] in ("ok", "interpolated") and prof[t]]
    if sides:
        out["sigma"], out["sigma_source"] = float(np.mean(sides)), ("profile" if len(sides) == 2
                                                                    else "profile_one_side")
    elif out["sigma_hess"]:
        out["sigma"], out["sigma_source"] = out["sigma_hess"], "hessian"
    else:
        out["sigma"], out["sigma_source"] = None, None
    conv = [s for s in ((float(np.mean(sides)) if sides else None), out["sigma_hess"]) if s]
    out["sigma_min"] = float(min(conv)) if conv else None
    mus = mu_scan_set(mu_hat, s0, lo_b, hi_b)
    scan, n_fail, _xs = conditional_scan(cond, reset, mu_hat, nll_free, mus)
    out["mu_scan"], out["n_scan_failed"], out["scan_version"] = scan, int(n_fail), SCAN_VERSION
    nll0 = next((v for m, v in scan if abs(m) <= SCAN_DEDUP), None)
    if nll0 is None:
        reset()
        nll0 = cond(0.0)
    out["nll0"] = nll0
    out["mu0_outside_bounds"] = not (lo_b <= 0.0 <= hi_b)
    if nll0 is None:
        q0 = None
    elif mu_hat < 0:
        q0 = 0.0
    else:
        q0 = max(2.0 * (nll0 - nll_free), 0.0)
    out["q0"] = q0
    out["z_obs"] = None if q0 is None else float(np.sqrt(q0))
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    return out


def _read_result(rj):
    """(mu poi, mH poi, self_quality, claimed Z_exp) from RESULT.json, as reported (result_fields), read as the
    contract reads output files (contract.read_output: never through a symlink); all None when absent or unreadable."""
    import os
    if not os.path.lexists(rj):
        return None, None, None, None
    try:
        from ...contract import read_output
        r = json.loads(read_output(rj))
    except (OSError, ValueError, RecursionError):
        r = None
    return result_fields(r)


def result_fields(r):
    """(mu poi, mH poi, self_quality, claimed Z_exp) of a parsed RESULT.json ``r``, as reported.  A file that is not a
    JSON object, a "pois" that is not a list and an entry whose name is not a string are not reported."""
    reported, reported_mh, self_quality, claimed_zexp = None, None, None, None
    if not isinstance(r, dict):
        return reported, reported_mh, self_quality, claimed_zexp
    entries = r.get("pois", [])
    pois = {x["name"]: x for x in (entries if isinstance(entries, list) else [])
            if isinstance(x, dict) and isinstance(x.get("name"), str)}
    reported = pois.get("mu")
    reported_mh = pois.get("mH")
    claimed_zexp = r.get("significance_exp")
    q = r.get("quality")
    if isinstance(q, dict):
        self_quality = {k: v for k, v in q.items() if isinstance(v, dict)}
    legacy = r.get("gof")
    if isinstance(legacy, dict) and legacy.get("p_value") is not None:
        self_quality = dict(self_quality or {})
        self_quality.setdefault("gof_pvalue",
                                {"value": legacy["p_value"],
                                 "note": f"legacy gof block ({legacy.get('method', '?')})"})
    return reported, reported_mh, self_quality, claimed_zexp


def finite_number(x):
    """A submission's optional number as the grader reads it (the quality block's values, significance_exp, the legacy
    gof block's p_value): a JSON number (an int or a float, never a bool) that is finite, as a float; None for anything
    else -- a string (a numeric one too: the contract asks for a number), a list, an object, null, NaN, an infinity, an
    integer beyond the float range.  Never raises."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        v = float(x)
    except OverflowError:
        return None
    return v if math.isfinite(v) else None


NOT_A_NUMBER = "not a number"
QUALITY_KEEP = 50


def _shown(x) -> str:
    """A short, bounded description of a reported value that is not a number (no repr of nested containers)."""
    if isinstance(x, dict):
        return f"a JSON object of {len(x)} entries"
    if isinstance(x, list):
        return f"a JSON array of {len(x)} entries"
    if isinstance(x, str):
        return f"a string {json.dumps(x[:40])}"
    if isinstance(x, int) and not isinstance(x, bool):
        return "an integer beyond the float range"
    return repr(x)[:40]


def _claim(given, ours, agree):
    """One reported claim against the evaluator's number: a claim that is not a finite number (finite_number) disagrees,
    {"self": None, "given", "evaluator", "agree": False, "why": NOT_A_NUMBER}, whether or not the evaluator has a number
    to compare; else None without one of the evaluator's (nothing to compare), else {"self", "evaluator", "agree":
    agree(self, evaluator)}."""
    v = finite_number(given)
    if v is None:
        return {"self": None, "given": _shown(given), "evaluator": None if ours is None else float(ours),
                "agree": False, "why": NOT_A_NUMBER}
    if ours is None:
        return None
    return {"self": v, "evaluator": float(ours), "agree": bool(agree(v, float(ours)))}


def _quality_record(self_quality) -> dict:
    """The quality block as the grader's record keeps it (bounded: RESULT.json keeps it as given): per name, at most
    QUALITY_KEEP of them, its value as a finite number or None, what was given where that is not a number, and its note
    as text of at most 300 characters."""
    out = {}
    for k, v in list(self_quality.items())[:QUALITY_KEEP]:
        val, note = v.get("value"), v.get("note")
        e = {"value": finite_number(val)}
        if val is not None and e["value"] is None:
            e["given"] = _shown(val)
        if note is not None:
            e["note"] = note[:300] if isinstance(note, str) else _shown(note)
        out[str(k)[:80]] = e
    return out


def _crosschecks(rec, self_quality, claimed_zexp, shipped=False):
    """Submission's claimed quality (gof_pvalue, coverage) and Z_exp vs the evaluator's numbers on its model.
    Scored: sbh.score.claims_rows charges every ``agree`` False (EVAL_CONTRACT section 2).  Every claim is read through
    finite_number (_claim): one that is not a finite number disagrees ("why": "not a number") and nothing here raises on
    what a submission reports.  ``shipped`` (verify.SHIPPED: a task kept on the rule it shipped with): the claims read
    with float() as its grader shipped (_crosschecks_shipped)."""
    if shipped:
        return _crosschecks_shipped(rec, self_quality, claimed_zexp)
    if self_quality is not None:
        rec["self_quality"] = _quality_record(self_quality)
        ft = rec.get("freq_toys") if isinstance(rec.get("freq_toys"), dict) else {}
        xc = {}
        sg = (self_quality.get("gof_pvalue") or {}).get("value")
        if sg is not None:
            ty = rec.get("toys") if isinstance(rec.get("toys"), dict) else {}
            ours = next((v for v in (ft.get("gof_p_glob", ft.get("gof_p_saturated")),
                                     ty.get("gof_p_glob", ty.get("gof_p_toys")),
                                     rec.get("gof_p_saturated"))
                         if v is not None and np.isfinite(v)), None)
            c = _claim(sg, ours, lambda s, e: (s < 0.05) == (e < 0.05))
            if c is not None:
                xc["gof_pvalue"] = c
        sc = (self_quality.get("coverage") or {}).get("value")
        if sc is not None:
            c = _claim(sc, ft.get("coverage_68"), lambda s, e: abs(s - e) <= 0.15)
            if c is not None:
                xc["coverage"] = c
        if xc:
            rec["quality_crosscheck"] = xc
    if claimed_zexp is not None:
        z_exp = rec.get("z_exp_asimov")
        c = _claim(claimed_zexp, z_exp if isinstance(z_exp, float) else None,
                   lambda s, e: abs(s - e) <= max(0.3, 0.15 * e))
        if c is not None:
            rec["significance_exp_crosscheck"] = c


def _crosschecks_shipped(rec, self_quality, claimed_zexp):
    """_crosschecks as the grader of a task kept on the rule it shipped with (verify.SHIPPED) shipped it: the quality
    block kept as given, each claim read with float(), so a claim float() does not read raises (the record's "verify
    error", which that task's grader stage charges to the model)."""
    if self_quality is not None:
        rec["self_quality"] = self_quality
        ft = rec.get("freq_toys") if isinstance(rec.get("freq_toys"), dict) else {}
        xc = {}
        sg = (self_quality.get("gof_pvalue") or {}).get("value")
        if sg is not None:
            ty = rec.get("toys") if isinstance(rec.get("toys"), dict) else {}
            ours = next((v for v in (ft.get("gof_p_glob", ft.get("gof_p_saturated")),
                                     ty.get("gof_p_glob", ty.get("gof_p_toys")),
                                     rec.get("gof_p_saturated"))
                         if v is not None and np.isfinite(v)), None)
            if ours is not None:
                xc["gof_pvalue"] = {"self": float(sg), "evaluator": float(ours),
                                    "agree": bool((float(sg) < 0.05) == (float(ours) < 0.05))}
        sc = (self_quality.get("coverage") or {}).get("value")
        if sc is not None and ft.get("coverage_68") is not None:
            xc["coverage"] = {"self": float(sc), "evaluator": float(ft["coverage_68"]),
                              "agree": bool(abs(float(sc) - float(ft["coverage_68"])) <= 0.15)}
        if xc:
            rec["quality_crosscheck"] = xc
    z_exp = rec.get("z_exp_asimov")
    if claimed_zexp is not None and isinstance(z_exp, float):
        rec["significance_exp_crosscheck"] = {
            "self": float(claimed_zexp), "evaluator": z_exp,
            "agree": bool(abs(float(claimed_zexp) - z_exp) <= max(0.3, 0.15 * z_exp))}

