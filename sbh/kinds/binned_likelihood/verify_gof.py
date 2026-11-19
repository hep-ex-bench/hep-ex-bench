"""The model verifier's goodness-of-fit statistics (verify.py): Poisson deviance, saturated q, toy p-values with ties,
the GOF degrees of freedom and the bound window. Evaluator-only (sbh/harbor_pack.py EVALUATOR_ONLY); verify.py re-
exports every name."""
import numpy as np
import pyhf


def _deviance_terms(exp, counts):
    """Per main bin, the Poisson deviance 2 [(nu - n) + n ln(n / nu)] (0 log 0 = 0)."""
    exp = np.asarray(exp, dtype=float)
    m = np.asarray(counts, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(m > 0, m * np.log(m / np.clip(exp, 1e-12, None)), 0.0)
    return 2.0 * ((exp - m) + term)


def _saturated_q(exp, counts):
    """Poisson deviance of the main bins: pyhf's saturated statistic (on jax, a diagnostic next to combine's q)."""
    return float(np.sum(_deviance_terms(exp, counts)))


def _twin_gof(ev, x, main, aux, window=None):
    """Twin GOF statistics at x: ``saturated`` (combine's q: deviance + constraint pulls), ``saturated_deviance``
    (diagnostic) and ``window`` (deviance over the window bins) when a mask is given."""
    terms = _deviance_terms(ev.expected_main(x), main)
    out = {"saturated": ev.q_saturated(x, main, aux), "saturated_deviance": float(np.sum(terms))}
    if window is not None and window.any():
        out["window"] = float(np.sum(terms[window]))
    return out


def _pyhf_gof(model, x, main, window=None):
    """pyhf GOF statistics at x: deviance and window deviance."""
    return _gof_stats(model.expected_actualdata(pyhf.tensorlib.astensor(x)), main, window)


GOF_TIE_EPS = 1e-3


def _toy_p(q_toys, q_obs, eps=0.0):
    """Fraction of toys with q_toy >= q_obs - eps (None without toys); combine leg.  Scored stages use
    _pooled_p's (k + 1) / (N + 1)."""
    q = np.asarray(q_toys, dtype=float)
    return float(np.mean(q >= q_obs - eps)) if len(q) else None


def _pooled_p(q_toys, q_obs, eps=GOF_TIE_EPS):
    """Toy p of every item in the pool {observed} + toys, itself included: element 0 = (k + 1) / (N + 1),
    k = #{toys with q >= q_obs - eps}; element j the same for toy j (exchangeable).  Ties within eps count."""
    pool = np.concatenate([[float(q_obs)], np.asarray(q_toys, dtype=float)])
    srt = np.sort(pool)
    k = len(pool) - np.searchsorted(srt, pool - eps, side="left")
    return k / len(pool)


def _gof_dof(model, points, rates=None):
    """Saturated-statistic dof: main bins minus the rank of d(main rates)/d(theta) over the free unconstrained
    parameters (max over ``points``).  Constrained nuisances pay their constraint and cost no dof (else every
    staterror model would be degenerate).  <= 0: the model fits any counts exactly and there is no p (e.g. one
    bin, mu x signal + background -> 1 - 1 = 0).  ``rates``: x -> the main rates (default pyhf's; a POI-scaled
    model's are its twin's)."""
    cfg = model.config
    fixed = cfg.suggested_fixed()
    cols = []
    for name in cfg.par_order:
        if getattr(cfg.param_set(name), "constrained", True):
            continue
        sl = cfg.par_slice(name)
        cols += [i for i in range(sl.start, sl.stop) if not fixed[i]]
    n_main = int(cfg.nmaindata)
    if not cols:
        return n_main
    rank = 0

    def nu(x):
        if rates is not None:
            return np.asarray(rates(x), dtype=float)
        return np.asarray(model.expected_actualdata(pyhf.tensorlib.astensor(x)), dtype=float)

    for pars in points:
        x = np.asarray(pars, dtype=float)
        base = nu(x)
        J = np.empty((n_main, len(cols)))
        for k, i in enumerate(cols):
            h = 1e-3 * max(1.0, abs(x[i]))
            xp = x.copy()
            xp[i] += h
            J[:, k] = (nu(xp) - base) / h
        if not np.isfinite(J).all():
            continue
        s = np.linalg.svd(J, compute_uv=False)
        if len(s) and s[0] > 0:
            rank = max(rank, int(np.sum(s > 1e-8 * s[0])))
    return n_main - rank


def _gof_undefined(dof):
    """Record note for dof <= 0."""
    return (f"saturated statistic has {dof} degrees of freedom (main bins <= free unconstrained "
            "rate parameters): q is the minimizer's residual, no toy p")


def _gof_stats(exp, counts, window=None):
    """Saturated deviance of counts vs ``exp`` (toy-calibrated only) and its window share.  No clipping of
    ``exp``, so it matches toy_ensembles on negative-rate fits."""
    terms = _deviance_terms(exp, counts)
    out = {"saturated": float(np.sum(terms))}
    if window is not None and window.any():
        out["window"] = float(np.sum(terms[window]))
    return out


def _bound_window(lo, hi, x=0.0):
    """'On a bound' tolerance 1e-4 (hi - lo), as the twin's active set (Migrad leaves a bounded POI ~1e-5 off)."""
    return 1e-4 * (hi - lo) if np.isfinite(hi - lo) else 1e-4 * max(abs(x), 1.0)

