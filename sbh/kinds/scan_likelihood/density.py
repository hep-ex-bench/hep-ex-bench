"""The densities the scan_likelihood kind scores (evaluator-only, imported where used): the scanned POI's density
from its profile scan, the two-piece ("split") Gaussian of any other interval, their log-losses and baselines.

The scanned POI (the task's first; the grader scans its profile q = -2 DeltalnL, verify.py) is scored on the scan's
density

    p(x) = exp(-q(x) / 2) / Z  on the POI's physical range,

q through the binned_likelihood kind's component_nll (PCHIP of the signed root about the scan's minimum, the end-knot
vertex rule, quadratic continuation beyond the scanned range; scan_density).  The case loss is -ln p(truth) - ln
sqrt(2 pi) and the baseline H[p_ref] - ln sqrt(2 pi), H the mean -ln p of the closure reference's scan density on the
dataset (closure.json reference_pois[seed][poi]["entropy"]; sbh.score.forecast_case_loss).  For a parabolic scan p is
the Gaussian and the case is the core's ln sigma + pull^2 / 2 against ln sigma_ref + 1/2.

Any other POI reported as (value m, err_minus s_m, err_plus s_p) is read as

    p(x) = 2 / (sqrt(2 pi) (s_m + s_p)) exp(-(x - m)^2 / (2 s^2)),   s = s_p if x > m else s_m,

truncated to the POI's physical range and renormalised by Z, its mass there (the normal CDF on each piece).  The
case loss is -ln p(truth) - ln sqrt(2 pi) (loss), which for s_m = s_p = sigma and no bound is exactly the Gaussian
ln sigma + pull^2 / 2 of the core's _v2_loss.  The baseline is the expected loss of a calibrated forecast at the
reference's (m, s_m, s_p) (expected_loss: the truncated density's H minus ln sqrt(2 pi)), which for the
symmetric, unbounded case is ln sigma + 1/2, the core's baseline.  The fallback forecast of every POI, the scanned
one included, is the task's (center, width) as a symmetric interval truncated to the range.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

LN_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _Phi(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _zphi(z: float) -> float:
    """z * phi(z), 0 at +-inf."""
    return 0.0 if math.isinf(z) else z * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _parts(m: float, sm: float, sp: float, lo: Optional[float], hi: Optional[float]) -> List[tuple]:
    """[(mass, E[z^2])] of the two pieces on [lo, hi], z = (x - m) / s on each piece: mass 2 s / (s_m + s_p)
    (Phi(b) - Phi(a)) and the truncated normal's E[z^2] = 1 + (a phi(a) - b phi(b)) / (Phi(b) - Phi(a)) over the
    piece's [a, b] in z."""
    lo = -math.inf if lo is None else float(lo)
    hi = math.inf if hi is None else float(hi)
    out = []
    for s, x0, x1 in ((sm, lo, min(hi, m)), (sp, max(lo, m), hi)):
        if s <= 0 or x1 <= x0:
            continue
        a, b = (x0 - m) / s, (x1 - m) / s
        d = _Phi(b) - _Phi(a)
        if d <= 0:
            continue
        out.append((2.0 * s / (sm + sp) * d, 1.0 + (_zphi(a) - _zphi(b)) / d))
    return out


def loss(x: float, m: float, sm: float, sp: float, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    """-ln p(x) - ln sqrt(2 pi) of the two-piece Gaussian (m, s_m, s_p) truncated to [lo, hi]:
    ln((s_m + s_p) / 2) + z^2 / 2 + ln Z.  inf outside the bounds or where the density vanishes."""
    if (lo is not None and x < lo) or (hi is not None and x > hi) or sm + sp <= 0:
        return math.inf
    s = sp if x > m else sm
    if s <= 0:
        if x != m:
            return math.inf
        z = 0.0
    else:
        z = (x - m) / s
    mass = sum(w for w, _ in _parts(m, sm, sp, lo, hi))
    if mass <= 0:
        return math.inf
    return math.log(0.5 * (sm + sp)) + 0.5 * z * z + math.log(mass)


def expected_loss(m: float, sm: float, sp: float, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    """E[loss(x)] for x drawn from the same truncated density (a calibrated forecast at that precision):
    ln((s_m + s_p) / 2) + ln Z + 1/2 sum_k (mass_k / Z) E_k[z^2].  Symmetric and unbounded: ln sigma + 1/2."""
    parts = _parts(m, sm, sp, lo, hi)
    mass = sum(w for w, _ in parts)
    if mass <= 0 or sm + sp <= 0:
        return math.inf
    return math.log(0.5 * (sm + sp)) + math.log(mass) + 0.5 * sum(w / mass * e2 for w, e2 in parts)


class ScanForecast(dict):
    """A scan density (scan_density) as scoring.score_inputs puts it on the scanned POI's entry ("forecast").
    scoring.case_loss and the closure's baseline read a density only of this type: a RESULT.json entry is plain JSON,
    so a "forecast" key a submission writes into its own entry is never read as a scan's density."""


SCAN_AXIS_INTERVALS = 20000
SCAN_LOCAL_HALF_WIDTH = 6.0
SCAN_LOCAL_STEP = 0.02


def scan_density(scan, lo: float, hi: float) -> dict:
    """The scanned POI's density from a scan (scan.points' form): p(x) = exp(-q(x) / 2) / Z on the POI's range
    [lo, hi] (both ends: the support), q read through sbh.kinds.binned_likelihood.scoring.component_nll on l = q / 2
    (PCHIP of the signed root about the scan's minimum, the end-knot vertex rule, quadratic continuation beyond the
    scanned range).  Normalised by the trapezoid on the axis (uniform
    SCAN_AXIS_INTERVALS over the support, refined over +-SCAN_LOCAL_HALF_WIDTH half-widths of the scan's q <= 1
    interval about its minimum, and the scan's own points).  {"status": "ok", "mu_axis", "log_p" (the keys
    log_density_at and cdf_at read), "log_z", "support", "mean", "sigma", "mode", "entropy", "p_lt_0p1"}, or
    {"status": "failed", "reason"} for a scan that gives none."""
    from ..binned_likelihood.scoring import _log_trapz, component_nll, density_summary
    from . import scan as SC
    try:
        xs, qs = SC.points(scan, lo, hi)
    except ValueError as e:
        return {"status": "failed", "reason": f"the scan: {e}"}
    point = {"mu_scan": [[x, 0.5 * q] for x, q in zip(xs, qs)], "bounds": [lo, hi]}
    span = xs[-1] - xs[0]
    a, b = float(lo), float(hi)
    iv = SC.interval_of(xs, qs, lo, hi)
    halves = [h for h in ((iv["x_min"] - iv["lo"]) if iv["lo"] is not None else None,
                          (iv["hi"] - iv["x_min"]) if iv["hi"] is not None else None) if h is not None and h > 0]
    s = min(halves) if halves else span / len(xs)
    ks = np.arange(-SCAN_LOCAL_HALF_WIDTH, SCAN_LOCAL_HALF_WIDTH + 1e-9, SCAN_LOCAL_STEP)
    axis = np.unique(np.round(np.concatenate([np.linspace(a, b, SCAN_AXIS_INTERVALS + 1),
                                              np.clip(iv["x_min"] + ks * s, a, b), np.clip(xs, a, b)]), 10))
    log_num = -component_nll(point, axis)
    log_z = _log_trapz(log_num, axis)
    if not np.isfinite(log_z):
        return {"status": "failed", "reason": "the scan's density does not normalise on the range"}
    log_p = log_num - log_z
    return {"status": "ok", "reason": None, "mu_axis": axis, "log_p": log_p, "log_z": float(log_z),
            "support": [float(a), float(b)], **density_summary(axis, log_p)}


def reference_entry(entry: dict, scored: Optional[dict]) -> dict:
    """Kind.reference_poi's addition for the scanned POI: H (the mean -ln p) of the reference's scan density on the
    dataset (None where its scan gives none: the dataset's case is then excluded), read off the scored entry's
    ScanForecast; any other entry is returned as it is."""
    fc = scored.get("forecast") if isinstance(scored, dict) else None
    if not isinstance(fc, ScanForecast):
        return entry
    h = fc.get("entropy") if fc.get("status") == "ok" else None
    return {**entry, "entropy": float(h) if h is not None and math.isfinite(h) else None}


def interval_case(p, truth, lo, hi, fb_loss: float, ref, ref_sigma) -> dict:
    """The two-piece case of an interval entry ``p`` (None: no usable answer, ``fb_loss`` its loss) truncated to the
    POI's range [lo, hi], against the calibrated two-piece forecast at the reference's interval ``ref`` on the dataset
    (a reference without one, or the scalar ``ref_sigma``: the symmetric ln sigma + 1/2).  The case's sigma is the mean
    half-width, the scale of its precision piece (sbh.score._loss_pieces); its signed pull mw_pull (the bias term's)
    is (value - truth) over the error on the truth's side; its fields keep both errors, the reference's, and |pull|."""
    from .scan import interval, side_sigma
    a = interval(p) if p is not None else None
    good = a is not None
    case = loss(truth, *a, lo, hi) if good else fb_loss
    ra = interval(ref) if ref else None
    if ra is not None:
        ref_loss, src = expected_loss(*ra, lo, hi), "seed"
    elif ref and float(ref.get("sigma", 0.0)) > 0:
        ref_loss, src = math.log(float(ref["sigma"])) + 0.5, "seed"
    elif ref_sigma > 0:
        ref_loss, src = math.log(ref_sigma) + 0.5, "scalar"
    else:
        ref_loss, src = None, None
    half = 0.5 * (a[1] + a[2]) if good else None
    fields = {"sigma_minus": a[1] if good else None, "sigma_plus": a[2] if good else None, "sigma": half,
              "abs_pull": abs(truth - a[0]) / side_sigma(*a, truth) if good else None,
              "ref_sigma_minus": ra[1] if ra else None, "ref_sigma_plus": ra[2] if ra else None}
    return {"good": good, "loss": case, "fb_loss": fb_loss, "ref_loss": ref_loss, "ref_source": src,
            "value": a[0] if good else None, "sigma": half,
            "mw_pull": (a[0] - truth) / side_sigma(*a, truth) if good else None, "fields": fields}
