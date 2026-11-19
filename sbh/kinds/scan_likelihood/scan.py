"""The interval form of a POI entry (the contract's result part states it) and the reading of a profile scan.

A POI entry is {"name", "value", "err_minus", "err_plus"} inside the POI's range (interval_problems), the scanned
POI's optionally with its 68 % set in several pieces (pieces_problems).  A scan is {"points": [[x, q], ...]}, q =
-2 DeltalnL profiled over everything else (the grader's, verify.py; the self-check's refit, scaling.py): points
reads it, pieces_of and interval_of give its q <= 1 pieces.
"""
from __future__ import annotations

import bisect
import math
from typing import Any, List, Optional, Tuple

from ...contract import _finite

INTERVAL_FIELDS = ("value", "err_minus", "err_plus")
SCAN_MIN_POINTS = 20


def _at(x: float, b: float) -> bool:
    """x sits on the bound b (floating-point slack 1e-9 relative)."""
    return abs(x - b) <= 1e-9 * max(1.0, abs(b))


def interval_problems(p: dict, lo: Optional[float], hi: Optional[float]) -> List[str]:
    """One POI entry: finite value, err_minus, err_plus; [value - err_minus, value + err_plus] inside [lo, hi];
    err_minus > 0 except 0 when the value sits on the lower bound, err_plus > 0 except 0 on the upper bound;
    optional stat / syst, each {"err_minus": f >= 0, "err_plus": f >= 0}."""
    n = p["name"]
    errs = []
    for f in INTERVAL_FIELDS:
        if f not in p:
            errs.append(f"poi {n}: missing '{f}'")
        elif not _finite(p[f]):
            errs.append(f"poi {n}: '{f}' is not a finite number")
    if errs:
        return errs
    v, em, ep = float(p["value"]), float(p["err_minus"]), float(p["err_plus"])
    if lo is not None and v < lo and not _at(v, lo):
        errs.append(f"poi {n}: value {v:g} below the lower bound {lo:g}")
    if hi is not None and v > hi and not _at(v, hi):
        errs.append(f"poi {n}: value {v:g} above the upper bound {hi:g}")
    if em < 0 or ep < 0:
        errs.append(f"poi {n}: negative uncertainty")
    if ep <= 0 and not (hi is not None and _at(v, hi)):
        errs.append(f"poi {n}: err_plus must be > 0")
    if em <= 0 and not (lo is not None and _at(v, lo)):
        errs.append(f"poi {n}: err_minus must be > 0"
                    + (f" (0 only when value = the lower bound {lo:g})" if lo is not None else ""))
    if lo is not None and em > 0 and v - em < lo and not _at(v - em, lo):
        errs.append(f"poi {n}: value - err_minus = {v - em:g} below the lower bound {lo:g}")
    if hi is not None and ep > 0 and v + ep > hi and not _at(v + ep, hi):
        errs.append(f"poi {n}: value + err_plus = {v + ep:g} above the upper bound {hi:g}")
    for comp in ("stat", "syst"):
        if comp not in p:
            continue
        c = p[comp]
        if not (isinstance(c, dict) and all(_finite(c.get(k)) and c[k] >= 0 for k in ("err_minus", "err_plus"))):
            errs.append(f"poi {n}: '{comp}' must be {{\"err_minus\": f >= 0, \"err_plus\": f >= 0}}")
    return errs


def interval(p) -> Optional[Tuple[float, float, float]]:
    """(value, err_minus, err_plus) of a POI entry, or None when it is not a usable interval."""
    if not isinstance(p, dict):
        return None
    try:
        v, em, ep = float(p["value"]), float(p["err_minus"]), float(p["err_plus"])
    except (KeyError, TypeError, ValueError):
        return None
    ok = all(math.isfinite(t) for t in (v, em, ep)) and em >= 0 and ep >= 0 and em + ep > 0
    return (v, em, ep) if ok else None


def side_sigma(v: float, em: float, ep: float, truth: Optional[float]) -> float:
    """The error for a pull against ``truth``: err_plus when the truth lies above the value, else err_minus (the
    other where that one is 0, at a boundary); without a truth their mean."""
    if truth is None:
        return 0.5 * (em + ep)
    s = ep if truth > v else em
    return s if s > 0 else max(em, ep)


def points(scan: Any, lower: Optional[float] = None, upper: Optional[float] = None):
    """(xs, qs) of a scan, sorted by x: at least SCAN_MIN_POINTS pairs of finite numbers, q >= 0, the x distinct and
    inside [lower, upper]; or ValueError with the reason."""
    if not isinstance(scan, dict):
        raise ValueError("not a JSON object")
    pts = scan.get("points")
    if not isinstance(pts, list):
        raise ValueError("'points' missing or not a list")
    if len(pts) < SCAN_MIN_POINTS:
        raise ValueError(f"{len(pts)} points; at least {SCAN_MIN_POINTS} required")
    rows = []
    for i, pt in enumerate(pts):
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2 and _finite(pt[0]) and _finite(pt[1])):
            raise ValueError(f"points[{i}] is not a pair of finite numbers [x, q]")
        x, q = float(pt[0]), float(pt[1])
        if q < 0:
            raise ValueError(f"points[{i}]: q = {q:g} < 0 (q is -2 DeltalnL from the global minimum)")
        if lower is not None and x < lower and not _at(x, lower):
            raise ValueError(f"points[{i}]: x = {x:g} below the lower bound {lower:g}")
        if upper is not None and x > upper and not _at(x, upper):
            raise ValueError(f"points[{i}]: x = {x:g} above the upper bound {upper:g}")
        rows.append((x, q))
    rows.sort()
    xs = [r[0] for r in rows]
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError("the x values are not distinct")
    return xs, [r[1] for r in rows]


def local_step(xs: List[float], x: float) -> float:
    """The local grid step at x: the spacing of the two points bracketing x (on a point the larger of its two
    spacings; outside the grid the spacing at that end)."""
    if x <= xs[0]:
        return xs[1] - xs[0]
    if x >= xs[-1]:
        return xs[-1] - xs[-2]
    j = bisect.bisect_left(xs, x)
    if xs[j] == x:
        return max(xs[j] - xs[j - 1], (xs[j + 1] - xs[j]) if j + 1 < len(xs) else 0.0)
    return xs[j] - xs[j - 1]


def pieces_of(xs: List[float], qs: List[float], lower: Optional[float] = None,
              upper: Optional[float] = None) -> List[dict]:
    """Every q <= 1 piece of a sorted scan (linear between adjacent points), in x order: a maximal run of points with
    q <= 1, {"i_min", "x_min", "q_min" (its lowest point), "lo", "hi", "lo_step", "hi_step", "lo_boundary",
    "hi_boundary"}.  hi: the q = 1 crossing above the run, or ``upper`` when the run reaches a point on the upper bound
    (hi_boundary), else None; lo likewise below with ``lower``.  *_step: the bracketing pair's spacing."""
    n = len(xs)

    def cross(a, b):
        return xs[a] + (1.0 - qs[a]) / (qs[b] - qs[a]) * (xs[b] - xs[a])

    out, i = [], 0
    while i < n:
        if qs[i] > 1.0:
            i += 1
            continue
        j = i
        while j + 1 < n and qs[j + 1] <= 1.0:
            j += 1
        im = min(range(i, j + 1), key=lambda k: (qs[k], k))
        p = {"i_min": im, "x_min": xs[im], "q_min": qs[im], "lo": None, "hi": None, "lo_step": None,
             "hi_step": None, "lo_boundary": False, "hi_boundary": False}
        if j + 1 < n:
            p["hi"], p["hi_step"] = cross(j, j + 1), xs[j + 1] - xs[j]
        elif upper is not None and n > 1 and _at(xs[-1], upper):
            p.update(hi=float(upper), hi_step=xs[-1] - xs[-2], hi_boundary=True)
        if i > 0:
            p["lo"], p["lo_step"] = cross(i, i - 1), xs[i] - xs[i - 1]
        elif lower is not None and _at(xs[0], lower):
            p.update(lo=float(lower), lo_step=xs[1] - xs[0], lo_boundary=True)
        out.append(p)
        i = j + 1
    return out


def interval_of(xs: List[float], qs: List[float], lower: Optional[float] = None,
                upper: Optional[float] = None) -> dict:
    """The q <= 1 interval of a sorted scan around its minimum: the piece of pieces_of holding the scan's lowest point
    (the first of equal lowest q).  When that point's own q > 1 there is none: the same keys with "lo" and "hi" None,
    "i_min", "x_min", "q_min" the lowest point's."""
    im = min(range(len(xs)), key=lambda i: (qs[i], i))
    return next((p for p in pieces_of(xs, qs, lower, upper) if p["i_min"] == im),
                {"i_min": im, "x_min": xs[im], "q_min": qs[im], "lo": None, "hi": None, "lo_step": None,
                 "hi_step": None, "lo_boundary": False, "hi_boundary": False})


def pieces_problems(p: dict, lo: Optional[float], hi: Optional[float]) -> List[str]:
    """A POI entry's optional "intervals", its 68 % set in several pieces (after interval_problems passed): a non-empty
    list of [lo, hi] pairs of finite numbers with lo < hi, sorted and disjoint, inside the range [lo, hi]; the piece
    holding value is [value - err_minus, value + err_plus]."""
    n, iv = p["name"], p["intervals"]
    if not (isinstance(iv, list) and iv and all(isinstance(x, list) and len(x) == 2 and all(_finite(t) for t in x)
                                                 for x in iv)):
        return [f"poi {n}: 'intervals' must be a non-empty list of [lo, hi] pairs of finite numbers"]
    a = [(float(x[0]), float(x[1])) for x in iv]
    errs = []
    if any(l0 >= h0 for l0, h0 in a) or any(h0 >= l1 for (_, h0), (l1, _) in zip(a, a[1:])):
        errs.append(f"poi {n}: 'intervals' must be sorted, disjoint pieces [lo, hi] with lo < hi")
    if (lo is not None and a[0][0] < lo and not _at(a[0][0], lo)) or (hi is not None and a[-1][1] > hi
                                                                     and not _at(a[-1][1], hi)):
        errs.append(f"poi {n}: 'intervals' must lie inside the quantity's range")
    v, em, ep = float(p["value"]), float(p["err_minus"]), float(p["err_plus"])
    own = [(l0, h0) for l0, h0 in a if l0 <= v <= h0]
    if not (len(own) == 1 and _at(own[0][0], v - em) and _at(own[0][1], v + ep)):
        errs.append(f"poi {n}: the piece of 'intervals' holding value must be [value - err_minus, value + err_plus] "
                    f"= [{v - em:g}, {v + ep:g}]")
    return errs
