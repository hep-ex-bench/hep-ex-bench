"""The evaluator's refit of a declared measurement (MEASUREMENT.json) -- EVALUATOR-ONLY (sbh/harbor_pack.py
EVALUATOR_ONLY): on the pairs measurement.py checks, each served correction refitted with its own form and tested
for goodness of fit (toys); the refit comparison is display only (the analysis's estimator need not be this one, and
a correction constant along what its measurement does not resolve is allowed).  verify() runs it (post.py).
"""
from __future__ import annotations

import copy
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...budget import BudgetExceeded
from ...contract import read_output
from . import VERIFY_FILE  # noqa: F401
from .measurement import (CPUS, FILE, SAMPLES, _n_bins, _objects_float, bin_index, check_pairs_efficiency,
                          histogram_cells, lineshape_objects, pair_variables, parse, pool_imap, read,
                          read_measurement_pairs, recount_efficiency, recount_lineshape)
from .measurement import undeclared as left_out

VERIFY_RULE = ("declared measurement (refit; the refit comparison for display, never a failed check: a "
               "correction its measurement does not determine stands as reported; toys, Sidak over a "
               "lineshape's tested histograms; an event's copies one unit, in the thresholds too; the measurement "
               "checks and the Z-peak comparison each within its own budget, what does not finish fails; a number's "
               "response from the objects its leaf reaches; a lineshape refit with more live numbers than fitted "
               "moments refits in the moments' space, one scale-free rule for its step, covariance and free "
               "directions, free below 1e-10; the efficiency fit's profile score where the simulation's efficiency "
               "sits at its bound above the ratio 1, up to 20 scoring steps, a step reversing within rounding "
               "settled, its convergence recorded)")
NOT_IN_BUDGET = "was not verified within the evaluation's budget"


N_TOYS = 200
GOF_TIE_EPS = 1e-3
GOF_MIN_CELL = 10.0
N_REPLICAS = 1
EFF_EPS = 1e-6
RESOLUTION_EFFECT = {"res_pt": "pt", "res_eta": "eta", "res_phi": "phi"}
SOFT = 0.02


def _support(var: str) -> Tuple[float, float]:
    """The range of a pair variable's values."""
    if var == "delta_phi":
        return 0.0, math.pi
    if var.endswith("_phi"):
        return -math.pi, math.pi
    if var.endswith("_charge") or var in ("first_eta", "second_eta"):
        return -math.inf, math.inf
    return 0.0, math.inf


def _step(v: np.ndarray, e, d, last=False) -> np.ndarray:
    """Soft step at edge e: 0 below e - d, 1 above e + d, the quintic smoothstep (C2) between; d = 0: v >= e (v > e
    at a binning's last edge, which counts in the last bin).  e, d, last may be arrays aligned with v."""
    v = np.asarray(v, dtype=float)
    e, d = np.broadcast_to(e, v.shape), np.broadcast_to(d, v.shape)
    out = np.where(np.broadcast_to(last, v.shape), v > e, v >= e).astype(float)
    near = np.abs(v - e) < d
    if near.any():
        u = (v[near] - e[near]) / (2.0 * d[near]) + 0.5
        out[near] = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
    return out


def _soft_bins(v: np.ndarray, edges, var: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft bin memberships (SOFT): (value index, bin, weight > 0) for a value's own bin and a neighbour's within the
    edge's soft zone; a partition of unity inside the edges."""
    e = np.asarray(edges, dtype=float)
    nb = len(e) - 1
    width = np.diff(e)
    d = np.empty(len(e))
    d[0], d[-1] = SOFT * width[0], SOFT * width[-1]
    d[1:-1] = SOFT * np.minimum(width[:-1], width[1:])
    lo, hi = _support(var)
    d[(e <= lo) | (e >= hi)] = 0.0
    c = e.copy()
    c[0], c[-1] = e[0] + d[0], e[-1] - d[-1]
    b = np.clip(np.searchsorted(e, v, side="right") - 1, 0, nb - 1)
    idx, bins, wts = [np.arange(len(v))], [b], [_step(v, c[b], d[b]) - _step(v, c[b + 1], d[b + 1], last=b + 1 == nb)]
    for off, near in ((-1, (b > 0) & (np.abs(v - e[b]) < d[b])), (1, (b + 1 < nb) & (np.abs(v - e[b + 1]) < d[b + 1]))):
        i = np.flatnonzero(near)
        if len(i):
            j = b[i] + off
            idx.append(i)
            bins.append(j)
            wts.append(_step(v[i], c[j], d[j]) - _step(v[i], c[j + 1], d[j + 1], last=j + 1 == nb))
    i, k, w = np.concatenate(idx), np.concatenate(bins), np.concatenate(wts)
    keep = w > 0
    return i[keep], k[keep], w[keep]


def soft_cells(h: dict, sel: dict, x: Dict[str, np.ndarray], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft count in a histogram's categories: (pair, category, weight > 0), the product of the soft memberships."""
    lo, hi = _support(h["variable"])
    e = np.asarray(h["edges"], dtype=float)
    xv = x[h["variable"]]
    d0 = 0.0 if e[0] <= lo else SOFT * (e[-1] - e[0])
    d1 = 0.0 if e[-1] >= hi else SOFT * (e[-1] - e[0])
    m = _step(xv, e[0] + d0, d0) - _step(xv, e[-1] - d1, d1, last=True)
    dp, de = SOFT * sel["pt_min"], SOFT * sel["abs_eta_max"]
    for role in ("first", "second"):
        m = m * _step(x[f"{role}_pt"], sel["pt_min"] + dp, dp, last=True) \
            * (1.0 - _step(x[f"{role}_abs_eta"], sel["abs_eta_max"] - de, de))
    rows = np.flatnonzero(m > 0)
    cat, m = np.zeros(len(rows), dtype=np.int64), m[rows]
    for var, edges in h["categories"]:
        i, k, w = _soft_bins(np.asarray(x[var], dtype=float)[rows], edges, var)
        rows, cat, m = rows[i], cat[i] * (len(edges) - 1) + k, m[i] * w
    return rows, cat, m


def split_histograms(pm: dict, data: dict, depth: int = 1) -> Tuple[List[dict], bool]:
    """As split_bins for a lineshape: each variable bin split at its data median (empty: middle), ``depth`` times;
    and whether _admissible (past the first split)."""
    x = pair_variables(data["first"], data["second"])
    n = len(x["mass"])
    out, ok_all = [], True
    for h in pm["histograms"]:
        e = h["edges"]
        for _ in range(depth):
            c = histogram_cells({**h, "edges": e}, pm["selection"], x, n)
            ok = c >= 0
            mid = _median_split(np.asarray(x[h["variable"]], dtype=float), np.where(ok, c % (len(e) - 1), -1),
                                len(e) - 1, ok)
            mid = np.where((mid > e[:-1]) & (mid < e[1:]), mid, 0.5 * (e[:-1] + e[1:]))
            e = np.sort(np.concatenate([e, mid]))
        if depth > 1:
            c = histogram_cells({**h, "edges": e}, pm["selection"], x, n)
            ok_all &= _admissible(c, _n_bins(h["categories"]) * (len(e) - 1), data.get("unit"))
        out.append({**h, "edges": e, "data": np.zeros((_n_bins(h["categories"]), len(e) - 1))})
    return out, ok_all


def _bits(v: np.ndarray) -> np.ndarray:
    """An array's values as their bit patterns, widened to uint64."""
    v = np.ascontiguousarray(v)
    if v.dtype == bool:
        return v.astype(np.uint64)
    return v.view(np.dtype(f"u{v.dtype.itemsize}")).astype(np.uint64)


def unverified(dataset_dir, why: str) -> dict:
    """verify's result for a dataset whose budget ran out before its measurements were told apart: every requested
    (name, flavour) failed with `why`."""
    from . import outputs as C
    request = json.loads((Path(dataset_dir) / "request.json").read_text())
    return {"measurements": {}, "seconds": None,
            "corrections": {f"{n}|{fl}": {"status": "failed", "why": why, "measurement": None}
                            for n, e in sorted(C.requested(request).items())
                            for fl in sorted(((e or {}).get("domain") or {}))}}


def verify_rule(task) -> str:
    """VERIFY_RULE plus whether the task refits (measurement_refit) and gates (measurement_required)."""
    return (VERIFY_RULE + ("" if getattr(task, "measurement_refit", True) else "; the checks only, no refit")
            + ("" if getattr(task, "measurement_required", True) else "; the declaration optional, no gate"))


def current(row: Optional[dict], task, run_dir) -> bool:
    """Whether a VERIFY_FILE matches verify_rule, the task's closure comparison's rule (comparison.rule), the budget
    factors and the run's reference wall; otherwise the post stage redoes it and the scorer leaves it pending."""
    from .comparison import rule
    from .post import budget_factors, reference_wall_of
    if not (row and row.get("rule") == verify_rule(task) and row.get("peak_rule") == rule(task)):
        return False
    b = row.get("budget") or {}
    return ((b.get("verify_y"), b.get("peak_y")) == budget_factors(task, run_dir)
            and b.get("reference_wall_s") == reference_wall_of(run_dir, task))


def copies(objs: dict) -> np.ndarray:
    """Each pair's statistical unit: the first pair whose objects equal its own in every value read (kinematics,
    charge, selection branches), else its own index.  Copies of an event fluctuate together, so bootstrap, weights
    and GOF count them as one unit (no copies: the identity).  64-bit hash, then checked value by value; compared
    role by role (a tag-and-probe pair and its swap are different measurements)."""
    cols = [objs[r][k] for r in ("first", "second") for k in sorted(objs[r])]
    n = len(cols[0]) if cols else 0
    own = np.arange(n, dtype=np.int64)
    if n < 2:
        return own
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for c in cols:
            h = (h ^ _bits(c)) * np.uint64(0x9E3779B97F4A7C15)
            h ^= h >> np.uint64(31)
    order = np.argsort(h, kind="stable")
    hs = h[order]
    start = np.r_[True, hs[1:] != hs[:-1]]
    first = order[np.flatnonzero(start)][np.cumsum(start) - 1]
    unit = np.empty(n, dtype=np.int64)
    unit[order] = first
    same = np.ones(n, dtype=bool)
    for c in cols:
        same &= np.asarray(c)[unit] == np.asarray(c)
    return np.where(same, unit, own)



class _LeafPoints:
    """The points of an evaluation grouped by the leaf of the correction each reaches (_Form.leaf_points)."""

    def __init__(self, ids: np.ndarray, leaf_of: np.ndarray):
        self.order = np.argsort(ids, kind="stable")
        self.sorted = ids[self.order]
        self.leaf_of = leaf_of

    def of_slot(self, slot: int) -> np.ndarray:
        """The points (their indices, ascending) where the number's leaf is evaluated."""
        lo, hi = np.searchsorted(self.sorted, [self.leaf_of[slot], self.leaf_of[slot] + 1])
        return self.order[lo:hi]


class _Form:
    """A correction with its free numbers (numeric_slots) as parameters -- a resolution's as x|x|, smooth through
    zero -- at simulated kinematics clamped to its domain, a variation input at "nominal".  A number moves it only
    where its leaf is evaluated (outputs.leaves), so moved() evaluates moved leaves alone at their points.
    Thread-safe: each thread sets the numbers in its own copy."""

    def __init__(self, corr: dict, flavour: str, domain: dict, signed: bool = False):
        from . import outputs as C
        self.C, self.flavour, self.signed = C, flavour, signed
        self.pt_lo = float(domain["pt_min"]) * (1.0 + 1e-6) + 1e-6
        self.eta_hi = float(domain["abs_eta_max"]) * (1.0 - 1e-6)
        self.var = "nominal" if "variation" in C._input_names(corr) else None
        self.work = C.normalized(corr)
        C.evaluator(self.work, checked=False)
        self.slots = C.numeric_slots(self.work, nominal=self.var is not None)
        self.theta0 = np.array([float(c[k]) for c, k in self.slots], dtype=float)
        self.active = np.arange(len(self.slots))
        self._own = threading.local()
        self.leaf_of = None
        pos = C.leaves(self.work)
        if pos is not None:
            place = {(id(c), k): i for i, (c, k) in enumerate(pos)}
            params = {id(c[k].get("parameters")): i for i, (c, k) in enumerate(pos) if isinstance(c[k], dict)}
            self.leaf_of = np.array([place[(id(c), k)] if (id(c), k) in place else params[id(c)]
                                     for c, k in self.slots], dtype=np.int64)
            self.leaf_slots: Dict[int, List[int]] = {}
            for i, leaf in enumerate(self.leaf_of.tolist()):
                self.leaf_slots.setdefault(leaf, []).append(i)
            self._leaf_corr = copy.deepcopy(self.work)
            for i, (c, k) in enumerate(C.leaves(self._leaf_corr)):
                c[k] = float(i)
            self._leaf_ev = C.evaluator(self._leaf_corr, checked=False)

    def _mine(self):
        """This thread's copy of the correction, its numbers' slots and its leaves in it."""
        own = self._own
        if not hasattr(own, "work"):
            own.work = copy.deepcopy(self.work)
            own.slots = self.C.numeric_slots(own.work, nominal=self.var is not None)
            own.leaves = self.C.leaves(own.work) if self.leaf_of is not None else None
        return own

    def _clamped(self, points) -> Dict[str, np.ndarray]:
        return {"pt": np.maximum(points["pt"], self.pt_lo), "eta": np.clip(points["eta"], -self.eta_hi, self.eta_hi),
                "phi": points["phi"], "charge": np.asarray(points["charge"]).astype(np.int64)}

    def _theta(self, x: float) -> float:
        """A number from its value in the parameters' space (_set's map, for one)."""
        return float(np.sign(x) * np.sqrt(np.abs(x))) if self.signed else float(x)

    def p0(self) -> np.ndarray:
        t = self.theta0[self.active]
        return t * np.abs(t) if self.signed else t.copy()

    def _set(self, own, p) -> None:
        theta = self.theta0.copy()
        p = np.asarray(p, dtype=float)
        theta[self.active] = np.sign(p) * np.sqrt(np.abs(p)) if self.signed else p
        for (c, k), v in zip(own.slots, theta):
            c[k] = float(v)

    def value(self, p, points) -> np.ndarray:
        own = self._mine()
        self._set(own, p)
        return self.C._run(self.C.evaluator(own.work, checked=False), own.work, self._clamped(points), self.flavour,
                           self.var)

    def leaf_points(self, points) -> Optional[_LeafPoints]:
        """The points grouped by leaf (None: no leaves, moved() unavailable)."""
        if self.leaf_of is None:
            return None
        ids = self.C._run(self._leaf_ev, self._leaf_corr, self._clamped(points), self.flavour, self.var)
        return _LeafPoints(np.rint(ids).astype(np.int64), self.leaf_of)

    def moved(self, p, moves, points, at: _LeafPoints) -> List[np.ndarray]:
        """For each (j, x) of ``moves``, the correction at number j's leaf points with j at x, the others at p: moved
        leaves alone, up to MOVED_POINTS per correctionlib call (an added integer category selects each move)."""
        own = self._mine()
        theta = self.theta0.copy()
        p = np.asarray(p, dtype=float)
        theta[self.active] = np.sign(p) * np.sqrt(np.abs(p)) if self.signed else p
        out: List[Optional[np.ndarray]] = [None] * len(moves)
        items, chunks, which = [], [], []
        size = [0]

        def run() -> None:
            if not items:
                return
            names = self.C._input_names(own.work)
            sel = "_move"
            while sel in names:
                sel += "_"
            batch = {key: own.work[key] for key in ("name", "version", "output", "generic_formulas") if key in own.work}
            batch["inputs"] = own.work["inputs"] + [{"name": sel, "type": "int"}]
            batch["data"] = {"nodetype": "category", "input": sel, "content": items}
            sizes = [len(c) for c in chunks]
            pts = {key: v[np.concatenate(chunks)] for key, v in self._clamped(points).items()}
            pts[sel] = np.repeat(np.arange(len(items), dtype=np.int64), sizes)
            vals = self.C._run(self.C.evaluator(batch, checked=False), batch, pts, self.flavour, self.var)
            for m, part in zip(which, np.split(vals, np.cumsum(sizes)[:-1])):
                out[m] = part
            items.clear()
            chunks.clear()
            which.clear()
            size[0] = 0
        for m, (j, x) in enumerate(moves):
            slot = int(self.active[j])
            idx = at.of_slot(slot)
            if not len(idx):
                out[m] = np.zeros(0)
                continue
            leaf = int(self.leaf_of[slot])
            c, k = own.leaves[leaf]
            for i in self.leaf_slots[leaf]:
                sc, sk = own.slots[i]
                sc[sk] = self._theta(x) if i == slot else float(theta[i])
            if chunks and size[0] + len(idx) > MOVED_POINTS:
                run()
            items.append({"key": len(items), "value": copy.deepcopy(c[k])})
            chunks.append(idx)
            which.append(m)
            size[0] += len(idx)
        run()
        return out

    def moved_everywhere(self, p, moves, points, f0: np.ndarray) -> List[np.ndarray]:
        """For each move (moved()), the correction at every point: f0 with the moved leaf's points replaced (without
        leaves, the whole correction per move)."""
        at = self.leaf_points(points)
        out = []
        if at is None:
            for j, x in moves:
                q = np.asarray(p, dtype=float).copy()
                q[j] = x
                out.append(self.value(q, points))
            return out
        for (j, _), v in zip(moves, self.moved(p, moves, points, at)):
            f = f0.copy()
            f[at.of_slot(int(self.active[j]))] = v
            out.append(f)
        return out

    def restrict(self, points) -> None:
        """Keep the numbers whose change moves the correction at the points."""
        self.active = np.arange(len(self.slots))
        base = self.p0()
        f0 = self.value(base, points)
        moves = []
        for j, t in enumerate(self.theta0):
            h = 1e-3 * max(abs(t), 1e-3)
            moves.append((j, (t + h) * abs(t + h) if self.signed else t + h))
        at = self.leaf_points(points)
        if at is None:
            keep = [j for j, f in zip(range(len(moves)), self.moved_everywhere(base, moves, points, f0))
                    if np.any(f != f0)]
        else:
            keep = [j for (j, _), v in zip(moves, self.moved(base, moves, points, at))
                    if np.any(v != f0[at.of_slot(j)])]
        self.active = np.asarray(keep, dtype=np.int64)

    def jacobian(self, p, points, f0=None, rel: float = 1e-5) -> np.ndarray:
        """d value / d p at the points by forward differences (a column nonzero only at its leaf's points)."""
        p = np.asarray(p, dtype=float)
        f0 = self.value(p, points) if f0 is None else f0
        jac = np.zeros((len(points["pt"]), len(p)))
        hs = [rel * max(abs(p[j]), 1e-6 if self.signed else 1e-3) for j in range(len(p))]
        moves = [(j, p[j] + hs[j]) for j in range(len(p))]
        at = self.leaf_points(points)
        if at is None:
            for j, f in enumerate(self.moved_everywhere(p, moves, points, f0)):
                jac[:, j] = (f - f0) / hs[j]
            return jac
        for j, v in enumerate(self.moved(p, moves, points, at)):
            idx = at.of_slot(int(self.active[j]))
            jac[idx, j] = (v - f0[idx]) / hs[j]
        return jac


def _unit(k: int, j: int) -> np.ndarray:
    """The j-th unit vector of k numbers."""
    u = np.zeros(k)
    u[j] = 1.0
    return u


def free_directions(a: np.ndarray) -> List[np.ndarray]:
    """Unit vectors of the directions an information matrix does not constrain: an unseen parameter, or an eigenvalue
    of the correlation-scaled matrix below FREE_EIGEN of the largest."""
    k = a.shape[0]
    d = np.diag(a).copy()
    on = np.flatnonzero(d > 0)
    out = [_unit(k, j) for j in range(k) if d[j] <= 0]
    if len(on):
        s = 1.0 / np.sqrt(d[on])
        lam, vec = np.linalg.eigh(a[np.ix_(on, on)] * np.outer(s, s))
        for i in np.flatnonzero(lam <= FREE_EIGEN * max(float(lam.max()), 1e-300)):
            u = np.zeros(k)
            u[on] = s * vec[:, i]
            out.append(u / np.linalg.norm(u))
    return out


def _points(o: dict) -> Dict[str, np.ndarray]:
    return {"pt": o["pt"], "eta": o["eta"], "phi": o["phi"], "charge": np.asarray(o["charge"]).astype(np.int64)}


def _comparison(f: np.ndarray, signed: bool) -> np.ndarray:
    return f * np.abs(f) if signed else f



SPLIT_MIN_PROBES = 20
SPLIT_MAX_DEPTH = 4
FISHER_STEPS = 20


def _median_split(x: np.ndarray, b: np.ndarray, n: int, ok: np.ndarray) -> np.ndarray:
    """Per bin (0..n-1) the median of x over the data pairs in it (ok: those counted); +inf in an empty bin."""
    order = np.lexsort((x[ok], b[ok]))
    b_s, x_s = b[ok][order], x[ok][order]
    lo = np.searchsorted(b_s, np.arange(n), side="left")
    hi = np.searchsorted(b_s, np.arange(n), side="right")
    med = np.full(n, np.inf)
    has = hi > lo
    med[has] = x_s[lo[has] + (hi[has] - lo[has] - 1) // 2]
    return med


def split_bins(pm: dict, mid: str, depth: int = 1) -> Tuple[np.ndarray, np.ndarray, int]:
    """Each declared bin split along each variable at its data probes' median, ``depth`` times, so a correction in
    the measurement's own bins has a misfit to show too.  (data sub-bins, sim sub-bins, count); -1 outside."""
    d = len(pm["binning"])
    arrs = [pm["pairs"][smp] for smp in SAMPLES]
    cur = [bin_index(pm["binning"], {v: a[f"x_{v}"] for v, _ in pm["binning"]}, len(a["entry"])) for a in arrs]
    n = _n_bins(pm["binning"])
    for _ in range(depth):
        nxt = [np.where(c >= 0, c * 2 ** d, -1) for c in cur]
        for j, (v, _) in enumerate(pm["binning"]):
            med = _median_split(arrs[0][f"x_{v}"].astype(float), cur[0], n, cur[0] >= 0)
            for k, a in enumerate(arrs):
                up = (a[f"x_{v}"].astype(float) >= med[np.maximum(cur[k], 0)]).astype(np.int64)
                nxt[k] = np.where(cur[k] >= 0, nxt[k] + (up << j), -1)
        cur, n = nxt, n * 2 ** d
    return cur[0], cur[1], n


def _admissible(data_bin: np.ndarray, n: int, unit: Optional[np.ndarray] = None) -> bool:
    """Whether the non-empty bins hold a median of SPLIT_MIN_PROBES data probes, as effective units (count^2 over
    the sum of each unit's count^2; ``unit`` None: every pair its own)."""
    ok = data_bin >= 0
    counts = np.bincount(data_bin[ok], minlength=n).astype(float)
    if unit is not None:
        sq = _unit_squares(data_bin[ok], np.ones(int(ok.sum())), unit[ok], n)
        counts = np.where(sq > 0, counts ** 2 / np.where(sq > 0, sq, 1.0), 0.0)
    counts = counts[counts > 0]
    return bool(len(counts)) and float(np.median(counts)) >= SPLIT_MIN_PROBES


def _unit_squares(b: np.ndarray, w: np.ndarray, unit: np.ndarray, n: int) -> np.ndarray:
    """Per bin (0..n-1) the sum over statistical units (copies) of the unit's weight in the bin, squared."""
    key = unit.astype(np.int64) * n + b
    order = np.argsort(key, kind="stable")
    k, ws = key[order], w[order]
    if not len(k):
        return np.zeros(n)
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    s = np.add.reduceat(ws, starts)
    return np.bincount(k[starts] % n, s * s, n)


def _efficiency_counts(pm: dict, mid: str, data_bin: np.ndarray, sim_bin: np.ndarray, n: int,
                       units: Optional[Dict[str, np.ndarray]] = None) -> dict:
    """Per bin pass / fail sums of weights and their squares per unit (k identical pairs count k^2)."""
    out = {}
    for sample, b in (("data", data_bin), ("sim", sim_bin)):
        arr = pm["pairs"][sample]
        ok, p = b >= 0, arr[f"pass_{mid}"].astype(bool)
        w = arr["weight"].astype(float) if sample == "sim" else np.ones(len(b))
        unit = None if units is None else units[sample]
        if unit is None or np.array_equal(unit, np.arange(len(b))):
            sq = lambda m: np.bincount(b[m], w[m] ** 2, n)
        else:
            sq = lambda m: _unit_squares(b[m], w[m], unit[m], n)
        out[sample] = {"pass": np.bincount(b[ok & p], w[ok & p], n), "fail": np.bincount(b[ok & ~p], w[ok & ~p], n),
                       "pass_w2": sq(ok & p), "fail_w2": sq(ok & ~p)}
    return out


def _xlogy(x, y) -> np.ndarray:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.where(x > 0, x * np.log(np.where(x > 0, np.maximum(y, 1e-300), 1.0)), 0.0)


def _profile_efficiency(m, kd, nd, ks, ns) -> np.ndarray:
    """Per bin the e maximising data kd of nd at m x e and simulation ks of ns at e: the root (bisection) of
    (kd + ks) / e - m (nd - kd) / (1 - m e) - (ns - ks) / (1 - e)."""
    lo = np.full(np.shape(m), 1e-12)
    hi = np.minimum(1.0, 1.0 / np.maximum(m, 1e-300)) * (1.0 - 1e-12)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        g = (kd + ks) / mid - m * (nd - kd) / (1.0 - m * mid) - (ns - ks) / (1.0 - mid)
        lo, hi = np.where(g > 0, mid, lo), np.where(g > 0, hi, mid)
    return 0.5 * (lo + hi)


def _efficiency_terms(m, kd, nd, ks, ns, use):
    """At ratios m: (profiled simulated efficiency, per-bin log-likelihood, score d/dm, profile information on m),
    zero outside `use`."""
    mm = np.where(use, np.maximum(m, 1e-12), 1.0)
    e = np.clip(_profile_efficiency(mm, kd, nd, ks, ns), 1e-12, 1.0 - 1e-12)
    pd = np.clip(mm * e, 1e-12, 1.0 - 1e-12)
    f = _xlogy(kd, pd) + _xlogy(nd - kd, 1.0 - pd) + _xlogy(ks, e) + _xlogy(ns - ks, 1.0 - e)
    bound = (nd - kd <= 1e-9 * nd) & (mm > 1.0) & (mm * e >= 1.0 - 1e-9)
    s = np.where(bound, -ks / mm + (ns - ks) * e / ((1.0 - e) * mm), kd / mm - (nd - kd) * e / (1.0 - pd))
    i_pp = nd / (pd * (1.0 - pd))
    i_mm, i_me = i_pp * e * e, i_pp * mm * e
    i_ee = i_pp * mm * mm + ns / (e * (1.0 - e))
    info = i_mm - i_me * i_me / i_ee
    z = lambda a: np.where(use, a, 0.0)
    return e, z(f), z(s), z(info)


def _saturated(kd, nd, ks, ns, use) -> np.ndarray:
    """Each bin's log-likelihood with its ratio free (both samples fitted exactly)."""
    pd, e = kd / np.maximum(nd, 1e-300), ks / np.maximum(ns, 1e-300)
    f = _xlogy(kd, pd) + _xlogy(nd - kd, 1.0 - pd) + _xlogy(ks, e) + _xlogy(ns - ks, 1.0 - e)
    return np.where(use, f, 0.0)


def _fit_efficiency(form: _Form, counts: dict, sim_bin: np.ndarray, sim_pass: np.ndarray, pts_all: dict,
                    w_all: np.ndarray, n: int, seed: int, toys: bool) -> dict:
    """ML fit of the form to the bins' effective counts (count / (sum w^2 / sum w)): data passing binomial at m x e,
    simulation at e, e profiled per bin, m the correction averaged over the bin's passing simulated probes.  Fisher
    scoring, at most FISHER_STEPS.  With ``toys`` the GOF: deviance against the saturated model, p from N_TOYS of
    both samples at the fit, each refitted linearly."""
    keep = (sim_bin >= 0) & sim_pass
    pts = {k: v[keep] for k, v in pts_all.items()}
    w, bk = w_all[keep], sim_bin[keep]
    wsum = np.bincount(bk, w, n)
    dp, df, dp2, df2 = (counts["data"][k] for k in ("pass", "fail", "pass_w2", "fail_w2"))
    dn = dp + df
    kap = np.where(dn > 0, (dp2 + df2) / np.where(dn > 0, dn, 1.0), 1.0)
    kd, nd = dp / kap, dn / kap
    ps, fs, ps2, fs2 = (counts["sim"][k] for k in ("pass", "fail", "pass_w2", "fail_w2"))
    sw, sw2 = ps + fs, ps2 + fs2
    ns = np.where(sw2 > 0, sw ** 2 / np.where(sw2 > 0, sw2, 1.0), 0.0)
    ks = np.where(sw > 0, ps / np.where(sw > 0, sw, 1.0), 0.0) * ns
    use = (nd > 0) & (ns > 0) & (wsum > 0)

    def model(p):
        f = form.value(p, pts)
        jp = form.jacobian(p, pts, f)
        m = np.bincount(bk, w * f, n) / np.maximum(wsum, 1e-300)
        jac = np.stack([np.bincount(bk, w * jp[:, j], n) for j in range(len(p))], axis=1) \
            / np.maximum(wsum, 1e-300)[:, None] if len(p) else np.zeros((n, 0))
        return m, jac

    p = form.p0()
    converged, prev = False, None
    for _ in range(FISHER_STEPS):
        m, jac = model(p)
        _, _, s, info = _efficiency_terms(m, kd, nd, ks, ns, use)
        cov = np.linalg.pinv(jac.T @ (jac * info[:, None]))
        step = cov @ (jac.T @ s)
        p = p + step
        settled = np.abs(step) <= 1e-6 * np.sqrt(np.maximum(np.diag(cov), 1e-300))
        if prev is not None:
            settled |= (np.abs(step) <= 1e-10 * np.maximum(1.0, np.abs(p))) & (np.sign(step) == -np.sign(prev))
        if np.all(settled):
            converged = True
            break
        prev = step
    m, jac = model(p)
    e, f, s, info = _efficiency_terms(m, kd, nd, ks, ns, use)
    a = jac.T @ (jac * info[:, None])
    cov = np.linalg.pinv(a)
    dev = float(2.0 * np.sum(_saturated(kd, nd, ks, ns, use) - f))
    rank = int(np.linalg.matrix_rank(a)) if len(p) else 0
    ndf = int(np.count_nonzero(use)) - rank
    out = {"p": p, "cov": cov, "chi2": dev, "ndf": ndf, "n_bins": int(np.count_nonzero(use)), "p_gof": None,
           "free": free_directions(a), "converged": converged}
    if toys and ndf > 0:
        rng = np.random.default_rng(seed)
        nd_i, ns_i = np.rint(nd).astype(np.int64), np.maximum(np.rint(ns), 1).astype(np.int64)
        exceed = 0
        for _ in range(N_TOYS):
            kd_t = rng.binomial(nd_i, np.clip(m * e, 0.0, 1.0)).astype(float) \
                * np.where(nd_i > 0, nd / np.maximum(nd_i, 1), 0.0)
            ks_t = rng.binomial(ns_i, e).astype(float) * np.where(ns_i > 0, ns / ns_i, 0.0)
            _, _, s_t, _ = _efficiency_terms(m, kd_t, nd, ks_t, ns, use)
            m_t = m + jac @ (cov @ (jac.T @ s_t))
            _, f_t, _, _ = _efficiency_terms(m_t, kd_t, nd, ks_t, ns, use)
            exceed += float(2.0 * np.sum(_saturated(kd_t, nd, ks_t, ns, use) - f_t)) >= dev - GOF_TIE_EPS
        out["p_gof"] = (1.0 + exceed) / (1.0 + N_TOYS)
    return out


def refit_efficiency(form: _Form, pm: dict, mid: str, sim_objects: dict, seed: int = 0,
                     units: Optional[Dict[str, np.ndarray]] = None) -> dict:
    """Refit an efficiency (_fit_efficiency) in the bins split once: {"p", "cov", "free"} and the GOF ("chi2",
    "ndf", "n_bins", "p_gof", "depth"), re-split while ndf <= 0 and _admissible (at most SPLIT_MAX_DEPTH; finer
    than all: "p_gof" None).  ``units``: each sample's statistical units."""
    arr = pm["pairs"]["sim"]
    sim_pass = arr[f"pass_{mid}"].astype(bool)
    pts = _points(sim_objects["second"])
    w = arr["weight"].astype(float)
    db, sb, ns = split_bins(pm, mid, 1)
    keep = (sb >= 0) & sim_pass
    form.restrict({k: v[keep] for k, v in pts.items()})
    fit = _fit_efficiency(form, _efficiency_counts(pm, mid, db, sb, ns, units), sb, sim_pass, pts, w, ns, seed, True)
    out = {**fit, "depth": 1}
    depth = 1
    while out["ndf"] <= 0 and depth < SPLIT_MAX_DEPTH:
        depth += 1
        db, sb, ns = split_bins(pm, mid, depth)
        if not _admissible(db, ns, (units or {}).get("data")):
            break
        g = _fit_efficiency(form, _efficiency_counts(pm, mid, db, sb, ns, units), sb, sim_pass, pts, w, ns, seed,
                            True)
        out.update(chi2=g["chi2"], ndf=g["ndf"], n_bins=g["n_bins"], p_gof=g["p_gof"], depth=depth)
    return out



REFIT_ROUNDS = 3
FREE_EIGEN = 1e-10
MOVED_POINTS = 4_000_000


def _moments_mode(k: int, m: int) -> bool:
    """Refit k live numbers on m moments in the moments' space (_MomentSpace) when k > m: there the k x k pinv's
    cutoff, in parameter units, would decide what is constrained, and its cost (~4e-10 k^3 s per step on one core,
    41 s at 4,658 parameters) could exceed the budget."""
    return k > m
MIN_CATEGORY_DATA = 50
MIN_CATEGORY_SIM = 50
FREE_TOL = 1e-3


def _category_moments(cat: np.ndarray, x: np.ndarray, w: np.ndarray, n_cat: int) -> np.ndarray:
    """Per category the sums (sum w, sum w x, ..., sum w x^4, sum w^2), shape (6, n_cat)."""
    ok = cat >= 0
    c, xx, ww = cat[ok], x[ok], w[ok]
    return np.stack([np.bincount(c, ww * xx ** k, n_cat) for k in range(5)] + [np.bincount(c, ww * ww, n_cat)])


def _moments(sums: np.ndarray, replicas: float = 1.0):
    """Per category (mean, variance, their variances, effective count / `replicas`: smearing copies of one pair are
    not independent) from _category_moments' sums."""
    s0 = np.where(sums[0] > 0, sums[0], 1.0)
    m1 = sums[1] / s0
    m2 = sums[2] / s0 - m1 ** 2
    c4 = sums[4] / s0 - 4 * m1 * sums[3] / s0 + 6 * m1 ** 2 * sums[2] / s0 - 3 * m1 ** 4
    neff = np.where(sums[5] > 0, sums[0] ** 2 / np.where(sums[5] > 0, sums[5], 1.0), 0.0) / replicas
    with np.errstate(invalid="ignore", divide="ignore"):
        v1 = np.where(neff > 0, m2 / neff, np.inf)
        v2 = np.where(neff > 0, np.maximum(c4 - m2 ** 2, 0.0) / neff, np.inf)
    return m1, m2, v1, v2, neff


def _influence_variances(entries, nc: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """(moment-vector variances under the Poisson(1) bootstrap by the delta method; effective counts
    S0^2 / sum s0^2).  Entries (pair, category, weight, value) summed per (pair, category) first, so a pair's copies
    count together; influence on the mean (s1 - M s0) / S0, on the variance (s2 - 2 M s1 + M^2 s0 - V s0) / S0."""
    vm, vv, ne = [], [], []
    for (pair, cat, w, x), n in zip(entries, nc):
        key = pair.astype(np.int64) * n + cat
        order = np.argsort(key, kind="stable")
        key, w, x = key[order], w[order], x[order]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.zeros(0, dtype=np.int64)
        s0, s1, s2 = (np.add.reduceat(w * x ** k, starts) if len(key) else np.zeros(0) for k in range(3))
        c = key[starts] % n
        S0 = np.bincount(c, s0, n)
        S0 = np.where(S0 != 0, S0, 1.0)
        M = np.bincount(c, s1, n) / S0
        V = np.bincount(c, s2, n) / S0 - M ** 2
        dm = (s1 - M[c] * s0) / S0[c]
        dv = (s2 - 2.0 * M[c] * s1 + (M[c] ** 2 - V[c]) * s0) / S0[c]
        vm.append(np.bincount(c, dm * dm, n))
        vv.append(np.bincount(c, dv * dv, n))
        s02 = np.bincount(c, s0 * s0, n)
        ne.append(np.where(s02 > 0, np.bincount(c, s0, n) ** 2 / np.where(s02 > 0, s02, 1.0), 0.0))
    return np.concatenate(vm + vv), np.concatenate(ne)


def negative_groups(n: int, neg: np.ndarray, w: np.ndarray) -> List[Tuple[np.ndarray, float, np.ndarray]]:
    """Terms [(pairs, k, weights)] for negative extra variances v applied as k |v|: 3 T(0) - 3 T(|v|) + T(2 |v|),
    the template continued through zero to second order (no spurious minimum below zero)."""
    idx = np.flatnonzero(neg)
    return [(np.arange(n), 0.0, w * np.where(neg, 3.0, 1.0)), (idx, 1.0, -3.0 * w[neg]), (idx, 2.0, w[neg])]


EFFECT = {"scale": "scale", **RESOLUTION_EFFECT}


def _effect(name: str, f: np.ndarray) -> np.ndarray:
    """A correction's values as its effect: the scale itself, a resolution's extra variance width x |width|."""
    return f if name == "scale" else f * np.abs(f)


def smearing_widths(e: Dict[str, np.ndarray], idx: np.ndarray, k: float) -> Dict[str, np.ndarray]:
    """Smearing widths in pT (relative), eta, phi; a negative variance as k |v| (negative_groups)."""
    return {c: np.sqrt(np.where(e[c][idx] >= 0, e[c][idx], -k * e[c][idx])) for c in ("pt", "eta", "phi")}


class _Lineshape:
    """A lineshape model: simulated pairs under the corrections (one parameter vector) -- scale multiplies both pT,
    a resolution v = width x |width| smears pT (relative), eta or phi with Gaussian variance v (negative_groups),
    N_REPLICAS antithetic replicas -- then selection and histograms.  The refit compares per category the truncated
    mean and variance (soft_cells, so the response is smooth); the GOF compares the histograms."""

    def __init__(self, pm: dict, forms: Dict[str, _Form], sim: dict, weights: np.ndarray, data: dict, seed: int):
        self.hists, self.sel, self.forms, self.sim, self.seed = pm["histograms"], pm["selection"], forms, sim, seed
        self.w = np.asarray(weights, dtype=float)
        self.n = len(self.w)
        self.nc = [_n_bins(h["categories"]) for h in self.hists]
        x = self.data_vars = pair_variables(data["first"], data["second"])
        self.n_data = len(x["mass"])
        self.data_unit = np.asarray(data.get("unit", np.arange(self.n_data)), dtype=np.int64)
        self.sim_unit = np.asarray(sim.get("unit", np.arange(self.n)), dtype=np.int64)
        self.data_cells = [soft_cells(h, self.sel, x, self.n_data) for h in self.hists]
        self.data_x = [np.asarray(x[h["variable"]], dtype=float)[c[0]] for c, h in zip(self.data_cells, self.hists)]
        self.needed = {h["variable"] for h in self.hists} | {v for h in self.hists for v, _ in h["categories"]}
        self.names = list(forms)
        self.role_points = {role: _points(sim[role]) for role in ("first", "second")}
        both = {k: np.concatenate([self.role_points["first"][k], self.role_points["second"][k]])
                for k in ("pt", "eta", "phi", "charge")}
        for name in self.names:
            forms[name].restrict(both)
        self._at: Dict[str, Optional[Dict[str, _LeafPoints]]] = {}
        self.k = [len(forms[n].active) for n in self.names]
        self.p0 = np.concatenate([forms[n].p0() for n in self.names]) if self.names else np.zeros(0)
        self.data_m = self.data_moments()
        self.data_var, self.data_ne = _influence_variances(
            [(self.data_unit[r], c, m, xv) for (r, c, m), xv in zip(self.data_cells, self.data_x)], self.nc)

    def block(self, name: str) -> slice:
        i = self.names.index(name)
        return slice(int(sum(self.k[:i])), int(sum(self.k[:i + 1])))

    def owner(self, j: int) -> _Form:
        return self.forms[self.names[int(np.searchsorted(np.cumsum(self.k), j, side="right"))]]

    def _normals(self, r: int) -> np.ndarray:
        """One replica's fixed standard normals (6 per pair: pT, eta, phi of each object), drawn once."""
        cache = self.__dict__.setdefault("_z", {})
        if r not in cache:
            cache[r] = np.random.default_rng([self.seed, r]).standard_normal((6, self.n)).astype(np.float32)
        return cache[r]

    def _effects(self, p) -> Dict[str, Dict[str, np.ndarray]]:
        """Per role the scale and the extra variances (pt, eta, phi) of every simulated object."""
        out = {}
        for role in ("first", "second"):
            e = {"scale": np.zeros(self.n), "pt": np.zeros(self.n), "eta": np.zeros(self.n), "phi": np.zeros(self.n)}
            for name in self.names:
                e[EFFECT[name]] = _effect(name, self.forms[name].value(p[self.block(name)], self.role_points[role]))
            out[role] = e
        return out

    def _smeared(self, eff, z, li, gi, k: float) -> Dict[str, Dict[str, np.ndarray]]:
        """The simulated pairs gi, their effects at li of eff, smeared with their normals z."""
        objs = {}
        for j, role in enumerate(("first", "second")):
            o, e = self.sim[role], eff[role]
            wd = smearing_widths(e, li, k)
            objs[role] = {"pt": o["pt"][gi] * (1.0 + e["scale"][li]) * (1.0 + wd["pt"] * z[3 * j]),
                          "eta": o["eta"][gi] + wd["eta"] * z[3 * j + 1],
                          "phi": o["phi"][gi] + wd["phi"] * z[3 * j + 2], "charge": o["charge"][gi]}
        return objs

    def replicas(self, p, hists: Optional[List[dict]] = None, soft: bool = False) -> list:
        """Per replica, sign and negative_groups term: (cells per histogram, variable, weights / (2 N_REPLICAS),
        simulated pairs); ``soft``: soft_cells' triples."""
        return list(self.each_replica(p, hists, soft))

    def each_replica(self, p, hists: Optional[List[dict]] = None, soft: bool = False):
        """replicas(), one at a time (a prediction holds one folded copy of the simulation)."""
        return self._terms(self._effects(np.asarray(p, dtype=float)), None, hists, soft)

    def _terms(self, eff, pairs: Optional[np.ndarray] = None, hists: Optional[List[dict]] = None, soft: bool = False):
        """replicas()' terms under effects eff, of every simulated pair or of ``pairs`` (eff then holds theirs, in
        order) -- the template is a sum over pairs."""
        P = np.arange(self.n) if pairs is None else pairs
        for cells, xs, wt, li in self._term_rows(eff, pairs, hists, soft):
            yield cells, xs, wt, P[li]

    def _term_rows(self, eff, pairs: Optional[np.ndarray] = None, hists: Optional[List[dict]] = None,
                   soft: bool = False):
        """_terms(), each term with the positions in ``pairs`` of the pairs it holds (a pair listed twice is two)."""
        hists = hists or self.hists
        P = np.arange(self.n) if pairs is None else pairs
        neg = np.zeros(len(P), dtype=bool)
        for role in ("first", "second"):
            for c in ("pt", "eta", "phi"):
                neg |= eff[role][c] < 0
        groups = negative_groups(len(P), neg, self.w[P])
        for r in range(N_REPLICAS):
            for sign in (1.0, -1.0):
                for li, k, wt in groups:
                    if not len(li):
                        continue
                    gi = P[li]
                    o = self._smeared(eff, sign * self._normals(r)[:, gi], li, gi, k)
                    x = pair_variables(o["first"], o["second"], self.needed)
                    if soft:
                        cells = [soft_cells(h, self.sel, x, len(li)) for h in hists]
                        xs = [np.asarray(x[h["variable"]], dtype=float)[c[0]] for c, h in zip(cells, hists)]
                    else:
                        cells = [histogram_cells(h, self.sel, x, len(li)) for h in hists]
                        xs = [np.asarray(x[h["variable"]], dtype=float) for h in hists]
                    yield cells, xs, wt / (2 * N_REPLICAS), li

    def _sums_by(self, eff, pairs: np.ndarray, label: np.ndarray, n_label: int) -> np.ndarray:
        """template_sums of ``pairs`` per label (label[i] of pairs[i]): shape (n_label, 6, all categories), each label's
        sums as a call with its pairs alone would give."""
        nct = int(sum(self.nc))
        offs = np.concatenate([[0], np.cumsum(self.nc)]).astype(np.int64)
        total = np.zeros((6, n_label * nct))
        for cells, xs, wt, li in self._term_rows(eff, pairs, soft=True):
            for h, ((rows, cat, m), xv) in enumerate(zip(cells, xs)):
                ok = cat >= 0
                key = label[li[rows[ok]]] * nct + offs[h] + cat[ok]
                w, x = (wt[rows] * m)[ok], xv[ok]
                total += np.stack([np.bincount(key, w * x ** k, n_label * nct) for k in range(5)]
                                  + [np.bincount(key, w * w, n_label * nct)])
        return total.reshape(6, n_label, nct).transpose(1, 0, 2)

    def _sums(self, cells_list, x_list, weights) -> np.ndarray:
        """Moment sums per category, all histograms concatenated, of soft counts (weights[pair] x membership)."""
        return np.concatenate([_category_moments(cat, xv, weights[rows] * m, nc)
                               for (rows, cat, m), xv, nc in zip(cells_list, x_list, self.nc)], axis=1)

    def data_moments(self, scale: Optional[np.ndarray] = None):
        w = np.ones(self.n_data) if scale is None else scale
        return _moments(self._sums(self.data_cells, self.data_x, w))

    def template_sums(self, reps) -> np.ndarray:
        total = None
        for cells, xs, wt, idx in reps:
            s = self._sums(cells, xs, wt)
            total = s if total is None else total + s
        return total

    def template_moments(self, reps):
        return _moments(self.template_sums(reps), replicas=float(2 * N_REPLICAS))

    def moment_vector(self, m) -> np.ndarray:
        return np.concatenate([m[0], m[1]])

    def predict(self, p) -> np.ndarray:
        """The template's moment vector (all means, then all variances) under p."""
        return self.moment_vector(self.template_moments(self.each_replica(p, soft=True)))

    def base(self, p) -> dict:
        """The template at p for predict_moved: p, effects, moment sums and vector; the forms' leaves found once."""
        p = np.asarray(p, dtype=float)
        eff = self._effects(p)
        sums = self.template_sums(self._terms(eff, soft=True))
        for name in self.names:
            self._leaves(name)
        return {"p": p, "eff": eff, "sums": sums,
                "m": self.moment_vector(_moments(sums, replicas=float(2 * N_REPLICAS)))}

    def _leaves(self, name: str) -> Optional[Dict[str, _LeafPoints]]:
        """Per role the simulated objects grouped by the form's leaf (None: no leaves)."""
        if name not in self._at:
            f = self.forms[name]
            self._at[name] = (None if f.leaf_of is None else
                              {role: f.leaf_points(self.role_points[role]) for role in ("first", "second")})
        return self._at[name]

    def reach(self, j: int) -> Tuple[str, int, Optional[Dict[str, np.ndarray]], Optional[np.ndarray]]:
        """(form, index in it, per role the objects at its leaf, their pairs) of parameter j; None, None: no leaves."""
        cache = self.__dict__.setdefault("_reach", {})
        if j not in cache:
            name = self.names[int(np.searchsorted(np.cumsum(self.k), j, side="right"))]
            jj, at = j - self.block(name).start, self._leaves(name)
            if at is None:
                cache[j] = (name, jj, None, None)
            else:
                slot = int(self.forms[name].active[jj])
                r = {role: at[role].of_slot(slot) for role in ("first", "second")}
                cache[j] = (name, jj, r, np.union1d(r["first"], r["second"]))
        return cache[j]

    def wide(self, j: int) -> bool:
        """Whether a move of parameter j is predicted whole: its form has no leaves, or it reaches most pairs."""
        _, _, r, P = self.reach(j)
        return r is None or 2 * len(P) >= self.n

    def predict_moved(self, base: dict, moves) -> List[np.ndarray]:
        """[predict(p + delta e_j) for (j, delta) in moves].  A wide() move is the whole template, each in a thread;
        the others go together: the base's sums less the reached pairs' terms, plus their terms moved."""
        out: List[Optional[np.ndarray]] = [None] * len(moves)
        wide = [m for m, (j, _) in enumerate(moves) if self.wide(j)]

        def whole(m: int) -> np.ndarray:
            j, delta = moves[m]
            name, jj, r, _ = self.reach(j)
            if r is None:
                q = base["p"].copy()
                q[j] += delta
                return self.predict(q)
            form, pb = self.forms[name], base["p"][self.block(name)]
            eff = {}
            for role in ("first", "second"):
                eff[role] = dict(base["eff"][role])
                e = eff[role][EFFECT[name]] = base["eff"][role][EFFECT[name]].copy()
                e[r[role]] = _effect(name, form.moved(pb, [(jj, pb[jj] + delta)], self.role_points[role],
                                                      self._leaves(name)[role])[0])
            return self.moment_vector(self.template_moments(self._terms(eff, soft=True)))
        for m, got in zip(wide, in_threads(whole, wide)):
            out[m] = got
        narrow: Dict[str, List[int]] = {}
        for m, (j, _) in enumerate(moves):
            if out[m] is None:
                narrow.setdefault(self.reach(j)[0], []).append(m)
        chunks = []
        for name, ms in narrow.items():
            cur, size = [], 0
            for m in ms:
                cur.append(m)
                size += len(self.reach(moves[m][0])[3])
                if size >= NARROW_PAIRS:
                    chunks.append((name, cur))
                    cur, size = [], 0
            if cur:
                chunks.append((name, cur))

        def together(chunk) -> List[np.ndarray]:
            name, ms = chunk
            form, pb, at = self.forms[name], base["p"][self.block(name)], self._leaves(name)
            got = [self.reach(moves[m][0]) for m in ms]
            mv = [(jj, pb[jj] + moves[m][1]) for m, (_, jj, _, _) in zip(ms, got)]
            vals = {role: form.moved(pb, mv, self.role_points[role], at[role]) for role in ("first", "second")}
            sizes = [len(P) for _, _, _, P in got]
            pairs = np.concatenate([P for _, _, _, P in got])
            label = np.repeat(np.arange(len(ms), dtype=np.int64), sizes)
            start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            was = {role: {c: v[pairs] for c, v in base["eff"][role].items()} for role in ("first", "second")}
            now = {role: dict(was[role]) for role in ("first", "second")}
            for role in ("first", "second"):
                e = now[role][EFFECT[name]] = was[role][EFFECT[name]].copy()
                for i, (_, _, r, P) in enumerate(got):
                    e[start[i] + np.searchsorted(P, r[role])] = _effect(name, vals[role][i])
            sums = (base["sums"][None] - self._sums_by(was, pairs, label, len(ms))
                    + self._sums_by(now, pairs, label, len(ms)))
            return [self.moment_vector(_moments(x, replicas=float(2 * N_REPLICAS))) for x in sums]
        for (name, ms), got in zip(chunks, in_threads(together, chunks)):
            for m, x in zip(ms, got):
                out[m] = x
        return out

    def template_variances(self, reps) -> Tuple[np.ndarray, np.ndarray]:
        """_influence_variances of the template, by histogram."""
        return _influence_variances([tuple(np.concatenate(a) for a in zip(*[
            (self.sim_unit[idx[cells[h][0]]], cells[h][1], wt[cells[h][0]] * cells[h][2], xs[h])
            for cells, xs, wt, idx in reps])) for h in range(len(self.hists))], self.nc)

    def weights(self, tvar: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """1 / (data + template moment variance, delta method) over the categories with enough of both (effective
        counts: data per unit, template per pair with its replicas together)."""
        ok = (self.data_ne >= MIN_CATEGORY_DATA) & (tvar[1] >= MIN_CATEGORY_SIM)
        v = self.data_var + tvar[0]
        good = np.concatenate([ok, ok]) & np.isfinite(v) & (v > 0)
        return np.where(good, 1.0 / np.where(good, v, 1.0), 0.0)


THREADS = min(4, CPUS)
NARROW_PAIRS = 250_000


def in_threads(f: Callable, items) -> list:
    """[f(x) for x in items], in up to THREADS threads (in order)."""
    items = list(items)
    if THREADS < 2 or len(items) < 2:
        return [f(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(min(THREADS, len(items))) as ex:
        return list(ex.map(f, items))


def lineshape_jacobian(ls: _Lineshape, p: np.ndarray, m0: np.ndarray, wt: np.ndarray,
                       steps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """(d moments / d p, steps, unconstrained unit vectors): central differences over a step moving the misfit by 0.1
    to 10 units (searched; none found: unconstrained, column zero), or the given ``steps``.  wide() columns are
    searched in threads, the others together."""
    k = len(p)
    base = ls.base(p)
    h = np.array([1e-5 if ls.owner(j).signed else 1e-3 * max(abs(p[j]), 1e-2) for j in range(k)]) \
        if steps is None else np.asarray(steps, dtype=float)
    d = np.zeros(k)

    def misfit(m: np.ndarray) -> float:
        return float(np.sum(wt * np.nan_to_num(m - m0) ** 2))

    def search(j: int) -> None:
        for _ in range(8):
            d[j] = misfit(ls.predict_moved(base, [(j, h[j])])[0])
            if 0.1 <= d[j] <= 10.0:
                return
            h[j] = h[j] * (10.0 if d[j] == 0.0 else float(np.clip(1.0 / math.sqrt(d[j]), 0.1, 10.0)))
    if steps is None:
        in_threads(search, [j for j in range(k) if ls.wide(j)])
        todo = [j for j in range(k) if not ls.wide(j)]
        for _ in range(8):
            if not todo:
                break
            again = []
            for j, m in zip(todo, ls.predict_moved(base, [(j, h[j]) for j in todo])):
                d[j] = misfit(m)
                if not 0.1 <= d[j] <= 10.0:
                    h[j] = h[j] * (10.0 if d[j] == 0.0 else float(np.clip(1.0 / math.sqrt(d[j]), 0.1, 10.0)))
                    again.append(j)
            todo = again
        live = [j for j in range(k) if 0.1 <= d[j] <= 10.0]
    else:
        live = [j for j in range(k) if h[j] > 0]
    ms = ls.predict_moved(base, [(j, h[j]) for j in live] + [(j, -h[j]) for j in live])
    jac, used = np.zeros((len(m0), k)), np.zeros(k)
    for i, j in enumerate(live):
        jac[:, j], used[j] = np.nan_to_num(ms[i] - ms[len(live) + i]) / (2.0 * h[j]), h[j]
    on = set(live)
    return jac, used, [_unit(k, j) for j in range(k) if j not in on]


class _MomentSpace:
    """A lineshape refit's linear algebra in the moments' space (_moments_mode), from one thin SVD of the weighted
    response with unit-norm columns, B S^-1 = U s V^T (B = W^1/2 J, m x k; unseen parameters left out), keeping
    s^2 > FREE_EIGEN s_max^2.  LM step dp = S^-1 V (s / (s^2 + lambda)) U^T W^1/2 r (the Marquardt-damped normal
    equations), errors S^-1 sqrt(sum V^2 / s^2), the rank, and the covariance's projection S^-1 V s^-1 U^T W^1/2,
    at O(m^2 k): one scale-free rule for step, covariance and free directions.  Well conditioned it equals the
    k x k pseudo-inverse's algebra to rounding."""

    def __init__(self, jac: np.ndarray, wt: np.ndarray):
        self.k = jac.shape[1]
        self.sw = np.sqrt(wt)
        b = self.sw[:, None] * jac
        norm = np.linalg.norm(b, axis=0)
        self.on = np.flatnonzero(norm > 0)
        self.s = norm[self.on]
        if len(self.on):
            u, sv, vt = np.linalg.svd(b[:, self.on] / self.s[None, :], full_matrices=False)
            keep = sv ** 2 > FREE_EIGEN * sv[0] ** 2
            self.u, self.sv, self.v = u[:, keep], sv[keep], vt[keep].T
        else:
            self.u, self.sv, self.v = np.zeros((len(wt), 0)), np.zeros(0), np.zeros((0, 0))

    @property
    def rank(self) -> int:
        return int(len(self.sv))

    @property
    def sd(self) -> np.ndarray:
        out = np.zeros(self.k)
        if self.rank:
            out[self.on] = np.sqrt(np.sum((self.v / self.sv[None, :]) ** 2, axis=1)) / self.s
        return out

    def step(self, lam: float, resid: np.ndarray) -> np.ndarray:
        dp = np.zeros(self.k)
        if self.rank:
            dp[self.on] = (self.v @ ((self.sv / (self.sv ** 2 + lam)) * (self.u.T @ (self.sw * resid)))) / self.s
        return dp

    def proj(self) -> np.ndarray:
        out = np.zeros((self.k, len(self.sw)))
        if self.rank:
            out[self.on] = ((self.v / self.sv[None, :]) @ (self.u.T * self.sw[None, :])) / self.s[:, None]
        return out


def refit_lineshape(ls: _Lineshape, log: Optional[Callable[[str], None]] = None) -> dict:
    """Weighted least-squares (Levenberg-Marquardt) refit of a lineshape's corrections to the data's moments.
    Response and weights renewed per round (at most REFIT_ROUNDS, another while one moves a parameter by > 0.5
    sigma); converged when no step moves a parameter by 0.1 sigma or two smaller steps fail to lower the misfit.
    Steps by pinv or _MomentSpace (_moments_mode on the first Jacobian).  {"p", "jac", "wt", "m", "chi2",
    "n_moments", "rank", "free", "lm", "lm_k", "lm_m"}."""
    p = ls.p0.copy()
    ydat = ls.moment_vector(ls.data_m)
    steps, shift = None, 0.0
    lm = None
    for rnd in range(REFIT_ROUNDS):
        reps = ls.replicas(p, soft=True)
        tm = ls.template_moments(reps)
        m0 = ls.moment_vector(tm)
        wt = ls.weights(ls.template_variances(reps))
        jac, steps, free = lineshape_jacobian(ls, p, m0, wt, steps)
        if lm is None:
            lm_k = int(np.count_nonzero(np.any((jac != 0) & (wt[:, None] > 0), axis=0)))
            lm_m = int(np.count_nonzero(wt > 0))
            lm = "moments" if _moments_mode(lm_k, lm_m) else "pinv"
        if lm == "moments":
            space = _MomentSpace(jac, wt)
            solve, sd = space.step, space.sd
        else:
            a = jac.T @ (jac * wt[:, None])
            cov = np.linalg.pinv(a)
            sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
        resid = np.nan_to_num(ydat - m0)
        chi = float(np.sum(wt * resid ** 2))
        lam, total, rejected = 1e-3, np.zeros(len(p)), 0
        for it in range(12):
            dp = (solve(lam, resid) if lm == "moments" else
                  np.linalg.pinv(a + lam * np.diag(np.diag(a))) @ (jac.T @ (wt * resid)))
            moved = float(np.max(np.abs(dp) / np.where(sd > 0, sd, np.inf), initial=0.0))
            if moved <= 0.1:
                break
            q = p + dp
            mq = ls.predict(q)
            cq = float(np.sum(wt * np.nan_to_num(ydat - mq) ** 2))
            if log:
                log(f"  lineshape refit {rnd + 1}.{it + 1}: chi2 {chi:.2f} -> {cq:.2f}, max |dp| / sigma {moved:.3f}"
                    f", lambda {lam:.0e}")
            if cq < chi:
                p, chi, lam, total, rejected = q, cq, lam / 3.0, total + dp, 0
                resid = np.nan_to_num(ydat - mq)
            else:
                lam, rejected = lam * 4.0, rejected + 1
                if rejected >= 2 and moved <= 0.3:
                    break
        shift = float(np.max(np.abs(total) / np.where(sd > 0, sd, np.inf), initial=0.0))
        if shift <= 0.5:
            break
    reps = ls.replicas(p, soft=True)
    tm = ls.template_moments(reps)
    m0 = ls.moment_vector(tm)
    wt = ls.weights(ls.template_variances(reps))
    if shift > 0.5:
        jac, steps, free = lineshape_jacobian(ls, p, m0, wt, steps)
    a = jac.T @ (jac * wt[:, None])
    rank = (_MomentSpace(jac, wt).rank if lm == "moments" else int(np.linalg.matrix_rank(a))) if a.size else 0
    return {"p": p, "jac": jac, "wt": wt, "m": m0, "chi2": float(np.sum(wt * np.nan_to_num(ydat - m0) ** 2)),
            "n_moments": int(np.count_nonzero(wt > 0)), "rank": rank,
            "free": free_directions(a) if a.size else free, "lm": lm, "lm_k": lm_k, "lm_m": lm_m}


def lineshape_covariance(ls: _Lineshape, fit: dict) -> np.ndarray:
    """The refit's covariance by the delta method of the Poisson(1) bootstrap of both samples (histograms sharing
    pairs fluctuate together; the template carries the simulation's statistics): proj (G^T G) proj^T, G the sparse
    per-unit influences on the moments -- mean w (x - M) / S0, variance w ((x - M)^2 - V) / S0, summed over a unit's
    counted copies."""
    jac, wt = fit["jac"], fit["wt"]
    k = len(fit["p"])
    if not k:
        return np.zeros((0, 0))
    proj = (_MomentSpace(jac, wt).proj() if fit.get("lm") == "moments"
            else np.linalg.pinv(jac.T @ (jac * wt[:, None])) @ (jac.T * wt))
    nct = int(sum(ls.nc))
    offs = np.concatenate([[0], np.cumsum(ls.nc)]).astype(np.int64)

    def outer(entries, n: int, sums: np.ndarray) -> np.ndarray:
        from scipy import sparse
        s0 = np.where(sums[0] > 0, sums[0], 1.0)
        mean = sums[1] / s0
        var = sums[2] / s0 - mean ** 2
        rows, cols, vals = [], [], []
        for h, (pair, cat, w, x) in enumerate(entries):
            c = offs[h] + cat
            rows += [pair, pair]
            cols += [c, nct + c]
            vals += [w * (x - mean[c]) / s0[c], w * ((x - mean[c]) ** 2 - var[c]) / s0[c]]
        if not sum(len(v) for v in vals):
            return np.zeros((k, k))
        g = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(n, 2 * nct))
        return proj @ (g.T @ g).toarray() @ proj.T

    data = [(ls.data_unit[rows], cat, m, x) for (rows, cat, m), x in zip(ls.data_cells, ls.data_x)]
    reps = ls.replicas(fit["p"], soft=True)
    sim = [tuple(np.concatenate(a) for a in zip(*[(ls.sim_unit[idx[cells[h][0]]], cells[h][1],
                                                    wt[cells[h][0]] * cells[h][2], xs[h])
                                                   for cells, xs, wt, idx in reps]))
           for h in range(len(ls.hists))]
    return outer(data, ls.n_data, ls._sums(ls.data_cells, ls.data_x, np.ones(ls.n_data))) \
        + outer(sim, ls.n, ls.template_sums(reps))


def _join_sparse(y: np.ndarray, te: np.ndarray, m: float = GOF_MIN_CELL) -> Tuple[np.ndarray, np.ndarray]:
    """Per category (row) consecutive bins joined until data count plus template effective count reach m (a
    remainder joins the last cell), rows padded with empty cells to one length."""
    ys, ts = [], []
    for yc, tc in zip(y, te):
        gy, gt, ay, at = [], [], 0.0, 0.0
        for a, b in zip(yc, tc):
            ay, at = ay + a, at + b
            if ay + at >= m:
                gy.append(ay)
                gt.append(at)
                ay = at = 0.0
        if ay + at > 0:
            if gy:
                gy[-1], gt[-1] = gy[-1] + ay, gt[-1] + at
            else:
                gy, gt = [ay], [at]
        ys.append(gy)
        ts.append(gt)
    n = max([len(r) for r in ys] + [1])
    return (np.array([r + [0.0] * (n - len(r)) for r in ys]), np.array([r + [0.0] * (n - len(r)) for r in ts]))


def _homogeneity(y: np.ndarray, te: np.ndarray, ok: np.ndarray) -> float:
    """Two-sample chi2 of per-category shapes (rows): data counts y and template effective counts te against their
    pooled shape, over the categories `ok`."""
    yy, tt = y[ok], te[ok]
    Y, T = yy.sum(axis=1, keepdims=True), tt.sum(axis=1, keepdims=True)
    pool = (yy + tt) / np.maximum(Y + T, 1e-300)
    ey, et = Y * pool, T * pool
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(ey > 0, (yy - ey) ** 2 / ey, 0.0) + np.where(et > 0, (tt - et) ** 2 / et, 0.0)
    return float(q.sum())


def lineshape_gof(ls: _Lineshape, p: np.ndarray, hists: List[dict], n_toys: int = N_TOYS, seed: int = 0) -> dict:
    """GOF in the given (finer) histograms: per category both samples as effective counts (per statistical unit)
    tested for one shape by the two-sample (homogeneity) chi2, so neither sample's fluctuation sets the other's
    weights; categories with MIN_CATEGORY_DATA / MIN_CATEGORY_SIM, cells joined (_join_sparse).  Per histogram p from
    n_toys of both samples at the pooled shapes; overall p = 1 - (1 - p_min)^n_tested (Sidak).
    {"p_gof", "chi2", "ndf", "n_cells", "per_histogram"}."""
    reps = ls.replicas(p, hists=hists)
    rng = np.random.default_rng(seed + 7)
    own = np.array_equal(ls.data_unit, np.arange(ls.n_data))
    per = []
    for h, hist in enumerate(hists):
        nb, nc = len(hist["edges"]) - 1, _n_bins(hist["categories"])
        size = nb * nc
        dc = histogram_cells(hist, ls.sel, ls.data_vars, ls.n_data)
        y = np.bincount(dc[dc >= 0], minlength=size).astype(float).reshape(nc, nb)
        if not own:
            inc = dc >= 0
            y2 = _unit_squares(dc[inc], np.ones(int(inc.sum())), ls.data_unit[inc], size).reshape(nc, nb)
            yw = y.sum(axis=1)
            y = y / np.where(yw > 0, y2.sum(axis=1) / np.where(yw > 0, yw, 1.0), 1.0)[:, None]
        t = np.zeros(size)
        keys, ws = [], []
        for cells, _, wt, idx in reps:
            c = cells[h]
            ok = c >= 0
            t += np.bincount(c[ok], wt[ok], size)
            keys.append(ls.sim_unit[idx[ok]] * size + c[ok])
            ws.append(wt[ok])
        k, wk = np.concatenate(keys), np.concatenate(ws)
        t2 = np.zeros(size)
        if len(k):
            order = np.argsort(k, kind="stable")
            k, wk = k[order], wk[order]
            starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
            t2 = np.bincount(k[starts] % size, np.add.reduceat(wk, starts) ** 2, size)
        t, t2 = t.reshape(nc, nb), t2.reshape(nc, nb)
        tw, tw2 = t.sum(axis=1), t2.sum(axis=1)
        kap = np.where(tw > 0, tw2 / np.where(tw > 0, tw, 1.0), 1.0)
        te = np.maximum(t, 0.0) / kap[:, None]
        ok = (y.sum(axis=1) >= MIN_CATEGORY_DATA) & (te.sum(axis=1) >= MIN_CATEGORY_SIM)
        y, te = _join_sparse(y, te)
        q = _homogeneity(y, te, ok)
        cells = int(np.count_nonzero((y + te)[ok] > 0))
        Y, T = y[ok].sum(axis=1, keepdims=True), te[ok].sum(axis=1, keepdims=True)
        pool = (y[ok] + te[ok]) / np.maximum(Y + T, 1e-300)
        exceed = 0
        for _ in range(n_toys if ok.any() else 0):
            y_t, t_t = np.zeros_like(y), np.zeros_like(te)
            y_t[ok] = rng.poisson(Y * pool)
            t_t[ok] = rng.poisson(T * pool)
            exceed += _homogeneity(y_t, t_t, ok) >= q - GOF_TIE_EPS
        per.append({"chi2": q, "ndf": cells - int(ok.sum()), "n_cells": cells, "n_categories": int(ok.sum()),
                    "p": (1.0 + exceed) / (1.0 + n_toys) if ok.any() else None})
    ndf = int(sum(r["ndf"] for r in per)) - len(p)
    tested = [r["p"] for r in per if r["p"] is not None]
    p_gof = 1.0 - (1.0 - min(tested)) ** len(tested) if tested else 1.0
    return {"p_gof": p_gof, "chi2": float(sum(r["chi2"] for r in per)), "ndf": ndf,
            "n_cells": int(sum(r["n_cells"] for r in per)), "per_histogram": per}



def refit_comparison(form: _Form, p_fit, cov, points, rep_value, rep_sigma, free=()) -> dict:
    """Reported correction against the refit at its points (display only, comparison space):
    z = RMS of |reported - refit| / sqrt(sigma_refit^2 + sigma_rep^2), and both sigma-bars.  Points without refit
    information (no moved number, or along a ``free`` direction) are left out; {"why"} when none is left."""
    p_fit = np.asarray(p_fit, dtype=float)
    f = form.value(p_fit, points)
    c = _comparison(f, form.signed)
    if not len(p_fit):
        return {"z": 0.0, "sigma_fit": None, "sigma_reported": None, "n_points": 0, "n_points_all": int(len(c)),
                "refit_mean": float(np.mean(c)), "reported_mean": float(np.mean(np.asarray(rep_value, dtype=float))),
                "note": "no free number"}
    g = np.zeros((len(c), len(p_fit)))
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0)) if len(p_fit) else np.zeros(0)
    hs = [1e-3 * sd[j] if sd[j] > 0 else 1e-7 * max(abs(p_fit[j]), 1e-3) for j in range(len(p_fit))]
    moves = [(j, p_fit[j] + hs[j]) for j in range(len(p_fit))]
    for j, fq in enumerate(form.moved_everywhere(p_fit, moves, points, f)):
        g[:, j] = (_comparison(fq, form.signed) - c) / hs[j]
    s = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", g, cov, g), 0.0)) if len(p_fit) else np.zeros(len(c))
    ok = s > 0
    gn = np.linalg.norm(g, axis=1)
    for u in free:
        ok &= ~(np.abs(g @ u) > FREE_TOL * gn * np.linalg.norm(u))
    if not np.any(ok):
        return {"why": "its measurement does not determine it at the points it is evaluated at"}
    rv, rs = np.asarray(rep_value, dtype=float)[ok], np.asarray(rep_sigma, dtype=float)[ok]
    z = float(np.sqrt(np.mean(((rv - c[ok]) / np.hypot(s[ok], rs)) ** 2)))
    return {"z": z, "sigma_fit": float(np.exp(np.mean(np.log(s[ok])))),
            "sigma_reported": float(np.exp(np.mean(np.log(np.maximum(rs, 1e-300))))),
            "n_points": int(np.count_nonzero(ok)), "n_points_all": int(len(c)),
            "refit_mean": float(np.mean(c[ok])), "reported_mean": float(np.mean(rv))}


def _round(x, sig: int = 4):
    return None if x is None or not np.isfinite(x) else float(f"{float(x):.{sig}g}")


def _variation_numbers(corr: dict, key: str, n: int) -> Optional[np.ndarray]:
    """The numbers of variation ``key`` in numeric_slots' nominal order (nominal swapped for it in every variation
    category), or None: no such variation, or one built otherwise than the nominal (another count of numbers)."""
    from . import outputs as C
    work, found = copy.deepcopy(corr), False
    stack = [work.get("data")]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
            continue
        if not isinstance(node, dict):
            continue
        if node.get("nodetype") == "category" and node.get("input") == "variation":
            items = {it.get("key"): it for it in node.get("content") or [] if isinstance(it, dict)}
            if key in items and "nominal" in items:
                items["nominal"]["value"] = copy.deepcopy(items[key]["value"])
                found = True
        stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
    if not found:
        return None
    slots = C.numeric_slots(work, nominal=True)
    return np.array([float(c[k]) for c, k in slots]) if len(slots) == n else None


def reported_sigma(form: "_Form", corr: dict, result: Any, cset: dict) -> Optional[np.ndarray]:
    """Each refitted number's reported uncertainty (parameter space): half the up-down distance, else the sources'
    half-distances and covariance sigmas in quadrature; None when not readable number by number."""
    from . import outputs as C
    spec, problems = C.uncertainty_spec(corr, result, cset)
    if problems:
        return None
    nom = form.theta0
    sq = (lambda x: x * np.abs(x)) if form.signed else (lambda x: x)

    def half(up, down):
        u, d = _variation_numbers(corr, up, len(nom)), _variation_numbers(corr, down, len(nom))
        return None if u is None or d is None else 0.5 * np.abs(sq(u) - sq(d))
    if spec["band"]:
        s = half("up", "down")
        return None if s is None else s[form.active]
    var, any_part = np.zeros(len(nom)), False
    for src in spec["sources"]:
        s = half(f"up_{src}", f"down_{src}")
        if s is None:
            return None
        var, any_part = var + s ** 2, True
    mats = [np.asarray(m, dtype=float) for m in spec["covariances"].values()] + \
        [np.asarray(m, dtype=float) for _, m in spec["joint"]]
    if mats:
        if len(C.param_slots(corr["data"], nominal=form.var is not None)) != len(nom):
            return None
        sd = np.sqrt(np.maximum(np.sum([np.diag(m) for m in mats], axis=0), 0.0))
        var, any_part = var + ((2.0 * np.abs(nom) * sd) if form.signed else sd) ** 2, True
    return np.sqrt(var)[form.active] if any_part else None


def parameter_table(form: "_Form", p_fit, cov, points, result: Any, cset: dict) -> dict:
    """Pulls-and-impacts table (display only, comparison space): {"mean", "sigma": the refitted mean over the
    points and its error, "rows": [name, refit, error, reported, reported error, pull = (refit - reported) /
    reported error, constraint = error / reported error, correlation with the mean, impact +/- (linear)]}."""
    from . import outputs as C
    p_fit, cov = np.asarray(p_fit, dtype=float), np.asarray(cov, dtype=float)
    labels = C.numeric_slot_labels(form.work, nominal=form.var is not None)
    names = [labels[j] if len(labels) == len(form.slots) else f"number {j}" for j in form.active]
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0)) if len(p_fit) else np.zeros(0)
    pre, pre_sd = form.p0(), reported_sigma(form, form.work, result, cset)

    f = form.value(p_fit, points)
    c0 = float(np.mean(_comparison(f, form.signed)))
    g = np.zeros(len(p_fit))
    hs = [1e-3 * sd[j] if sd[j] > 0 else 1e-7 * max(abs(p_fit[j]), 1e-3) for j in range(len(p_fit))]
    moves = [(j, p_fit[j] + hs[j]) for j in range(len(p_fit))]
    for j, fq in enumerate(form.moved_everywhere(p_fit, moves, points, f)):
        g[j] = (float(np.mean(_comparison(fq, form.signed))) - c0) / hs[j]
    cg = cov @ g if len(p_fit) else g
    tot = float(np.sqrt(max(float(g @ cg), 0.0))) if len(p_fit) else 0.0
    rows = []
    for j in range(len(p_fit)):
        s = float(sd[j])
        imp = float(cg[j] / s) if s > 0 else None
        ps = float(pre_sd[j]) if pre_sd is not None else None
        rows.append([names[j], _round(p_fit[j]), _round(s), _round(pre[j]), _round(ps),
                     _round((p_fit[j] - pre[j]) / ps) if ps and ps > 0 else None,
                     _round(s / ps) if ps and ps > 0 else None,
                     _round(cg[j] / (s * tot)) if s > 0 and tot > 0 else None,
                     _round(imp), _round(-imp) if imp is not None else None])
    return {"mean": _round(c0), "sigma": _round(tot), "rows": rows}



def _reported(corr: dict, result: Any, points, flavour: str, cset: dict, signed: bool):
    """The reported correction at the points in the comparison space: (value, half-band)."""
    from . import outputs as C
    comp = C.components(corr, result, points, flavour, cset)
    if signed:
        return C.signed_variance(comp["value"], comp["lo"], comp["hi"], flavour)
    return comp["value"], 0.5 * np.abs(comp["hi"] - comp["lo"])


def verify(out_dir, dataset_dir, cases: Dict[Tuple[str, str], dict], seed: int = 0,
           log: Optional[Callable[[str], None]] = None, refit: bool = True) -> dict:
    """Every MEASUREMENT.json check for one output directory:
      {"measurements": {id: {"type", "status" ("ok" | "failed"), "why", "p_gof", "chi2", "ndf", ...}},
       "corrections": {"<name>|<flavour>": {"status" ("ok", "failed", "undeclared"), "why", "measurement",
                       "z", "sigma_fit", "sigma_reported", ... or "refit_why", "params"}}}
    ``cases``: {(name, flavour): {"points", "space"}}, where the comparison is taken.  ``refit`` False: checks only.
    A budget ringing while reading or refitting fails the unfinished measurements (NOT_IN_BUDGET,
    "budget_exceeded"); elsewhere it propagates."""
    from . import outputs as C
    log = log or (lambda m: None)
    t0 = time.time()
    out_dir, dataset_dir = Path(out_dir), Path(dataset_dir)
    request = json.loads((dataset_dir / "request.json").read_text())
    entries = C.requested(request)
    wanted = [(n, fl) for n, e in sorted(entries.items()) for fl in sorted(((e or {}).get("domain") or {}))]
    res: Dict[str, Any] = {"measurements": {}, "corrections": {}, "seconds": None, "refit": bool(refit)}

    def fail(key, why, mid=None):
        res["corrections"][f"{key[0]}|{key[1]}"] = {"status": "failed", "why": why, "measurement": mid}

    def undeclared(key, why):
        res["corrections"][f"{key[0]}|{key[1]}"] = {"status": "undeclared", "why": why, "measurement": None}

    cset, why_set = C.read_set(out_dir / C.FILE)
    try:
        result = json.loads(read_output(out_dir / "RESULT.json", max_bytes=C.MAX_BYTES))
    except (OSError, ValueError, RecursionError):
        result = None
    spec, arrays, why = read(out_dir)
    if why or cset is None:
        for key in wanted:
            (undeclared if cset is not None and not (out_dir / FILE).exists() else fail)(
                key, why or f"{C.FILE}: {why_set}")
        res["seconds"] = time.time() - t0
        return res
    parsed, problems = parse(spec, arrays, request)
    users: Dict[str, List[Tuple[str, str]]] = {}
    for key in wanted:
        if key in problems:
            (undeclared if left_out(spec, *key) else fail)(key, problems[key])
        else:
            users.setdefault(spec["corrections"][key[0]][key[1]], []).append(key)
    try:
        objs, list_whys = read_measurement_pairs(dataset_dir, parsed, users, entries, log)
    except BudgetExceeded as e:
        log(f"{e}: reading the measurements' pairs")
        res["budget_exceeded"] = f"{e} (reading the measurements' pairs)"
        for mid, keys in users.items():
            res["measurements"][mid] = {"type": parsed[mid]["type"], "status": "failed", "why": NOT_IN_BUDGET}
            for key in keys:
                fail(key, f"its measurement {mid!r} {NOT_IN_BUDGET}", mid)
        res["seconds"] = time.time() - t0
        return res
    for got in objs.values():
        for o in got.values():
            o["unit"] = copies(o)
    global _JOB
    _JOB = {"parsed": parsed, "users": users, "entries": entries, "cset": cset, "result": result, "cases": cases,
            "objs": objs, "list_whys": list_whys, "seed": seed, "log": log, "refit": bool(refit)}
    order = sorted(users, key=lambda m: (parsed[m]["type"] != "lineshape", m))
    done = []
    try:
        for got in pool_imap(_measurement_job, order, MEASURE_JOBS, "fork", log):
            done.append(got)
    except BudgetExceeded as e:
        late = [m for m in order if m not in {d[0] for d in done}]
        log(f"{e}: {len(late)} measurement(s) not verified: {', '.join(late)}")
        res["budget_exceeded"] = f"{e} (refitting the measurements)"
        done += [(mid, {"type": parsed[mid]["type"], "status": "failed", "why": NOT_IN_BUDGET},
                  {f"{n}|{fl}": {"status": "failed", "why": f"its measurement {mid!r} {NOT_IN_BUDGET}",
                                 "measurement": mid} for n, fl in users[mid]}) for mid in late]
    finally:
        _JOB = None
    for mid, row, entries_out in sorted(done):
        res["measurements"][mid] = row
        res["corrections"].update(entries_out)
    res["seconds"] = time.time() - t0
    return res


MEASURE_JOBS = 4
_JOB: Optional[dict] = None


def _measurement_job(mid: str):
    """One measurement's verification and refit comparisons, in a forked worker of verify(): (id, row,
    {"<name>|<flavour>": entry})."""
    from . import outputs as C
    j = _JOB
    pm, keys, log = j["parsed"][mid], j["users"][mid], j["log"]
    log(f"measurement {mid} ({pm['type']}, {', '.join(f'{n}|{fl}' for n, fl in keys)})")
    out: Dict[str, dict] = {}
    try:
        row, fits = _verify_one(mid, pm, keys, j["entries"], j["cset"], j["objs"], j["list_whys"].get(pm["list"]),
                                j["seed"], log, j["refit"])
    except C.CorrectionError as e:
        row, fits = {"type": pm["type"], "status": "failed", "why": f"a correction it uses {e}"}, {}
    except Exception as e:  # noqa: BLE001
        row, fits = {"type": pm["type"], "status": "failed",
                     "why": f"could not be refitted ({type(e).__name__})"}, {}
    for key in keys:
        k = f"{key[0]}|{key[1]}"
        if row["status"] != "ok":
            out[k] = {"status": "failed", "why": f"its measurement {mid!r} {row['why']}", "measurement": mid}
            continue
        entry = {"status": "ok", "why": None, "measurement": mid}
        if key not in fits:
            out[k] = entry
            continue
        form, p_fit, cov, free = fits[key]
        case = j["cases"].get(key)
        corr, _ = C.find(j["cset"], key[0])
        if case is not None:
            try:
                rv, rs = _reported(corr, j["result"], case["points"], key[1], j["cset"], form.signed)
                t = refit_comparison(form, p_fit, cov, case["points"], rv, rs, free)
            except C.CorrectionError as e:
                t = {"why": f"is not evaluable at its points: {e}"}
            if "why" in t:
                entry["refit_why"] = t["why"]
            else:
                entry.update(t)
            try:
                entry["params"] = parameter_table(form, p_fit, cov, case["points"], j["result"], j["cset"])
            except Exception as e:  # noqa: BLE001
                entry["params"] = {"why": type(e).__name__}
        out[k] = entry
    return mid, row, out


def _verify_one(mid: str, pm: dict, keys, entries, cset, objs: Dict[str, Dict[str, dict]], list_why: Optional[str],
                seed: int, log, refit: bool = True):
    """(row, {(name, flavour): (form, refitted parameters, covariance, unconstrained directions)}); without
    ``refit`` the checks only."""
    from . import outputs as C
    t0 = time.time()
    n_pairs = {s: int(len(pm["pairs"][s]["entry"])) for s in SAMPLES}
    n_copies = {s: int(len(u) - len(np.unique(u))) for s in SAMPLES
                if (u := (objs.get(s) or {}).get(pm["list"], {}).get("unit")) is not None}
    if pm["type"] == "efficiency":
        (name, flavour), = keys
        corr, why = C.find(cset, name)
        if corr is None:
            return {"type": "efficiency", "status": "failed", "why": f"serves {name}, which is {why}"}, {}
        why = list_why or recount_efficiency(pm, mid) or check_pairs_efficiency(pm, mid, entries[name], flavour, objs)
        if why:
            return {"type": "efficiency", "status": "failed", "why": why, "n_pairs": n_pairs}, {}
        if not refit:
            return {"type": "efficiency", "status": "ok", "why": None, "n_pairs": n_pairs, "n_copies": n_copies,
                    "seconds": time.time() - t0}, {}
        form = _Form(corr, flavour, entries[name]["domain"][flavour])
        fit = refit_efficiency(form, pm, mid, _objects_float(objs["sim"][pm["list"]]), seed=seed,
                               units={s: objs[s][pm["list"]]["unit"] for s in SAMPLES})
        log(f"  refit: chi2 {fit['chi2']:.1f} / ndf {fit['ndf']}, p {fit['p_gof']} ({time.time() - t0:.0f} s)")
        return ({"type": "efficiency", "status": "ok", "why": None, "p_gof": fit["p_gof"], "chi2": fit["chi2"],
                 "ndf": fit["ndf"], "n_bins": fit["n_bins"], "split_depth": fit["depth"],
                 "n_parameters": int(len(fit["p"])), "converged": fit["converged"], "n_pairs": n_pairs,
                 "n_copies": n_copies, "seconds": time.time() - t0},
                {(name, flavour): (form, fit["p"], fit["cov"], fit["free"])})
    flavour = pm["flavour"]
    forms = {}
    for c in pm["corrections"]:
        corr, why = C.find(cset, c)
        if corr is None:
            return {"type": "lineshape", "status": "failed", "why": f"lists {c}, which is {why}"}, {}
        forms[c] = _Form(corr, flavour, entries[c]["domain"][flavour], signed=entries[c]["kind"] == "resolution")
    data, sim = ((None, None) if list_why else
                 (_objects_float(objs["data"].get(pm["list"], {})), _objects_float(objs["sim"].get(pm["list"], {}))))
    why = list_why or lineshape_objects(pm, objs) or recount_lineshape(pm, data)[1]
    if why:
        return {"type": "lineshape", "status": "failed", "why": why, "n_pairs": n_pairs}, {}
    if not refit:
        return {"type": "lineshape", "status": "ok", "why": None, "n_pairs": n_pairs, "n_copies": n_copies,
                "seconds": time.time() - t0}, {}
    ls = _Lineshape(pm, forms, sim, pm["pairs"]["sim"]["weight"], data, seed)
    fit = refit_lineshape(ls, log=log)
    cov = lineshape_covariance(ls, fit)
    depth = 1
    g = lineshape_gof(ls, fit["p"], split_histograms(pm, data, 1)[0], seed=seed)
    while g["ndf"] <= 0 and depth < SPLIT_MAX_DEPTH:
        hists, admissible = split_histograms(pm, data, depth + 1)
        if not admissible:
            break
        depth += 1
        g = lineshape_gof(ls, fit["p"], hists, seed=seed)
    p_gof = g["p_gof"] if g["ndf"] > 0 else None
    log(f"  refit: moments chi2 {fit['chi2']:.1f} / {fit['n_moments'] - fit['rank']}; histograms chi2 "
        f"{g['chi2']:.1f} / ndf {g['ndf']}, p {p_gof} ({time.time() - t0:.0f} s)")
    row = {"type": "lineshape", "status": "ok", "why": None, "p_gof": p_gof, "chi2": g["chi2"], "ndf": g["ndf"],
           "split_depth": depth, "n_cells": g["n_cells"], "moments_chi2": fit["chi2"], "n_moments": fit["n_moments"],
           "n_parameters": int(len(fit["p"])), "lm": fit["lm"], "lm_k": fit["lm_k"], "lm_m": fit["lm_m"],
           "n_pairs": n_pairs, "n_copies": n_copies,
           "seconds": time.time() - t0}
    out = {}
    for key in keys:
        b = ls.block(key[0])
        free = [u[b] for u in fit["free"] if np.linalg.norm(u[b]) >= FREE_TOL]
        out[key] = (forms[key[0]], fit["p"][b], cov[b, b], free)
    row["n_free"] = len(fit["free"])
    return row, out
