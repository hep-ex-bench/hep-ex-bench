"""The scan_likelihood kind's POI scaling of MODEL.json (participant-shipped): the gate's rules on
`_meta.poi_scaling` beyond its grammar (sbh/kinds/binned_likelihood/model.py poi_scaling_entries, the one parser
every reader of the table uses), each scaled sample's factor at the sizing point, and `sbh check`'s refit of the
model with its scaling (refit: pyhf's per-sample rates times the scaling, fitted with scipy; a pyhf fit alone would
read the scaled samples as a plain product of normfactors).
"""
from __future__ import annotations

import math
import time
from collections import Counter
from typing import Dict, List, Tuple

REFIT_MAX_FREE = 20
REFIT_POINTS = 40
REFIT_BISECT = 8
REFIT_SCAN_S = 120.0
RATE_MIN = 1e-9


def table(spec: dict):
    """MODEL.json's `_meta.poi_scaling` ({} when absent)."""
    meta = spec.get("_meta") if isinstance(spec, dict) else None
    return meta.get("poi_scaling", {}) if isinstance(meta, dict) else {}


def problems(spec: dict, task) -> List[str]:
    """The gate's rules on the POI scaling of a workspace in pyhf's shape (texts; check_model_contract refuses any
    other shape first and makes these contract failures): every sample's data finite; every modifier of a type the
    evaluator's engine reads (model.ENGINE_MODIFIERS: its grader fits no model with pyhf, so it refuses any other);
    `_meta.poi_scaling` an object {channel: {sample: entry}} in the grammar of poi_scaling_entries naming only the
    task's POIs; a scaled sample's normfactors among the task's POIs exactly its markers (the grammar sees only the
    entry's own keys); every task POI a normfactor of the model; no sample carrying one modifier (name and type)
    twice (pyhf applies a repeated normfactor once); a scaled sample's nominal >= 0; a monomial with a negative power
    on a POI whose range reaches 0 only inside a min_of that holds a monomial without one (its rate is infinite
    there)."""
    from ..binned_likelihood.model import ENGINE_MODIFIERS, poi_scaling_entries
    from . import poi_bounds
    from ...contract import _finite
    errs = [f"{c['name']}/{s['name']}: a sample's data must be finite numbers"
            for c in spec["channels"] for s in c["samples"] if not all(_finite(x) for x in s["data"])]
    other = {m["type"] for c in spec["channels"] for s in c["samples"] for m in s["modifiers"]} - set(ENGINE_MODIFIERS)
    if other:
        errs.append(f"modifiers of type {', '.join(sorted(other))}: the evaluator reads only "
                    f"{', '.join(ENGINE_MODIFIERS)}")
    tab = table(spec)
    if not (isinstance(tab, dict) and all(isinstance(v, dict) and all(isinstance(e, dict) for e in v.values())
                                          for v in tab.values())):
        return errs + ["_meta.poi_scaling must be an object {channel: {sample: entry}}"]
    try:
        entries = poi_scaling_entries(tab, spec)
    except (ValueError, TypeError, AttributeError) as e:
        return errs + [str(e)]
    pois = set(task.pois)
    samples = {(c.get("name"), s.get("name")): s for c in spec.get("channels") or [] for s in c.get("samples") or []}
    nf = {m.get("name") for s in samples.values() for m in s.get("modifiers") or [] if m.get("type") == "normfactor"}
    if pois - nf:
        errs.append(f"the POIs {sorted(pois - nf)} are no normfactor of the model")
    for (ch, name), s in samples.items():
        mods = Counter((m.get("name"), m.get("type")) for m in s.get("modifiers") or [])
        twice = [k for k, n in mods.items() if n > 1]
        if twice:
            errs.append(f"{ch}/{name} carries a modifier twice: " + ", ".join(f"{n} ({t})" for n, t in twice))
    zero = {p for p in pois if (poi_bounds(task, p)[0] or 0.0) <= 0.0}
    for (ch, name), (_c, monos, markers) in entries.items():
        if set().union(*monos) - pois:
            errs.append(f"poi_scaling {ch}/{name}: {sorted(set().union(*monos) - pois)} is not a POI of the task "
                        f"({sorted(pois)})")
        carried = {m.get("name") for m in samples[(ch, name)].get("modifiers") or []
                   if m.get("type") == "normfactor"} & pois
        if carried != markers:
            errs.append(f"poi_scaling {ch}/{name}: carries the normfactors {sorted(carried)} of the task's POIs, not "
                        f"exactly those with a non-zero power, {sorted(markers)}")
        if min(samples[(ch, name)].get("data") or [0.0]) < 0:
            errs.append(f"poi_scaling {ch}/{name}: a scaled sample's nominal must be >= 0 (a negative contribution "
                        "is a positive template with a negative factor)")
        infinite = [any(p < 0 and q in zero for q, p in m.items()) for m in monos]
        if all(infinite):
            errs.append(f"poi_scaling {ch}/{name}: a negative power on a POI whose range reaches 0 needs a min_of "
                        "with a monomial without one (its rate is infinite at 0)")
    return errs


def sm_scale(spec: dict) -> Dict[Tuple[str, str], float]:
    """{(channel, sample): its factor at the sizing point, every POI at 1} (each monomial is 1 there: the factor c);
    ValueError outside the grammar."""
    from ..binned_likelihood.model import poi_scaling_entries
    return {k: c for k, (c, _m, _mk) in poi_scaling_entries(table(spec), spec).items()}


def _factor(x, c: float, monos) -> float:
    """c x the least of the monomials [[(parameter index, power), ...], ...] at the parameters x: a power 1 is the
    value itself; any other positive power of a value <= 0 is 0 and a negative one +inf."""
    def mono(m):
        if any(pw < 0 and x[i] <= 0 for i, pw in m):
            return math.inf
        out = 1.0
        for i, pw in m:
            out *= x[i] if pw == 1 else (x[i] ** pw if x[i] > 0 else 0.0)
        return out
    return c * min(mono(m) for m in monos)


def refit(spec: dict, task, pyhf) -> dict:
    """MODEL.json refitted with its POI scaling, a diagnostic for `sbh check` (the evaluator's own fit decides).
    The expected rates are pyhf's per sample with the task's POIs at 1, each sample times its scaling (a sample
    without an entry: the product of the POI normfactors it carries, as pyhf reads it), each bin's total at least
    RATE_MIN; the likelihood is their Poisson term plus pyhf's constraint terms, minimised with scipy inside the
    task's POI ranges.  The first POI's profile is scanned (REFIT_POINTS points, upward and downward, the lower fit
    kept; each q = 1 bracket halved REFIT_BISECT times), the best fit is a free fit from its lowest point.
    {"pois": {name: best fit}, "pieces": [[lo, hi], ...] (the first POI's q <= 1 pieces; absent with "note": why
    there was no scan)}, or only {"note"} for a model beyond REFIT_MAX_FREE free parameters."""
    import numpy as np
    from scipy.optimize import minimize
    from ..binned_likelihood.model import poi_scaling_entries
    from . import poi_bounds
    from . import scan as SC
    entries = poi_scaling_entries(table(spec), spec)
    ws = pyhf.Workspace({k: v for k, v in spec.items() if k != "_meta"})
    model = ws.model()
    cfg, tl = model.config, pyhf.tensorlib
    data = np.asarray(ws.data(model), float)
    main, aux = data[:cfg.nmaindata], data[cfg.nmaindata:]
    idx = {P: cfg.par_slice(P).start for P in task.pois}
    bounds = [list(map(float, b)) for b in cfg.suggested_bounds()]
    for P, i in idx.items():
        lo, hi = poi_bounds(task, P)
        bounds[i] = [bounds[i][0] if lo is None else lo, bounds[i][1] if hi is None else hi]
    fixed = np.array(cfg.suggested_fixed(), bool)
    fixed[list(idx.values())] = False
    if int((~fixed).sum()) > REFIT_MAX_FREE:
        return {"note": f"not refitted here: {int((~fixed).sum())} free parameters (the refit runs up to "
                        f"{REFIT_MAX_FREE})"}
    x0 = np.clip(np.asarray(cfg.suggested_init(), float), [b[0] for b in bounds], [b[1] for b in bounds])
    off = np.cumsum([0] + [cfg.channel_nbins[c] for c in cfg.channels])
    rows = []
    for ci, ch in enumerate(cfg.channels):
        for smp in next(c for c in spec["channels"] if c["name"] == ch)["samples"]:
            ent = entries.get((ch, smp["name"]))
            if ent is not None:
                c, monos = ent[0], [[(idx[P], pw) for P, pw in m.items() if pw != 0] for m in ent[1]]
            else:
                c, monos = 1.0, [[(idx[m["name"]], 1) for m in smp["modifiers"]
                                  if m["type"] == "normfactor" and m["name"] in idx]]
            if ent is not None or monos != [[]]:
                rows.append((cfg.samples.index(smp["name"]), slice(off[ci], off[ci + 1]), c, monos))
    pidx = list(idx.values())

    def nll(x):
        unit = np.array(x, float)
        unit[pidx] = 1.0
        per = np.asarray(tl.tolist(model.main_model.expected_data(tl.astensor(unit), return_by_sample=True)), float)
        for i, b, c, monos in rows:
            per[i, b] *= _factor(x, c, monos)
        nu = np.maximum(per.sum(axis=0), RATE_MIN)
        cons = float(np.ravel(tl.tolist(model.constraint_logpdf(tl.astensor(aux), tl.astensor(x))))[0]) if len(aux) else 0.0
        return float(np.sum(nu - main * np.log(nu))) - cons

    def fit(start, pin=None):
        """(nll, parameters) of the fit from ``start`` with the first POI held at ``pin`` (None: free)."""
        bnd = [(v, v) if f else tuple(b) for v, f, b in zip(start, fixed, bounds)]
        start = np.array(start, float)
        if pin is not None:
            start[i_s], bnd[i_s] = pin, (pin, pin)
        r = minimize(nll, start, method="L-BFGS-B", bounds=bnd)
        return float(r.fun), np.asarray(r.x, float)

    i_s = idx[task.pois[0]]
    lo, hi = bounds[i_s]
    t0 = time.monotonic()
    best = fit(x0)
    cost = (time.monotonic() - t0) * 2 * REFIT_POINTS
    if cost > REFIT_SCAN_S:
        return {"pois": {P: float(best[1][i]) for P, i in idx.items()},
                "note": f"a fit from the model's init only: the scan of {task.pois[0]} would take about {cost:.0f} s "
                        "here"}
    grid = [lo + (hi - lo) * (k / REFIT_POINTS) ** 2 for k in range(REFIT_POINTS + 1)]
    at: Dict[float, tuple] = {}
    for order in (grid, grid[::-1]):
        prev = best[1]
        for v in order:
            r = fit(prev, v)
            if v not in at or r[0] < at[v][0]:
                at[v] = r
            prev = r[1]
    v_low = min(at, key=lambda v: at[v][0])
    free = fit(at[v_low][1])
    best = free if free[0] <= at[v_low][0] else at[v_low]
    at[float(best[1][i_s])] = best
    for _ in range(REFIT_BISECT):
        xs = sorted(at)
        for a, b in zip(xs, xs[1:]):
            if (2.0 * (at[a][0] - best[0]) - 1.0) * (2.0 * (at[b][0] - best[0]) - 1.0) < 0:
                at[0.5 * (a + b)] = fit(at[a if at[a][0] < at[b][0] else b][1], 0.5 * (a + b))
    xs = sorted(at)
    qs = [max(2.0 * (at[v][0] - best[0]), 0.0) for v in xs]
    pieces = [[xs[0] if pc["lo"] is None else pc["lo"], xs[-1] if pc["hi"] is None else pc["hi"]]
              for pc in SC.pieces_of(xs, qs, lo, hi)]
    return {"pois": {P: float(best[1][i]) for P, i in idx.items()}, "pieces": pieces}


def refit_lines(model_path, result, task, pylib=None) -> List[str]:
    """`sbh check`'s lines on MODEL.json refitted with its POI scaling (refit) beside RESULT.json's numbers: every
    POI's best fit and the first POI's 68 % pieces, the model's then the reported ones."""
    from ...contract import load_output_json
    from ..binned_likelihood.model import import_pyhf
    scanned = task.pois[0]

    def show(pieces):
        return " ∪ ".join(f"[{a:.4g}, {b:.4g}]" for a, b in pieces)
    try:
        r = refit(load_output_json(model_path), task, import_pyhf(pylib))
    except Exception as e:  # noqa: BLE001
        return [f"refit with the POI scaling: not available here ({type(e).__name__}: {str(e).splitlines()[0][:160]})"]
    if "pois" not in r:
        return [f"refit with the POI scaling: {r['note']}"]
    rep = {x.get("name"): x for x in (result or {}).get("pois") or [] if isinstance(x, dict)}
    own = rep.get(scanned) or {}
    own_pieces = own.get("intervals") or ([[own["value"] - own["err_minus"], own["value"] + own["err_plus"]]]
                                          if all(k in own for k in ("value", "err_minus", "err_plus")) else [])
    return ["refit with the POI scaling (a diagnostic): " + ", ".join(f"{P} = {v:.4g}" for P, v in r["pois"].items())
            + (f"; {scanned}'s 68 % set {show(r['pieces'])}" if "pieces" in r else f" ({r['note']})"),
            "RESULT.json reports:                       " + ", ".join(f"{P} = {rep[P]['value']:.4g}" for P in r["pois"]
                                                                       if P in rep and "value" in rep[P])
            + (f"; {scanned}'s 68 % set {show(own_pieces)}" if own_pieces else "")]
