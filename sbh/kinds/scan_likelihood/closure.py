"""The closure of a scan_likelihood task: likelihood pulls from the grader's scans (evaluator only).

The linear pull (value - truth) / sigma breaks down at a physical boundary and on an asymmetric profile: a dataset
whose data prefer the boundary has a lower error near 0 and a pull of many sigma for an ordinary fluctuation.  The
likelihood pull sign(value - truth) sqrt(q(truth)), q read off the grader's scan of the dataset's model, judges the
same dataset by its likelihood and is a standard normal under Wilks' theorem.  The closure grades bias, width and
max |pull| on it; the linear-pull verdict is kept beside it as a diagnostic.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from . import poi_bounds
from . import scan as SC


def likelihood_pull(scan, value: float, truth: float, lo: Optional[float] = None,
                    hi: Optional[float] = None) -> Optional[float]:
    """sign(value - truth) sqrt(q(truth)), q linear between scan points and measured from the scan's minimum; +-inf
    when the truth lies outside the scanned range, None for an unreadable scan."""
    try:
        xs, qs = SC.points(scan, lo, hi)
    except ValueError:
        return None
    x, q = np.asarray(xs, float), np.asarray(qs, float)
    sign = 1.0 if value >= truth else -1.0
    if truth < x[0] or truth > x[-1]:
        return sign * float("inf")
    qt = float(np.interp(truth, x, q)) - float(q.min())
    return sign * math.sqrt(max(qt, 0.0))


def likelihood_pulls(records: List[dict], truths: Dict[str, dict], task, scans: Dict[str, Optional[dict]],
                     max_pull: float = 3.5) -> Dict[str, dict]:
    """For the scanned POI (the task's first) over the completed datasets: the likelihood pulls of the records'
    values on ``scans`` ({seed: the dataset's scan in scan.points' form, None where there is none: no pull}) and the
    closure gates on them, with metrics.py's tolerances: bias |mean| <= 3 cluster-robust SE, width |sd - 1| <=
    max(0.25, 3 / sqrt(2 (n - 1))), max |pull|; coverage (|pull| <= 1) as a diagnostic."""
    from ...score import _cluster_se, cluster_of, record_ok, scoring_truth
    out = {}
    for p in task.pois[:1]:
        lo, hi = poi_bounds(task, p)
        pulls, clusters, seeds = [], [], []
        for rec in records:
            t = truths[rec["seed"]]
            if not record_ok(rec) or p not in scoring_truth(task, t):
                continue
            a = next((SC.interval(x) for x in rec["result"]["pois"] if x.get("name") == p), None)
            lp = likelihood_pull(scans.get(rec["seed"]), a[0], float(t[p]), lo, hi) if a is not None else None
            if lp is None:
                continue
            pulls.append(lp)
            clusters.append(cluster_of(task, t, rec["seed"]))
            seeds.append(rec["seed"])
        if not pulls:
            continue
        d = np.array(pulls)
        fin = np.isfinite(d)
        n = int(fin.sum())
        mean = float(d[fin].mean()) if n else float("inf")
        sd = float(d[fin].std(ddof=1)) if n > 1 else float("inf")
        se = _cluster_se(d[fin], [c for c, f in zip(clusters, fin) if f]) if n else float("inf")
        z = float(mean / se) if se > 0 else float("inf")
        width_tol = max(0.25, 3.0 / math.sqrt(2.0 * max(n - 1, 1)))
        mx = float(np.max(np.abs(d)))
        out[p] = {"n": len(pulls), "n_outside_scan": int((~fin).sum()), "mean": mean, "width": sd,
                  "se_cluster": se, "z": z, "coverage": float(np.mean(np.abs(d) <= 1.0)), "max_abs": mx,
                  "bias_pass": bool(n >= 3 and abs(z) <= 3.0), "width_pass": bool(n > 1 and abs(sd - 1.0) <= width_tol),
                  "max_pull": max_pull, "max_pass": bool(mx <= max_pull),
                  "per_seed": dict(zip(seeds, [float(x) for x in d]))}
    return out


def closure_criteria(pub: dict, priv: dict, task, a, offsets=None, scored_calibration=None):
    """Kind.closure_criteria of the kind: the core's closure criteria with the scanned POI's gates and max |pull| on
    its likelihood pulls: recovery, the slope gate of every POI, the likelihood-pull bias and width of the scanned POI
    (the core's linear gates for any other POI), max |pull|, and the core's calibration and leakage criteria; and one
    more, that the grader measured the scanned POI's density of the reference on every dataset: its H (its mean -ln
    p) is every run's baseline there (closure.json reference_pois).  The linear-pull verdict goes to closure.json as a
    diagnostic, and its "scan_baseline" records the grader's baseline: {"source": "grader" where it holds, else
    "pending", "n", "n_datasets", "missing": the datasets without one, "scan_version"}."""
    from ...cli import POI_MAX_PULL, _bias_ok, _pass_criteria
    from ...score import entropy_ref_loss
    from .verify import SCAN_VERSION
    max_pull = a.max_pull if a.max_pull is not None else POI_MAX_PULL
    kw = dict(offsets=offsets, max_leakage=a.max_leakage,
              leakage_nsigma=a.leakage_nsigma if a.leakage_nsigma is not None else 2.0,
              leakage_se_cap=a.leakage_se_cap, scored_calibration=scored_calibration,
              calibration_slope_tol=a.calibration_slope_tol if a.calibration_slope_tol is not None else 0.25)
    linear, lin_mp = _pass_criteria(pub, priv, max_pull, **kw)
    lp = priv.get("likelihood_pulls") or {}
    if offsets is not None or not lp:
        return None
    others = [p for p in pub["pois"] if p not in lp]
    other_pulls = [abs(v) for s in priv["per_seed"] for p, v in s["pulls"].items() if p in others and v is not None]
    mp = max([m["max_abs"] for m in lp.values()] + other_pulls, default=float("inf"))
    gates = all(pub["pois"][p]["sensitivity_pass"] for p in pub["pois"]) and all(
        m["bias_pass"] and m["width_pass"] for m in lp.values()) and all(
        pub["pois"][p]["calibration_pass"] for p in others) and _bias_ok(pub, others)
    crit = {"recovery == 1": pub["recovery"] == 1.0,
            "all POI gates pass (slope; likelihood-pull bias and width)": gates,
            f"max |likelihood pull| <= {max_pull}": mp <= max_pull}
    crit.update({k: v for k, v in linear.items()
                 if k not in ("recovery == 1", "all POI gates pass", f"max |pull| <= {max_pull}")})
    for p, m in lp.items():
        print(f"{p}: likelihood pulls n={m['n']} mean={m['mean']:+.2f} width={m['width']:.2f} "
              f"coverage={m['coverage']:.2f} max|pull|={m['max_abs']:.2f} (cluster z={m['z']:+.2f}); "
              f"linear-pull verdict {'PASS' if all(linear.values()) else 'FAIL'} (max|pull| {lin_mp:.2f}), "
              "a diagnostic")
    fields = {"pull_definition": "likelihood",
              "likelihood_pulls": {p: {k: v for k, v in m.items() if k != "per_seed"} for p, m in lp.items()},
              "linear_pull_diagnostic": {"criteria": linear, "max_abs_pull": lin_mp}}
    poi, ref = task.pois[0], priv.get("reference_pois") or {}
    missing = sorted(s for s, e in ref.items() if entropy_ref_loss((e or {}).get(poi)) is None)
    measured = bool(ref) and not missing
    crit[f"the grader measured {poi}'s density on every dataset"] = measured
    fields["scan_baseline"] = {"source": "grader" if measured else "pending", "n": len(ref) - len(missing),
                               "n_datasets": len(ref), "missing": missing, "scan_version": SCAN_VERSION}
    return crit, mp, fields
