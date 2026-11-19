"""The scan_likelihood kind's scoring -- EVALUATOR-ONLY: the physics cases from the grader's measurement (verify.py's
per-dataset records), RESULT.json's truthfulness and other claims against it, the declared model's systematics, its
GOF, and the closure's gate.

- load_rows: the run's per-dataset grader records; one graded under another scan rule stops the score.
- score_inputs: per completed dataset the grader's status (binned_likelihood scoring._grader_status).  ok: the
  scanned POI's entry is the grader's best fit and main piece, its scan density (density.scan_density of the
  grader's scan, a ScanForecast) and every q <= 1 piece ("intervals"); every other POI's the grader's best fit
  and q = 1 interval (a POI without one is a failed case).  failed (the model's: verify.py): no entries (failed
  cases).  pending (the evaluator's): the reported entries, provisionally, except on an evaluator fault (a record
  carrying "evaluator_error", or a stored record graded before the grader marked its own errors whose status is a
  "verify error", sbh.score.REGRADE: no entries, the cases left out); under Task.grader_strict failed, except an
  evaluator fault and a completed dataset with no grader record (the evaluation lost its grading,
  sbh.score.NO_GRADER_RECORD), which stay left out (the evaluator's, never the model's: strict fails only what the
  contract fails).  The kind's contract states no case without a measurement on a completed dataset: a case the
  grader cannot give fails its dataset.
- case_loss: the scanned POI's density against the reference's H (its density's mean -ln p); pending, the two-piece
  loss of its reported main piece against that same baseline; any other POI the two-piece loss (density.py).
- closure_gate: the grader must have measured the reference's scanned POI on every dataset (its density's H
  is every run's baseline there; closure.closure_criteria) before any other run is scored on the suite.
- truth_cases (Delta_truth): the scanned POI against the grader's scan q, every other POI against its interval
  (scan_truth, interval_truth).
- claims_rows (Delta_claims): gof_pvalue and coverage as the grader's cross-checks say; each POI's reported stat /
  syst sides against the grader's stat-only interval of its main piece.
- score_terms: binned_likelihood's terms on these (a GOF case left out on an evaluator fault named as such; a
  systemic evaluator fault and one a rerun reproduced named apart, both decided over the completed datasets' records
  as they stand, binned_likelihood scoring.fault_split, whatever order the parallel post stage graded them in); the
  systematics probes size each modifier's net per-bin change with the scaled samples at the sizing point
  (scaling.sm_scale), so a change and its negative copy cancel.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ...score import EVALUATOR_FAULT, NO_GRADER_RECORD, cluster_of, record_ok
from ..binned_likelihood.scoring import TRUTH_CAP, TRUTH_RHO, TRUTH_TAU_ABS, TRUTH_TAU_REL, _finite_p
from . import VERIFY_FILE, poi_bounds
from . import scan as SC
from .verify import SCAN_VERSION, closed

FIT_FIELDS = ("value", "err_minus", "err_plus", "sigma_hess", "stat_minus", "stat_plus")
SCAN_DIAGNOSTICS = ("n_failed", "n_sweep_disagree")


def load_rows(run_dir: Path) -> Dict[str, dict]:
    """{seed: grader record} of the run's datasets that have one (runs/<seed>/VERIFY_FILE).  A record graded under
    another scan rule (its scan_version) stops the reader: the post stage grades the run again."""
    rows = {}
    for p in sorted(Path(run_dir).glob(f"runs/seed_*/{VERIFY_FILE}")):
        try:
            row = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if row.get("scan_version") != SCAN_VERSION:
            raise SystemExit(f"{p} was graded under the scan rule {row.get('scan_version')!r}, this evaluator reads "
                             f"{SCAN_VERSION!r}: run the post stage on the run again (--phase post)")
        rows[p.parent.name] = row
    return rows


def verification(run_dir: Path):
    """Kind.verification of the kind: (VERIFY_FILE, {"per_seed": the run's grader records, "n_seeds",
    "n_with_model"}); None for a run without one."""
    rows = list(load_rows(run_dir).values())
    if not rows:
        return None
    return VERIFY_FILE, {"per_seed": rows, "n_seeds": len(rows),
                         "n_with_model": sum(not str(r.get("status") or "").startswith(("no MODEL", "invalid"))
                                             for r in rows)}


def _reported(rec: dict) -> Dict[str, dict]:
    return {x.get("name"): x for x in (rec.get("result") or {}).get("pois") or [] if isinstance(x, dict)}


def score_inputs(run_dir, records, truths, task) -> dict:
    """Kind.score_inputs of the kind (module docstring): {"records", "source": "model", "info": {seed: {poi: {"status",
    "reason", the grader's FIT_FIELDS and, for the scanned POI, its "pieces" and SCAN_DIAGNOSTICS}}},
    "verify_rows"}."""
    from ..binned_likelihood.scoring import _grader_status
    from .density import ScanForecast, scan_density
    rows = load_rows(run_dir)
    scanned = task.pois[0]
    strict = bool(getattr(task, "grader_strict", False))
    out, info = [], {}
    for rec in records:
        if not record_ok(rec):
            out.append(rec)
            continue
        row = rows.get(rec["seed"])
        st, why = _grader_status(row)
        fault = st == "pending" and why.startswith(EVALUATOR_FAULT)
        lost = row is None and strict
        if lost:
            why = NO_GRADER_RECORD
        elif st == "pending" and strict and not fault:
            st, why = "failed", f"strict: {why}"
        rep = _reported(rec)
        m, pois = {}, []
        for p in task.pois:
            if st == "pending":
                m[p] = {"status": st, "reason": why, **({"evaluator_fault": True} if fault else {})}
                if p in rep and not (fault or lost):
                    pois.append({k: rep[p][k] for k in ("name", "value", "err_minus", "err_plus", "stat", "syst",
                                                        "intervals") if k in rep[p]})
                continue
            pf = (row or {}).get("poi_fits", {}).get(p) if st == "ok" else None
            entry = {"name": p, "value": pf["value"], "err_minus": pf["err_minus"],
                     "err_plus": pf["err_plus"]} if pf else None
            if entry is None or SC.interval(entry) is None:
                m[p] = {"status": "failed", "reason": why or "the model gives no q = 1 interval"}
                continue
            m[p] = {"status": "ok", "reason": "", **{k: pf.get(k) for k in FIT_FIELDS}}
            if p == scanned:
                lo, hi = poi_bounds(task, p)
                xs = [x for x, _ in row["scan"]["points"]]
                pieces = [[pc["lo"], pc["hi"]] for pc in (closed(pc, xs) for pc in row["scan"]["pieces"])]
                m[p].update(pieces=pieces, **{k: row["scan"].get(k) for k in SCAN_DIAGNOSTICS})
                entry.update(intervals=pieces, forecast=ScanForecast(scan_density({"points": row["scan"]["points"]},
                                                                                  lo, hi)))
            pois.append(entry)
        info[rec["seed"]] = m
        out.append(dict(rec, result=dict(rec["result"], pois=pois)))
    return {"records": out, "source": "model", "info": info, "verify_rows": rows}


def case_loss(p, poi, truth, task, fallback, ref, ref_sigma) -> dict:
    """Kind.case_loss of the kind (module docstring).  The fallback forecast is the task's (center, width) as a
    symmetric interval truncated to the POI's range.  The scanned POI's case is excluded where the reference's has
    no H on the dataset, never scored against the scalar sigma_ref: the closure's own run on a dataset its
    grader gave no density for, whose closure then fails and holds every other run's score (closure_gate).  A density
    case's mw_pull is its likelihood pull (sbh.score.likelihood_pull)."""
    from ...score import entropy_ref_loss, forecast_case_loss
    from .density import ScanForecast, interval_case, loss
    lo, hi = poi_bounds(task, poi)
    fc, fs = fallback
    fb_loss = loss(truth, fc, fs, fs, lo, hi)
    if poi != task.pois[0]:
        return interval_case(p, truth, lo, hi, fb_loss, ref, ref_sigma)
    sf = p.get("forecast") if isinstance(p, dict) else None
    if p is None or isinstance(sf, ScanForecast):
        return forecast_case_loss(sf, truth, fallback, ref, ref_sigma, fb_loss=fb_loss)
    cl = interval_case(p, truth, lo, hi, fb_loss, None, 0.0)
    rl = entropy_ref_loss(ref)
    return dict(cl, ref_loss=rl, ref_source="seed" if rl is not None else None)


def scan_truth(a, pieces_rep: List[list], scan: dict, lo: float, hi: float) -> dict:
    """Delta_truth of the scanned POI's report (a = (value, err_minus, err_plus), ``pieces_rep`` its pieces) against
    the grader's scan q (linear between its points): c_val = max(0, sqrt(q(value)) - tau), tau = max(local step,
    TRUTH_TAU_REL s + TRUTH_TAU_ABS) / s, s the main piece's half-width on the report's side; c_set = max(0,
    ln(TRUTH_RHO / c)), c = W(R n M) / W(M) the share of the model's 68 % set M (the scan's q <= 1 pieces) that the
    reported set R covers, W(S) the integral of exp(-q/2) over S.  c depends on the reported set alone (however it is
    cut into pieces), its shortfall W(M \\ R) adds up over the pieces, and a set too wide costs nothing.
    T = min(c_val + c_set, TRUTH_CAP).  A piece's ends are read closed (verify.closed)."""
    xs = np.array([x for x, _ in scan["points"]])
    qs = np.array([q for _, q in scan["points"]])

    def root_q(x):
        return math.sqrt(max(float(np.interp(x, xs, qs)), 0.0))
    x_hat, iv = scan["x_hat"], closed(scan["interval"], list(xs))
    s = SC.side_sigma(x_hat, x_hat - iv["lo"], iv["hi"] - x_hat, a[0])
    tau = max(SC.local_step(list(xs), x_hat), TRUTH_TAU_REL * s + TRUTH_TAU_ABS) / s
    c_val = max(0.0, root_q(a[0]) - tau)
    model = [(mp["lo"], mp["hi"]) for mp in (closed(mp, list(xs)) for mp in scan["pieces"])]
    ends = [e for piece in model + [tuple(r) for r in pieces_rep] for e in piece]
    grid = np.unique(np.clip(np.concatenate([np.linspace(lo, hi, 4001), xs, ends]), lo, hi))
    w = np.exp(-0.5 * np.maximum(np.interp(grid, xs, qs), 0.0))
    seg, mid = 0.5 * (w[1:] + w[:-1]) * np.diff(grid), 0.5 * (grid[1:] + grid[:-1])
    in_m = np.zeros(len(mid), bool)
    in_r = np.zeros(len(mid), bool)
    for (r0, r1), inside in [(m, in_m) for m in model] + [(tuple(r), in_r) for r in pieces_rep]:
        inside |= (mid >= r0) & (mid <= r1)
    w_m = float(seg[in_m].sum())
    cover = float(seg[in_m & in_r].sum()) / w_m if w_m > 0 else 1.0
    c_set = min(max(0.0, math.log(TRUTH_RHO / max(cover, 1e-300))), TRUTH_CAP)
    return {"model_sigma": s, "tau": tau, "z": root_q(a[0]), "c_val": c_val, "coverage": cover, "c_set": c_set,
            "T": min(c_val + c_set, TRUTH_CAP)}


def interval_truth(a, mm: dict) -> dict:
    """Delta_truth of a POI's reported interval a = (value, err_minus, err_plus) against the grader's (mm):
    z = |value - x_model| / s, s the model's error on the report's side, tau = TRUTH_TAU_REL + TRUTH_TAU_ABS / s; per
    side with both errors > 0, max(0, ln(TRUTH_RHO s_min / s_rep)), s_min the smaller of the model's side and its
    Hessian width, their mean taken.  T = min(max(0, z - tau) + mean side cost, TRUTH_CAP)."""
    v, x_m = a[0], mm["value"]
    s = SC.side_sigma(x_m, mm["err_minus"], mm["err_plus"], v)
    z = abs(v - x_m) / s
    tau = TRUTH_TAU_REL + TRUTH_TAU_ABS / s
    costs = []
    for rep, mod in ((a[1], mm["err_minus"]), (a[2], mm["err_plus"])):
        if rep > 0 and mod > 0:
            s_min = min(mod, mm["sigma_hess"]) if mm.get("sigma_hess") else mod
            costs.append(max(0.0, math.log(TRUTH_RHO * s_min / rep)))
    c_sig = float(np.mean(costs)) if costs else 0.0
    return {"model_sigma": s, "tau": tau, "z": z, "c_val": max(0.0, z - tau), "c_sig": c_sig,
            "T": min(max(0.0, z - tau) + c_sig, TRUTH_CAP)}


def truth_cases(records, info, rows, truths, task) -> List[dict]:
    """Delta_truth's cases: per scored (dataset, POI) with a grader measurement and a reported interval, scan_truth
    for the scanned POI (all its reported pieces), interval_truth for the others."""
    cases = []
    for rec in records:
        if not record_ok(rec) or rec["seed"] not in info:
            continue
        t, rep = truths[rec["seed"]], _reported(rec)
        for p in task.pois:
            mm = info[rec["seed"]].get(p) or {}
            a = SC.interval(rep.get(p))
            if p not in t or not task.poi_scored_v2(p, t) or mm.get("status") != "ok" or a is None:
                continue
            if p == task.pois[0]:
                pieces = rep[p].get("intervals") or [[a[0] - a[1], a[0] + a[2]]]
                f = scan_truth(a, pieces, rows[rec["seed"]]["scan"], *poi_bounds(task, p))
            else:
                f = interval_truth(a, mm)
            cases.append({"seed": rec["seed"], "poi": p, "reported": a[0], "model": mm["value"], **f,
                          "cluster": cluster_of(task, t, rec["seed"])})
    return cases


def claims_rows(records, rows, info, truths, task) -> List[dict]:
    """Delta_claims' rows: per completed dataset {check: failed?} -- gof_pvalue and coverage as the grader's
    quality_crosscheck says; each POI's reported "stat" / "syst" sides against the grader's stat-only interval
    (syst: sqrt(side^2 - stat side^2)), failed beyond max(0.3 x the model's, 0.1 x its side) on either side."""
    out = []
    for rec in records:
        if not record_ok(rec):
            continue
        row, rep = rows.get(rec["seed"]) or {}, _reported(rec)
        ch = {}
        qc = row.get("quality_crosscheck") if isinstance(row.get("quality_crosscheck"), dict) else {}
        for k in ("gof_pvalue", "coverage"):
            if isinstance(qc.get(k), dict) and qc[k].get("agree") is not None:
                ch[k] = not bool(qc[k]["agree"])
        for p in task.pois:
            mm = (info.get(rec["seed"]) or {}).get(p) or {}
            if mm.get("status") != "ok" or p not in rep:
                continue
            for comp in ("stat", "syst"):
                c = rep[p].get(comp)
                fails = []
                for side in ("minus", "plus"):
                    s_m, st_m = mm.get(f"err_{side}"), mm.get(f"stat_{side}")
                    got = _finite_p(c.get(f"err_{side}")) if isinstance(c, dict) else None
                    if s_m is None or st_m is None or got is None:
                        continue
                    want = st_m if comp == "stat" else math.sqrt(max(s_m ** 2 - st_m ** 2, 0.0))
                    fails.append(abs(got - want) > max(0.3 * want, 0.1 * s_m))
                if fails:
                    ch[f"{p}_{comp}"] = any(fails)
        if ch:
            out.append({"seed": rec["seed"], "checks": ch, "delta": sum(ch.values()) / len(ch),
                        "cluster": cluster_of(task, truths[rec["seed"]], rec["seed"])})
    return out


def score_terms(run_dir, records, truths, task, inputs: dict, v2: dict, leak, yld) -> dict:
    """Kind.score_terms of the kind: binned_likelihood's terms on the grader's numbers (module docstring), the closure's
    likelihood pulls on the grader's scans (closure.likelihood_pulls, the scored records' values)."""
    from ...hep.calibration import calibration_cases
    from ..binned_likelihood import scoring as BS
    from . import scaling
    from .closure import likelihood_pulls
    rows, info = inputs["verify_rows"], inputs["info"]
    strict = bool(getattr(task, "grader_strict", False))
    ungraded = ({rec["seed"] for rec in records if record_ok(rec) and rows.get(rec["seed"]) is not None
                 and rows[rec["seed"]].get("status") != "ok"
                 and not BS.record_fault(rows.get(rec["seed"]))} if strict else None)
    cov = {rec["seed"]: BS.model_coverage(run_dir, rec, task, ungraded, sample_scale_of=scaling.sm_scale,
                                          extra_pois=task.pois, net_size=True) for rec in records}
    cal_cases = calibration_cases(records, truths, task, coverage=cov)
    systc = BS.completeness_cases(run_dir, records, truths, task, ungraded=ungraded, coverage=cov)
    left: Dict[str, str] = {}
    gofc = BS.gof_cases(run_dir, records, combined=True, strict=strict, rows=rows, left_out=left)
    tcases = truth_cases(records, info, rows, truths, task)
    crows = claims_rows(records, rows, info, truths, task)
    gate = BS.calibration_gate(leak, yld, task)
    done = {rec["seed"] for rec in records if record_ok(rec)}
    repro, systemic = BS.fault_split(rows, done)
    terms, pending, extra = BS.v3_terms(v2, cal_cases, gofc, truths, len(records), syst_cases=systc,
                                        truth=BS.truth_term(tcases), claims=BS.claims_term(crows), cal_gate=gate,
                                        rule=BS.FIT_RULE, task=task, gof_left_out=left, reproduced=repro,
                                        claims_faults=BS.claims_stage_faults(rows, done), systemic=systemic)
    scans = {r["seed"]: {"points": rows[r["seed"]]["scan"]["points"]} if (rows.get(r["seed"]) or {}).get("status")
             == "ok" else None for r in records}
    return {"terms": terms, "pending": pending, "extra": extra,
            "private": {"v3_calibration_cases": cal_cases, "v3_completeness_cases": systc, "v3_gof_cases": gofc,
                        "v3_model_coverage": cov, "fit_rule": BS.FIT_RULE, "fit_info": info,
                        "v4_truth_cases": tcases, "v4_claims": crows, "gof_left_out": left or None,
                        "reproduced_faults": repro or None, "systemic_faults": systemic or None,
                        "likelihood_pulls": likelihood_pulls(inputs["records"], truths, task, scans)},
            "gof_cases": gofc, "calibration_response": {"gate": gate}}


def reference_entry_of(v: dict, scored: Optional[dict]) -> dict:
    """The entry whose interval the closure keeps of the reference on a dataset: the grader's (``scored``) where it
    measured one, else the reported ``v`` (pending)."""
    return scored if isinstance(scored, dict) and SC.interval(scored) is not None else v


def closure_gate(suite_dir, task, label: str) -> Optional[str]:
    """Kind.closure_gate of the kind: every run but the closure's reference is scored against the grader's density
    of the reference on every dataset (closure.json "scan_baseline", closure.closure_criteria), made under this
    evaluator's scan rule.  A dataset without one would drop out of every run's cases, so no run is scored before it
    is there."""
    from ...suite import CLOSURE_LABEL
    if label == CLOSURE_LABEL:
        return None
    p, poi = Path(suite_dir) / "closure.json", task.pois[0]
    redo = "sbh closure --phase post, then sbh closure --phase score"
    try:
        sb = json.loads(p.read_text()).get("scan_baseline") or {}
    except FileNotFoundError:
        return (f"no closure.json in {Path(suite_dir).name}: {task.name} scores {poi} against the grader's density "
                "of the closure's reference (sbh closure)")
    except (OSError, ValueError) as e:
        return f"{p} is unreadable ({type(e).__name__})"
    if sb.get("scan_version") != SCAN_VERSION:
        return (f"{p}'s grader baseline of {poi} was scanned under {sb.get('scan_version')!r}, this evaluator reads "
                f"{SCAN_VERSION!r}: {redo}")
    if sb.get("source") != "grader":
        return (f"{p} holds no grader density of {poi} for the reference on "
                f"{', '.join(sb.get('missing') or []) or 'its datasets'}: {redo}")
    return None
