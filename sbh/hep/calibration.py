"""The calibration term (docs/METRICS.md Delta_cal) and nuisance-log diagnostics: the injected detector effects a task
value-scores (Task.calibration_spec), read from the RESULT.json nuisance log and scored against the injected truth.

Delta_cal scores each (dataset, value-scored nuisance) with the physics term's per-case charge (sbh.score.case_charge),
one formula for every quantity, the effect's target tau_j in the role of sigma_ref:
    Delta_j = min(max(ln(s_j / tau_j), 0) + (z_j^2 - 1) / 2, C_j),   z_j = (reported - injected) / s_j,
s_j the reported uncertainty, tau_j = target_j the task's target precision: an uncertainty below the target earns
no more precision credit than the target (CAL_PRECISION_FLOOR; a case is capped above, so an uncapped credit would
pay for a tiny uncertainty on a guessed value), capped at
C_j = max(max(F_j, 0), CAL_CAP_FLOOR); F_j is the
charge of the nuisance's no-answer forecast (its nominal value at the task's prior width).  An effect is value-scored only where its prior is wider than its target (value_scored:
prior_j > target_j): measuring it at the target is then more informative than its prior, so logging the nominal value
never beats measuring.  An effect whose prior is no wider than its target cannot be measured usefully at the target
(the prior already says more, and a constant logged without the data could beat a measurement), so it is not
value-scored (held_out); it still enters the calibration gate where the task has one.  Every valid logged entry of a
value-scored effect is scored on its value, modelled or not (the contract: "A correction you apply without a nuisance
parameter is logged the same way"); an unlogged one, or one logged with no uncertainty, costs max(F_j, 0): "don't know" never scores better than the
target precision with an accurate uncertainty.  Delta_cal is the plain mean over the cases (calibration_cases,
calibration_term); the model enters only through the calibration gate, which under score-from-fit withdraws a share
of a case's credit (gated_charge).  A task kept on the rule it shipped with (sbh.score.SHIPPED_RULE) keeps its
calibration rule (_shipped_*)."""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..metrics import score_poi
from ..score import (SHIPPED_RULE, UNBOUNDED_CHARGE, V2_CAP_FLOOR, _cluster_se, _signed_pull, _v2_loss, case_charge,
                     cluster_of, no_measurement_charge, record_ok, score_rule)
from .names import expand_shared

SPLIT_RULE = "nuisance split (calibration logged and modelled; completeness of the unvaried units)"


def nuisance_log(rec: dict) -> dict:
    """A record's nuisance log with shared values expanded ({} for a failed run or malformed log)."""
    log = (rec.get("result") or {}).get("nuisances") if record_ok(rec) else None
    return expand_shared(log) if isinstance(log, dict) else {}


def unread_wp_entries(task, log) -> set:
    """eff_<obj> entries whose "wp" the task does not accept; the scorer reads them as not logged."""
    wps = getattr(task, "working_points", None)
    if not wps:
        return set()
    from .nuisances import declared_working_points
    return {f"eff_{o}" for o, w in declared_working_points(log, wps).items() if w is None}


def calibration_spec(task, log=None) -> Dict[str, Dict[str, float]]:
    """The injected effects under the working points ``log`` declares (fallback: Task.calibration_scored); scored_spec
    keeps those Delta_cal value-scores."""
    f = getattr(task, "calibration_spec", None)
    return f(log) if callable(f) else (getattr(task, "calibration_scored", None) or {})


def nuisance_metrics(records, truths, task, cfg) -> Dict[str, dict]:
    """Recovery diagnostics (slope, bias, pull width) per logged nuisance with an injected truth."""
    out = {}
    names = {}
    for rec in records:
        for n in task.calibration_truth(truths[rec["seed"]], nuisance_log(rec)):
            names.setdefault(n, None)
    for key in names:
        rows = []
        for rec in records:
            if not record_ok(rec):
                continue
            t = task.calibration_truth(truths[rec["seed"]], nuisance_log(rec)).get(key)
            if t is None:
                continue
            claim = _reported(nuisance_log(rec).get(key))
            if claim is not None:
                rows.append((float(t), claim[0], claim[1]))
        if len(rows) < 3:
            continue
        a = np.array(rows)
        pos = a[:, 2][a[:, 2] > 0]
        sigma_ref = float(np.median(pos)) if pos.size else float("nan")
        m = asdict(score_poi(key, a[:, 0], a[:, 1], a[:, 2], sigma_ref, cfg))
        exp = [t for t in (task.calibration_truth(truths[rec["seed"]], nuisance_log(rec)).get(key)
                           for rec in records if record_ok(rec)) if t is not None]
        m["n_expected"] = len(exp)
        m["truth_varies"] = bool(np.ptp(exp) > 0) if exp else m["truth_varies"]
        out[key] = m
    return out


def _reported(entry):
    """(value, unc) of a RESULT.json nuisance-log entry, or None."""
    if not isinstance(entry, dict):
        return None
    try:
        v, u = float(entry["value"]), float(entry.get("unc", 0.0) or 0.0)
    except (TypeError, KeyError, ValueError):
        return None
    return (v, u) if np.isfinite(v) and np.isfinite(u) and u >= 0 else None


CAL_PRECISION_FLOOR = 0.0
CAL_CAP_FLOOR = 4.0
CAL_RULE = ("per (dataset, nuisance) case min(max(ln(s / tau), 0) + (z^2 - 1) / 2, C), z = (reported - injected) / s, s the "
            "reported uncertainty, tau the task's target precision, C = max(unlogged charge, 4); only an effect whose prior is wider than its "
            "target is value-scored; every valid logged entry scored on its value, modelled or not; an unlogged "
            "nuisance, or one logged with no uncertainty, charged max(F, 0), F the charge of the nominal value at "
            "the prior width; Delta_cal the mean over the cases")


def value_scored(sp: Dict[str, float]) -> bool:
    """Whether an effect of the spec (Task.calibration_spec) is value-scored: its prior is wider than its target
    (prior > target).  Then its tau_j is the target, and an accurate measurement there (charge 0) beats logging the
    nominal value at the prior width (precision piece ln(prior / target) > 0) and not logging it.  An effect whose prior
    is no wider than its target is not: a measurement at the target says less than the prior does."""
    return float(sp["prior"]) > float(sp["target"])


def scored_spec(task, log=None) -> Dict[str, Dict[str, float]]:
    """The effects Delta_cal value-scores under the working points ``log`` declares: calibration_spec's with
    value_scored."""
    return {n: sp for n, sp in calibration_spec(task, log).items() if value_scored(sp)}


def held_out(task, log=None) -> Dict[str, Dict[str, float]]:
    """The effects of calibration_spec that Delta_cal does not value-score (prior no wider than the target,
    value_scored); the calibration gate still reads them where the task has one."""
    return {n: sp for n, sp in calibration_spec(task, log).items() if not value_scored(sp)}


def _answer_pieces(value: float, unc: float, truth: float, tau: float) -> Tuple[float, float, float]:
    """(precision piece ln(s / tau), pull piece (z^2 - 1) / 2, z) of a logged value +- s against the injected one.  A
    zero uncertainty states no width: its precision piece -inf, its pull 0 on the exact value and +-inf off it
    (calibration_cases charges such an entry as unlogged)."""
    err = float(value) - float(truth)
    if unc > 0:
        z = _signed_pull(err, unc)
        d_prec = math.log(unc) - math.log(tau)
    else:
        z = 0.0 if err == 0.0 else math.copysign(math.inf, err)
        d_prec = -math.inf
    return d_prec, 0.5 * z * z - 0.5, z


def calibration_cases(records: List[dict], truths: Dict[str, dict], task,
                      coverage: Optional[Dict[str, dict]] = None) -> List[dict]:
    """One case per (dataset, value-scored nuisance: scored_spec, prior wider than the target), its charge Delta_j
    (CAL_RULE, sbh.score.case_charge) with tau_j = the target in the role of sigma_ref: an effect logged (a valid
    nuisance-log entry) is
    "counted" and charged on its value and reported uncertainty, modelled or
    not; an unlogged one (or one logged with no uncertainty, "no_uncertainty") is charged the unlogged charge max(F_j, 0), F_j the case_charge of the nominal value at the
    prior width (the nuisance's no-answer forecast).  ``coverage`` ({seed: model_coverage}) records whether the model
    carries the effect ("modelled": a constrained, unfixed modifier in MODEL.json passing nuisance_probes) and why not
    ("model_fails"), a diagnostic: the model enters the term only through the calibration gate (calibration_term).
    Truth and scored effects follow the working point each log declares; an eff_<obj> entry naming no accepted one
    counts as not logged (``wp_unread``).  A case records logged / modelled (None = not checked) / counted,
    model_fails, value / unc / pull of the logged entry, "target" tau_j (the task's target, also "target_task"), its
    pieces (the fallback's where not counted), the cap C_j, "fallback_charge" F_j, "unlogged" max(F_j, 0) and "delta"
    Delta_j (before the calibration gate, calibration_term).  A task kept on the rule it shipped with
    (sbh.score.SHIPPED_RULE) keeps its cases (_shipped_calibration_cases)."""
    if score_rule(task) == SHIPPED_RULE:
        return _shipped_calibration_cases(records, truths, task, coverage)
    wps = getattr(task, "working_points", None) or {}
    cases = []
    for rec in records:
        t_full = truths[rec["seed"]]
        log = nuisance_log(rec)
        tv_all = task.calibration_truth(t_full, log)
        spec = scored_spec(task, log)
        unread = unread_wp_entries(task, log)
        cov = None if coverage is None else (coverage.get(rec["seed"]) or {})
        for n, sp in spec.items():
            if n not in tv_all:
                continue
            tv = float(tv_all[n])
            nom, tau, pri = float(sp["nominal"]), float(sp["target"]), float(sp["prior"])
            fb_prec, fb_pull, _ = _answer_pieces(nom, pri, tv, tau)
            fb = case_charge(fb_prec, fb_pull)
            unlogged = no_measurement_charge(fb_prec, fb_pull)
            cap, capped, clipped = max(unlogged, CAL_CAP_FLOOR), False, False
            claim = None if n in unread else _reported(log.get(n))
            no_unc = bool(claim is not None and not claim[1] > 0)
            wp = (log.get(n) or {}).get("wp") if (n.startswith("eff_") and n[4:] in wps
                                                  and isinstance(log.get(n), dict)) else None
            modelled = None if cov is None else bool(n in (cov.get("modelled", cov.get("effects")) or ()))
            counted = claim is not None and not no_unc
            fails = None
            if modelled is False:
                fails = list((cov.get("model_probes", cov.get("probes")) or {}).get(n)
                             or ([] if not cov.get("model") else ["absent"])) or ["no model"]
            z = None
            if counted:
                d_prec, d_pull, z = _answer_pieces(claim[0], claim[1], tv, tau)
                clipped = bool(d_prec < CAL_PRECISION_FLOOR)
                d_prec = max(d_prec, CAL_PRECISION_FLOOR)
                raw = case_charge(d_prec, d_pull)
                delta = min(raw, cap) if not math.isnan(raw) else cap
                capped = bool(not raw <= cap)
            else:
                d_prec, d_pull, delta = fb_prec, fb_pull, unlogged
            cases.append({"seed": rec["seed"], "effect": n, "reported": claim is not None,
                          "logged": claim is not None, "modelled": modelled, "counted": counted, "model_fails": fails,
                          **({"wp": str(wp)[:80], "wp_unread": n in unread} if wp is not None else {}),
                          "value": claim[0] if claim is not None else None,
                          "unc": claim[1] if claim is not None else None,
                          "pull": z if (z is not None and math.isfinite(z)) else None,
                          "target": tau, "target_task": float(sp["target"]),
                          "delta_precision": d_prec if math.isfinite(d_prec) else None,
                          "delta_pull": d_pull if math.isfinite(d_pull) else None,
                          **({"no_uncertainty": True} if no_unc else {}),
                          "cap": cap, "fallback_charge": fb, "unlogged": unlogged, "delta": float(delta),
                          "capped": capped, "clipped": clipped,
                          "cluster": cluster_of(task, t_full, rec["seed"])})
    return cases


def gated_charge(delta: float, unlogged: float, f: Optional[float]) -> float:
    """A case's charge after the calibration gate: delta + f max(0, unlogged - delta), the share f of its credit
    against the unlogged charge withdrawn (f = 0 or None: delta)."""
    return float(delta + f * max(0.0, unlogged - delta)) if f else float(delta)


def calibration_term(cases: List[dict], gate: Optional[Dict[str, dict]] = None) -> Optional[dict]:
    """Delta_cal: the plain mean over the cases of their charges (calibration_cases), each after the calibration gate
    of its effect (``gate``: {effect: {"f", ...}}, sbh.kinds.binned_likelihood.scoring.calibration_gate; the share f
    of the case's credit against its unlogged charge withdrawn, gated_charge): the gate is where the model's evidence
    enters (a logged entry is scored on its value, modelled or not).  0 at the target precision with accurate
    uncertainties; an unlogged nuisance costs max(F_j, 0); a confidently wrong one up to its cap.  "se" the cases'
    cluster-robust standard error (report-only); per effect ("per_effect", diagnostics) the mean of its cases' charges
    ("delta"; "delta_ungated" before the gate), their mean unlogged charge ("delta_unlogged"), n, n_logged (alias
    n_reported), n_modelled (None = not checked; a diagnostic), n_counted (= n_logged), n_capped and the gate; "rule"
    None (no split of the counted effects by the model: SPLIT_RULE is the shipped rule's), "model_checked" whether the
    cases carry the model's coverage.  None without a case.  Cases of the shipped rule (they carry "fb_excess") are
    aggregated as it shipped (_shipped_calibration_term_gated)."""
    if not cases:
        return None
    if "fb_excess" in cases[0]:
        return _shipped_calibration_term_gated(cases, gate)
    f_of = {e: float(g.get("f") or 0.0) for e, g in (gate or {}).items() if isinstance(g, dict)}
    d = np.array([gated_charge(c["delta"], c["unlogged"], f_of.get(c["effect"])) for c in cases])
    checked = any(c.get("modelled") is not None for c in cases)
    per = {}
    for k in sorted({c["effect"] for c in cases}):
        ix = [i for i, c in enumerate(cases) if c["effect"] == k]
        cs = [cases[i] for i in ix]
        n_log = int(sum(c["reported"] for c in cs))
        per[k] = {"delta": float(np.mean(d[ix])), "delta_ungated": float(np.mean([c["delta"] for c in cs])),
                  "delta_unlogged": float(np.mean([c["unlogged"] for c in cs])),
                  "se": _cluster_se(d[ix], [c["cluster"] for c in cs]), "n": len(cs),
                  "n_reported": n_log, "n_logged": n_log,
                  "n_modelled": int(sum(bool(c.get("modelled")) for c in cs)) if checked else None,
                  "n_counted": int(sum(c["counted"] for c in cs)), "n_capped": int(sum(c["capped"] for c in cs)),
                  **({"gate": (gate or {}).get(k)} if gate is not None else {})}
    out = {"delta": float(d.mean()), "se": _cluster_se(d, [c["cluster"] for c in cases]), "n": len(cases),
           "n_counted": int(sum(c["counted"] for c in cases)), "n_capped": int(sum(c["capped"] for c in cases)),
           "charge_rule": CAL_RULE, "rule": None, "model_checked": checked, "per_effect": per}
    if gate:
        out["delta_ungated"] = float(np.mean([c["delta"] for c in cases]))
    return out


def calibration_term_gated(cal_cases: List[dict], cal_gate: Optional[Dict[str, dict]] = None) -> Optional[dict]:
    """The calibration term with a kind's gate (calibration_term)."""
    return calibration_term(cal_cases, cal_gate)



SHIPPED_CAL_AGGREGATE = "meanmax"

CAL_AGGREGATES = ("meanmax", "mean", "sum")


def _shipped_calibration_cases(records: List[dict], truths: Dict[str, dict], task,
                               coverage: Optional[Dict[str, dict]] = None) -> List[dict]:
    """calibration_cases under the shipped rule: unlogged = the assumed-nominal forecast; "delta" a per-case
    diagnostic, "excess" / "fb_excess" / "w" what _shipped_calibration_term aggregates."""
    agg = SHIPPED_CAL_AGGREGATE
    wps = getattr(task, "working_points", None) or {}
    cases = []
    for rec in records:
        t_full = truths[rec["seed"]]
        log = nuisance_log(rec)
        tv_all = task.calibration_truth(t_full, log)
        spec = calibration_spec(task, log)
        unread = unread_wp_entries(task, log)
        cov = None if coverage is None else (coverage.get(rec["seed"]) or {})
        for n, sp in spec.items():
            if n not in tv_all:
                continue
            tv = tv_all[n]
            nom, tgt, pri = sp["nominal"], sp["target"], sp["prior"]
            ref = math.log(tgt) + 0.5
            span = math.log(pri / tgt)
            fb_loss = _v2_loss(pri, nom - tv)
            claim = None if n in unread else _reported(log.get(n))
            wp = (log.get(n) or {}).get("wp") if (n.startswith("eff_") and n[4:] in wps
                                                  and isinstance(log.get(n), dict)) else None
            modelled = None if cov is None else bool(n in (cov.get("modelled", cov.get("effects")) or ()))
            counted = claim is not None and modelled is not False
            loss = fb_loss if not counted else min(_v2_loss(math.hypot(claim[1], tgt), claim[0] - tv), fb_loss)
            fails = None
            if modelled is False:
                fails = list((cov.get("model_probes", cov.get("probes")) or {}).get(n)
                             or ([] if not cov.get("model") else ["absent"])) or ["no model"]
            cases.append({"seed": rec["seed"], "effect": n, "reported": claim is not None,
                          "logged": claim is not None, "modelled": modelled, "counted": counted, "model_fails": fails,
                          **({"wp": str(wp)[:80], "wp_unread": n in unread} if wp is not None else {}),
                          "aggregate": agg,
                          "delta": (loss - ref) / span, "excess": loss - ref, "fb_excess": fb_loss - ref,
                          "w": math.hypot(tv - nom, tgt), "cluster": cluster_of(task, t_full, rec["seed"])})
    return cases


def _shipped_cal_aggregate(out: dict, aggregate: str) -> dict:
    """The shipped Delta_cal over the per-effect deltas (in place): all three aggregates plus max_effect, and "delta" /
    "se" by ``aggregate``:
      meanmax  (mean_e Delta_e + max_e Delta_e) / 2 (the shipped rule's)
      mean     mean_e Delta_e
      sum      sum_e Delta_e
    SEs: per-effect SEs in quadrature with each aggregate's coefficients (meanmax: 1/(2n) each, +1/2 on the max)."""
    if aggregate not in CAL_AGGREGATES:
        raise ValueError(f"cal_aggregate must be one of {CAL_AGGREGATES}, not {aggregate!r}")
    items = list(out["per_effect"].items())
    pe = [v for _, v in items]
    n = len(pe)
    q = math.sqrt(sum(float(v["se"]) ** 2 for v in pe))
    kmax = max(range(n), key=lambda i: (pe[i]["delta"], -i))
    d_mean = float(np.mean([v["delta"] for v in pe]))
    se_mm = math.sqrt(sum(((0.5 / n) + (0.5 if i == kmax else 0.0)) ** 2 * float(v["se"]) ** 2
                          for i, v in enumerate(pe)))
    out.update(aggregate=aggregate, delta_mean=d_mean, delta_sum=float(sum(v["delta"] for v in pe)),
               delta_meanmax=float(0.5 * (d_mean + pe[kmax]["delta"])), max_effect=items[kmax][0],
               se_mean=q / n, se_sum=q, se_meanmax=se_mm)
    out["delta"], out["se"] = out[f"delta_{aggregate}"], out[f"se_{aggregate}"]
    return out


def _shipped_calibration_term(cases: List[dict], aggregate: Optional[str] = None) -> Optional[dict]:
    """The shipped Delta_cal: per effect max(0, sum_j w_j (loss_j - ref) / sum_j w_j (fb_j - ref)),
    w_j = hypot(t_j - nominal, target); aggregated over effects by meanmax (``aggregate``: "meanmax", "mean" or "sum";
    None = the cases' own).  Per effect: n_logged (alias n_reported), n_modelled (None = not checked), n_counted."""
    if not cases:
        return None
    if aggregate is None:
        aggregate = next((c["aggregate"] for c in cases if c.get("aggregate")), None) or SHIPPED_CAL_AGGREGATE
    per = {}
    for c in cases:
        per.setdefault(c["effect"], []).append(c)
    checked = any(c.get("modelled") is not None for c in cases)
    out = {"per_effect": {}, "rule": SPLIT_RULE if checked else None}
    for k in sorted(per):
        cs = per[k]
        den = sum(c["w"] * c["fb_excess"] for c in cs)
        if den <= 0:
            continue
        m = sum(c["w"] * c["excess"] for c in cs) / den
        g: Dict[str, float] = {}
        for c in cs:
            g[c["cluster"]] = g.get(c["cluster"], 0.0) + c["w"] * (c["excess"] - m * c["fb_excess"]) / den
        se = float(np.sqrt(sum(v * v for v in g.values())))
        n_log = int(sum(c["reported"] for c in cs))
        out["per_effect"][k] = {"delta": float(max(m, 0.0)), "raw": float(m), "se": se, "n": len(cs),
                                "n_reported": n_log, "n_logged": n_log,
                                "n_modelled": int(sum(bool(c.get("modelled")) for c in cs)) if checked else None,
                                "n_counted": int(sum(c.get("counted", c["reported"]) for c in cs))}
    if not out["per_effect"]:
        return None
    _shipped_cal_aggregate(out, aggregate)
    out["n"] = len(cases)
    return out


def _shipped_calibration_term_gated(cal_cases: List[dict], cal_gate: Optional[Dict[str, dict]] = None) -> \
        Optional[dict]:
    """The shipped calibration term with a kind's gate: delta' = delta + f (1 - delta), f the fraction the model check
    forfeits; ungated values kept, term re-aggregated."""
    cal = _shipped_calibration_term(cal_cases)
    if cal and cal_gate:
        cal["delta_ungated"] = cal["delta"]
        for k, v in cal["per_effect"].items():
            g = cal_gate.get(k)
            v["delta_ungated"] = v["delta"]
            v["gate"] = g
            if g:
                v["delta"] = float(v["delta"] + g["f"] * (1.0 - v["delta"]))
        _shipped_cal_aggregate(cal, cal["aggregate"])
    return cal


def syst_units(task, log: Optional[dict] = None) -> List[tuple]:
    """Delta_syst's units: task.completeness_units minus those with an effect of the calibration spec under ``log``'s
    working points (None = defaults): an effect the injection varies, value-scored or held out (held_out), is never a
    unit of the never-varied nuisances."""
    scored = set(calibration_spec(task, log) if log is not None else (getattr(task, "calibration_scored", None) or {}))
    return [(label, tuple(names)) for label, names in (getattr(task, "completeness_units", None) or [])
            if not scored & set(names)]

