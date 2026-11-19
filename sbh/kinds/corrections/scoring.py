"""The corrections kind's scoring -- EVALUATOR-ONLY (sbh/harbor_pack.py EVALUATOR_ONLY): the requested corrections
at the task's evaluation points (function cases) through the core's v2 aggregation, and S_v3 (score_v3) with the
kind's terms -- completeness of the stated sources, the declared measurements' GOF (post.py), the closure
comparison (comparison.py; Delta_peak whatever it compares) -- and compute.  The physics term's log loss prices a
stated uncertainty both ways (too narrow through z^2 / 2, too wide through ln sigma), so there is no separate
truthfulness term.  docs/METRICS.md "Function outputs"."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...score import (EVALUATOR_FAULT, NO_REFERENCE_SCALE, PLACEHOLDER_REASON, UNBOUNDED_WHY,
                      REPRODUCED_BUDGET, SCORE_RULE,
                      V2_CAP_FLOOR, V2_CLIP, V3_W_COMP, V3_W_GOF, V3_W_SYST, _cluster_se, _loss_pieces, _wls_slope,
                      case_charge, compute_cost, compute_gate_params, compute_gate_source, failure_reason,
                      fallback_block, fallback_entry, load_closure_baseline, load_scored_records,
                      no_measurement_charge, own_budget_compute, reference_compute, read_json_maybe_shielded,
                      record_ok, run_yardstick, score_gamma, score_v3, v2_aggregate, write_scores, yardstick_weight)
from ..binned_likelihood.scoring import (GOF_CHARGE_MAX, gof_charge, GOF_RULE, V3_GOF_ALPHA, V3_GOF_DELTA_MIN, V3_GOF_P_FLOOR,
                                         _GOF_MAX, _chi2_logsf_even, _gof_usable, completeness_term, gof_delta,
                                         gof_gauss)


FUNCTION_RULE = ("function cases, completeness, the {closure} at 0.5, no truthfulness term; the declared "
                 "measurement's gate and goodness of fit as the task sets them")


def function_rule(task) -> str:
    """FUNCTION_RULE with the task's closure comparison named (comparison.label)."""
    from .comparison import label
    return FUNCTION_RULE.format(closure=label(task))
ACCURACY_RULE = "ideal-case reference: sigma_bar x the reference's pull RMS in the role group, both ways"
ACCURACY_FLOOR = 0.01
MEASUREMENT_RULE = ("the declared measurement: a gate on its own checks where the task requires it (one "
                    "that does not determine a correction passes: degenerate corrections stand), its refit's "
                    "goodness of fit (rule E) where the task refits it")

def _mean_loss(center, width, truth, tau) -> float:
    """The log loss: mean over the points of ln s + z^2 / 2, s = hypot(width, tau), z = (center - truth) / s."""
    se = np.hypot(np.asarray(width, dtype=float), np.asarray(tau, dtype=float))
    with np.errstate(over="ignore"):
        z = (np.asarray(center, dtype=float) - truth) / se
        return float(np.mean(np.log(se) + 0.5 * (z * z)))


def _sigma_bar(sigma, tau) -> float:
    """sigma_bar = exp(mean ln sigma_eff) at the points: a case's loss is ln sigma_bar + mean z^2 / 2."""
    return float(np.exp(np.mean(np.log(np.hypot(np.asarray(sigma, dtype=float), tau)))))


def _baseline_loss(sigma_b: float) -> float:
    """A calibrated answer's expected _mean_loss at half-band sigma_b: ln sigma_b + 1/2."""
    return float(np.log(sigma_b)) + 0.5


def _floor(sp: dict):
    """What a case adds in quadrature to every answer's sigma: hypot(tau, the truth's own sp["truth_sigma"])."""
    return np.hypot(float(sp["tau"]), np.asarray(sp.get("truth_sigma", 0.0), dtype=float))


def function_evaluations(run_dir: Path, records: List[dict], truths: Dict[str, dict], task) -> List[dict]:
    """Every scored case (Task.function_cases) with the answer in the comparison space: value "f" and half-band
    "sigma" at the points, or "why" there is none, and "run_failed" where the dataset's run failed (record_ok).  A
    completed run whose corrections.json is gone cannot be scored (an evaluator-side loss, never the submission's
    failure)."""
    from . import outputs as C
    if C.import_correctionlib() is None:
        raise RuntimeError(f"scoring a corrections task needs correctionlib {C.GRADER_CORRECTIONLIB} (pixi.toml)")
    out = []
    for rec in records:
        ok = record_ok(rec)
        cset, why_run = None, (None if ok else f"the run failed: {failure_reason(rec)}")
        if ok:
            cset, why_file = C.read_set(Path(run_dir) / "runs" / rec["seed"] / "out" / C.FILE)
            if cset is None:
                raise RuntimeError(f"{rec['seed']}: the completed run's {C.FILE} cannot be read ({why_file}); "
                                   f"scoring needs every completed run's out/ directory")
        result = rec["result"] if ok else None
        checked: Dict[tuple, List[str]] = {}
        for sp in task.function_cases(truths[rec["seed"]]):
            if not float(sp["tau"]) > 0:
                raise ValueError(f"{task.name}: case {sp['role']} has no positive floor tau")
            f = sig = None
            sources: List[str] = []
            impacts: Dict[str, float] = {}
            why = why_run
            if cset is not None:
                corr, why = C.find(cset, sp["name"])
                if corr is not None:
                    signed = sp["space"] == "signed_variance"
                    key = (sp["name"], sp["flavour"])
                    if key not in checked:
                        checked[key] = C.flavour_problems(corr, result, {sp["flavour"]: sp["domain"]},
                                                          signed=signed, cset=cset)[sp["flavour"]]
                    try:
                        if checked[key]:
                            raise C.CorrectionError("; ".join(checked[key]))
                        comp = C.components(corr, result, sp["points"], sp["flavour"], cset, signed=signed)
                        f, lo, hi = comp["value"], comp["lo"], comp["hi"]
                        f, sig = (C.signed_variance(f, lo, hi, sp["flavour"]) if signed
                                  else (f, 0.5 * np.abs(hi - lo)))
                        sources = sorted(src for src, v in comp["parts"].items() if np.any(v > 0)) \
                            if comp["consistent"] else []
                        impacts = {src: float(np.mean(comp["shifts"].get(src, v))) for src, v in comp["parts"].items()}
                    except C.CorrectionError as e:
                        f, why = None, str(e)
            out.append({"seed": rec["seed"], "spec": sp, "f": f, "sigma": sig,
                        "sources": sources, "impacts": impacts,
                        "why": None if f is not None else why, "run_failed": not ok})
    return out


def function_completeness_cases(evals: List[dict], truths: Dict[str, dict]) -> List[dict]:
    """The completeness term's cases: per dataset, delta = the fraction of (case, spec["required_sources"] class)
    pairs whose answer states no source of that class (an unanswered case states none; a case whose measurement is not
    checked yet, "pending", function_measurement_gate, is left out).  Empty if none required."""
    from .outputs import source_class
    per: Dict[str, List[Tuple[str, bool]]] = {}
    for e in evals:
        if e.get("pending"):
            continue
        req = e["spec"].get("required_sources") or ()
        have = {source_class(x) for x in e.get("sources") or ()} if e["f"] is not None else set()
        for cls in req:
            per.setdefault(e["seed"], []).append((f"{e['spec']['role']}:{cls}", cls in have))
    return [{"seed": seed, "missing": sorted(lab for lab, ok in pairs if not ok), "rejected": {},
             "delta": sum(not ok for _, ok in pairs) / len(pairs),
             "cluster": str(truths[seed].get("profile", seed))} for seed, pairs in sorted(per.items()) if pairs]


def load_measurement_rows(run_dir: Path, records: List[dict], task) -> Dict[str, dict]:
    """{seed: its measurement_verify.json} for every completed record with one made under the current rules
    (refit.current); any other seed is pending."""
    from .refit import VERIFY_FILE, current
    out = {}
    for rec in records:
        f = Path(run_dir) / "runs" / rec["seed"] / VERIFY_FILE
        if record_ok(rec) and f.is_file():
            row = json.loads(f.read_text())
            if current(row, task, run_dir):
                out[rec["seed"]] = row
    return out


def measurement_faults(rows: Dict[str, dict]) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """(the checked rows, the fault rows) of load_measurement_rows' rows.  A row whose status is not "ok" is the post
    stage's record of its own failure on the dataset (a "verify error" row, its message in "evaluator_error", its
    attempts in "grader_attempts"; the kind's checks write the model's failures per correction inside an "ok" row): the
    dataset is not checked, its terms left out as an evaluator fault (fault_reason), never charged."""
    ok, faults = {}, {}
    for seed, row in rows.items():
        (ok if row.get("status") == "ok" else faults)[seed] = row
    return ok, faults


def fault_reason(row: dict) -> str:
    """The pending reason of a fault row (measurement_faults): "evaluator fault: <its message>"
    (sbh.score.EVALUATOR_FAULT; the private scores keep it, the public reasons count it)."""
    msg = row.get("evaluator_error") or row.get("status") or "the post stage failed on the dataset"
    return f"{EVALUATOR_FAULT}: {str(msg)[:200]}"


def fault_split(faults: Dict[str, dict], completed=()) -> Tuple[List[str], List[str]]:
    """(reproduced, systemic) of the fault rows (measurement_faults), decided over the run's rows as they stand,
    whatever order the post stage checked the datasets in (binned_likelihood scoring.fault_split): systemic, a dataset
    whose fault places all lie where the rows of more than half of the run's completed datasets ``completed`` fault
    (run_systemic_places; the same fault on most datasets: the run provisional, never unranked by it, whether or not
    the post stage marked it "systemic_fault" or retried it); reproduced, any other whose fault the post stage saw
    again on its retry ("grader_attempts" n_evaluator_error >= 2: the evaluator's code needs a fix, the run shown
    unranked).  Without ``completed`` no fault is systemic."""
    from ..binned_likelihood.scoring import regrade_marked, run_systemic_places
    from ..binned_likelihood.verify import is_systemic
    places = run_systemic_places(faults, completed)
    repro, systemic = [], []
    for seed, row in sorted(faults.items()):
        att = row.get("grader_attempts") if isinstance(row.get("grader_attempts"), dict) else {}
        if is_systemic(regrade_marked(row), places):
            systemic.append(seed)
        elif int(att.get("n_evaluator_error") or 0) >= 2:
            repro.append(seed)
    return repro, systemic


NOT_CHECKED = "its measurement not checked yet"


def function_measurement_gate(evals: List[dict], rows: Dict[str, dict], faults: Optional[Dict[str, dict]] = None) \
        -> None:
    """The contract's measurement gate, in place: a case whose measurement is missing, malformed or fails a post-stage
    check counts as not reported for that flavour (no answer, no stated sources).  A completed dataset the post stage
    has not checked (``rows``, the checked rows, has none) has its answered cases "pending": left out of every term
    until it is checked, never scored as answered and never charged -- with the evaluator-fault reason where the post
    stage failed on it (``faults``, the fault rows of measurement_faults: fault_reason), else NOT_CHECKED
    (sbh.score.pending_reason)."""
    from ...score import pending_reason
    for e in evals:
        row = rows.get(e["seed"])
        if e["f"] is None:
            continue
        if row is None:
            fr = (faults or {}).get(e["seed"])
            e["pending"] = fault_reason(fr) if fr is not None else pending_reason(NOT_CHECKED)
            continue
        c = row["corrections"].get(f"{e['spec']['name']}|{e['spec']['flavour']}")
        if c is None or c["status"] != "ok":
            why = c["why"] if c else "not checked for this correction"
            e.update(f=None, sigma=None, sources=[],
                     why=why if why.startswith("its measurement") else f"its measurement: {why}")


def function_gof_cases(evals: List[dict], rows: Dict[str, dict], records: List[dict],
                       truths: Dict[str, dict]) -> List[dict]:
    """The GOF cases: one per requested correction and flavour of every dataset (so every dataset weighs alike) with
    the toy p of its measurement's refit (refit.py); corrections sharing a measurement enter the ensemble once.
    Charged the maximum: a failed run, no measurement, a failed check, and a measurement finer than the data support
    (no degrees of freedom in any binning the statistics allow).  An unchecked completed dataset is pending."""
    cases = []
    done = {r["seed"] for r in records if record_ok(r)}
    for e in evals:
        seed, key = e["seed"], f"{e['spec']['name']}|{e['spec']['flavour']}"
        base = {"seed": seed, "correction": key, "cluster": str(truths[seed].get("profile", seed))}
        if seed not in done:
            cases.append({**base, "measurement": None, "p": None, "charged": True, "missing": "run", **_GOF_MAX})
            continue
        row = rows.get(seed)
        if row is None:
            continue
        mid = (row["corrections"].get(key) or {}).get("measurement")
        m = (row["measurements"].get(mid) or {}) if mid is not None else {}
        missing = ("measurement" if mid is None else "check" if m.get("status") != "ok"
                   else "dof" if m.get("p_gof") is None else None)
        if missing:
            cases.append({**base, "measurement": mid, "p": None, "charged": True, "missing": missing, **_GOF_MAX})
        else:
            cases.append({**base, "measurement": mid, "p": float(m["p_gof"]), "charged": False,
                          **gof_delta(float(m["p_gof"]))})
    return cases


def _function_gof_ensemble(cases: List[dict]) -> dict:
    """gof_ensemble for function cases: Fisher's test over the distinct measurements' p's, every usable case charged
    Delta_F, every other the maximum; delta = the mean over the cases."""
    ps = {(c["seed"], c["measurement"]): float(c["p"]) for c in cases if _gof_usable(c)}
    n, n_ok = len(cases), sum(1 for c in cases if _gof_usable(c))
    out = {"rule": GOF_RULE, "alpha": V3_GOF_ALPHA, "n": n, "n_ok": n_ok, "n_fail": n - n_ok, "n_wall": 0,
           "n_measurements": len(ps), "fisher_x": None, "dof": 2 * len(ps), "log_p_f": None, "p_f": None,
           "delta_f": None, "delta": None}
    if ps:
        x = -2.0 * sum(math.log(max(p, V3_GOF_P_FLOOR)) for p in ps.values())
        lp = _chi2_logsf_even(x, len(ps))
        out.update(fisher_x=x, log_p_f=lp, p_f=math.exp(lp),
                   delta_f=gof_charge(lp))
    if n:
        out["delta"] = ((n_ok * out["delta_f"] if n_ok else 0.0) + (n - n_ok) * GOF_CHARGE_MAX) / n
    return out


def function_gof_term(cases: Optional[List[dict]]) -> Optional[dict]:
    """gof_term for function cases: the ensemble, each case's "delta" set to its charge, the delete-one-cluster
    jackknife SE, and report-only diagnostics (mean capped and raw charges, Gaussian check of the p's)."""
    if not cases:
        return None
    raw = [c.get("delta_raw", c.get("delta")) for c in cases]
    cap = [c.get("delta_cap", max(V3_GOF_DELTA_MIN, r)) for c, r in zip(cases, raw)]
    e = _function_gof_ensemble(cases)
    for c in cases:
        c["delta"] = e["delta_f"] if _gof_usable(c) else GOF_CHARGE_MAX
    groups = sorted({c["cluster"] for c in cases})
    se = 0.0
    if len(groups) >= 2:
        ts = [_function_gof_ensemble([c for c in cases if c["cluster"] != g])["delta"] for g in groups]
        tb = sum(ts) / len(ts)
        se = math.sqrt((len(ts) - 1) / len(ts) * sum((t - tb) ** 2 for t in ts))
    ps = {(c["seed"], c["measurement"]): float(c["p"]) for c in cases if _gof_usable(c)}
    return {**e, "se": float(se), "delta_cap": float(np.mean(cap)), "delta_raw": float(np.mean(raw)),
            "delta_min": V3_GOF_DELTA_MIN, "n_charged": int(sum(bool(c.get("charged")) for c in cases)),
            "gauss": gof_gauss(list(ps.values()))}


V3_W_PEAK = 0.5


def peak_baseline_of(rows: Dict[str, dict]) -> Dict[str, dict]:
    """A closure run's comparison baseline from its own post stage: {seed: {flavour: {"loss", "loss_fallback"}}}."""
    return {seed: {fl: {"loss": float(v["loss"]), "loss_fallback": float(v["loss_fallback"])}
                   for fl, v in sorted((row.get("peak") or {}).items()) if v.get("loss") is not None}
            for seed, row in sorted(rows.items())}


def closure_extras(task, priv: dict) -> Dict[str, Any]:
    """closure.json's further baselines: the achieved-accuracy factors per group (function_accuracy, under
    baseline_accuracy) and the closure comparison's baseline, `peak_baseline` whatever it compares (the reference's
    and the no-answer forecast's loss per dataset, from the closure run's post stage), stamped with the comparison's
    rule (comparison.rule) and verify_rule (which corrections enter at their forecast).  Without the post stage there
    is none, and every run's peak term stays pending."""
    from .comparison import noun, rule
    acc = {"function_accuracy": priv["function_accuracy"]} if priv.get("function_accuracy") else {}
    from .refit import verify_rule
    pb = priv.get("peak_baseline")
    if not pb:
        print(f"closure: no {noun(task)} baseline yet (run the post stage on the closure run, then `sbh closure "
              "--phase score`); the peak term of every run scored against this closure stays pending")
        return acc
    return {**acc, "peak_baseline": {"rule": rule(task), "verify_rule": verify_rule(task), "per_seed": pb}}


def load_peak_baseline(suite_dir: Path, task) -> Tuple[Optional[Dict[str, dict]], Optional[str]]:
    """(closure.json's comparison baseline per seed, None) or (None, why: none, or made under another comparison
    rule / verify_rule)."""
    from .comparison import noun, rule
    from .refit import verify_rule
    p = Path(suite_dir) / "closure.json"
    pb = json.loads(p.read_text()).get("peak_baseline") if p.exists() else None
    what = noun(task)
    if not pb:
        return None, f"closure.json has no {what} baseline (rerun `sbh closure`)"
    want, rl = verify_rule(task), rule(task)
    if (pb.get("rule"), pb.get("verify_rule")) != (rl, want):
        return None, (f"closure.json's {what} baseline was made under {pb.get('rule')!r} / {pb.get('verify_rule')!r}, "
                      f"not {rl!r} / {want!r}")
    return pb["per_seed"], None


PEAK_RERUN_X = 2.0


def budget_reproduced(row: Optional[dict]) -> bool:
    """Whether a dataset's closure comparison ran out of its budget again on a rerun at a budget at least PEAK_RERUN_X
    times one it missed: its latest attempt (post.peak_attempts, the post stage's file) ran out, and so did an earlier
    one at a budget no more than 1 / PEAK_RERUN_X of the latest's.  A rerun at a smaller factor reproduces nothing (the
    comparison is made again at PEAK_RERUN_X or more)."""
    att = [a for a in ((row or {}).get("peak_attempts") or []) if isinstance(a, dict)]
    if len(att) < 2 or not att[-1].get("not_in_budget"):
        return False
    last = _num(att[-1].get("peak_s"))
    return last is not None and any(
        a.get("not_in_budget") and (_num(a.get("peak_s")) or 0.0) > 0.0
        and last >= PEAK_RERUN_X * _num(a.get("peak_s")) * (1.0 - 1e-9) for a in att[:-1])


def rerun_remedy() -> str:
    """The remedy a comparison not made within its budget names: made again with measurement_verify --peak-budget-x
    at PEAK_RERUN_X or more times its budget (a further reference's evaluator_budget_x only scales that reference's own
    budgets, post.budget_factors)."""
    return f"made again with measurement_verify --peak-budget-x X, X >= {PEAK_RERUN_X:g}"


def _num(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def function_peak_cases(records: List[dict], rows: Dict[str, dict], baseline: Dict[str, dict],
                        truths: Dict[str, dict]) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """(Delta_peak's cases, the comparisons not made).  The cases (the closure comparison's), one per completed dataset
    and flavour of the baseline, the v2 rule on the loss L against the reference's L_ref: delta = min(max(L - L_ref,
    V2_CLIP), C), C = max(L_fallback - L_ref, V2_CAP_FLOOR), L_fallback the loss with every correction at its
    no-answer forecast.  A dataset whose run failed has none (it fails the run).  A comparison not made gives no case
    and leaves the score provisional: one not made within its budget (post.PEAK_NOT_IN_BUDGET; no contract states
    that budget, so it is the evaluation's own limit: made again at PEAK_RERUN_X or more times it, rerun_remedy;
    counted as the fallback "peak_not_in_budget"; "reproduced" where it ran out again on that rerun,
    budget_reproduced), any
    other (the evaluator could not read the outputs it compares) as the evaluator's ({"not_in_budget": [...],
    "not_made": [...]}, each {"seed", "flavour", "why"}).  A completed dataset not yet compared is pending."""
    from .post import PEAK_NOT_IN_BUDGET
    cases: List[dict] = []
    missed: Dict[str, List[dict]] = {"not_in_budget": [], "not_made": []}
    for rec in records:
        seed = rec["seed"]
        if not record_ok(rec):
            continue
        peak = (rows.get(seed) or {}).get("peak")
        if peak is None:
            continue
        cluster = str(truths[seed].get("profile", seed))
        for fl, b in sorted((baseline.get(seed) or {}).items()):
            pk = peak.get(fl)
            if pk is None:
                continue
            if pk.get("loss") is None:
                why = str(pk.get("why") or "no comparison")
                nib = why.startswith(PEAK_NOT_IN_BUDGET)
                missed["not_in_budget" if nib else "not_made"].append(
                    {"seed": seed, "flavour": fl, "why": why,
                     **({"reproduced": budget_reproduced(rows.get(seed))} if nib else {})})
                continue
            cap = max(b["loss_fallback"] - b["loss"], V2_CAP_FLOOR)
            raw = float(pk["loss"]) - b["loss"]
            cases.append({"seed": seed, "flavour": fl, "ref_loss": b["loss"], "cap": cap, "cluster": cluster,
                          "loss": float(pk["loss"]), "delta": min(max(raw, V2_CLIP), cap), "delta_raw": raw,
                          "failed": False, "capped": raw > cap, "clipped": raw < V2_CLIP, "why": None,
                          "fallback": pk.get("fallback") or [], "n_categories": pk.get("n_categories"),
                          "per_category": pk.get("per_category"), "curve": pk.get("curve"),
                          "source_fit": pk.get("source_fit"),
                          **{k: pk[k] for k in ("per_stream", "curves") if k in pk}})
    return cases, missed


FALLBACKS = {
    "peak_not_in_budget": "the closure comparison was not made within the evaluation's budget for it: the dataset "
                          "has no peak case, and the score stays provisional until the comparison is made again at "
                          "the rule's rerun factor times that budget or more (measurement_verify --peak-budget-x; "
                          "missed again there, the run is shown unranked)",
}


def fallbacks(missed: Optional[Dict[str, List[dict]]]) -> Dict[str, dict]:
    """{kind: sbh.score.fallback_entry} of FALLBACKS taken (function_peak_cases' comparisons not made within their
    budget); a kind not taken is absent."""
    nib = (missed or {}).get("not_in_budget") or []
    return {"peak_not_in_budget": fallback_entry([m["seed"] for m in nib], FALLBACKS["peak_not_in_budget"])} \
        if nib else {}


def peak_pending_reasons(task, missed: Optional[Dict[str, List[dict]]]) -> List[str]:
    """The provisional reasons of the closure comparisons not made (function_peak_cases' ``missed``), named reasons
    and counts of datasets, never the evaluator's message (the private scores keep it, "v4_peak_missed"): one the
    evaluator could not make (an evaluator fault: fixed and rerun), one not made within the evaluation's budget (made
    again at PEAK_RERUN_X or more times it, rerun_remedy; also counted as the fallback "peak_not_in_budget"), and one
    that ran out again on that rerun (budget_reproduced: sbh.score.REPRODUCED_BUDGET, escalated as a reproduced
    evaluator fault is: the run is shown unranked with this reason, sbh.score.unranked_reason)."""
    from .comparison import pending as pending_note
    out = []
    not_made = (missed or {}).get("not_made") or []
    if not_made:
        out.append(f"{pending_note(task)} not made on {len({m['seed'] for m in not_made})} completed datasets: an "
                   f"{EVALUATOR_FAULT}, until it is fixed and the evaluation rerun")
    not_in_budget = (missed or {}).get("not_in_budget") or []
    again = {m["seed"] for m in not_in_budget if m.get("reproduced")}
    once = {m["seed"] for m in not_in_budget} - again
    if again:
        out.append(f"{REPRODUCED_BUDGET} on {len(again)} completed datasets: {pending_note(task)} not made within the "
                   f"evaluation's budget, again on a rerun at {PEAK_RERUN_X:g} or more times it (the comparison needs a "
                   "fix or a decision on its budget)")
    if once:
        out.append(f"{pending_note(task)} not made within the evaluation's budget on {len(once)} completed datasets: "
                   f"{rerun_remedy()}")
    return out


def measurement_fault_reasons(faults: Dict[str, dict], n_records: int, completed=()) -> List[str]:
    """The provisional reasons of the datasets the post stage failed on (the fault rows, measurement_faults), named
    reasons and counts, never the evaluator's message: their checks left out on an evaluator fault (fixed and rerun);
    a fault seen again on the stage's retry (sbh.score.REPRODUCED_FAULT: the run shown unranked,
    sbh.score.unranked_reason); a systemic one (sbh.kinds.binned_likelihood.verify.SYSTEMIC_FAULT: provisional,
    never unranked by it), both as fault_split decides them over the run's completed datasets ``completed``.  Each
    left-out dataset's corrections are named apart among the physics cases left out on an evaluator fault where the
    task gates on its checks (function_measurement_gate)."""
    from ..binned_likelihood.scoring import reproduced_reason
    from ..binned_likelihood.verify import SYSTEMIC_FAULT
    if not faults:
        return []
    repro, systemic = fault_split(faults, completed)
    rest = len(faults) - len(repro) - len(systemic)
    out = []
    if rest:
        out.append(f"measurement checks left out on an {EVALUATOR_FAULT} on {rest} of {n_records} datasets, until it "
                   "is fixed and the evaluation rerun")
    if systemic:
        out.append(f"measurement checks left out on a {SYSTEMIC_FAULT} on {len(systemic)} of {n_records} datasets, "
                   "until the evaluator is fixed and the run checked again")
    if repro:
        out.append(reproduced_reason(repro))
    return out


def peak_term(cases: Optional[List[dict]], rule: str) -> Optional[dict]:
    """Delta_peak: the mean of the cases' deltas, its cluster standard error, and per flavour the mean, stamped with
    the comparison's `rule`."""
    if not cases:
        return None
    d = np.array([c["delta"] for c in cases])
    return {"rule": rule, "delta": float(d.mean()), "se": _cluster_se(d, [c["cluster"] for c in cases]),
            "n": len(cases), "n_capped": int(sum(c["capped"] for c in cases)),
            "per_flavour": {fl: float(np.mean([c["delta"] for c in cases if c["flavour"] == fl]))
                            for fl in sorted({c["flavour"] for c in cases})}}


def function_sigma_ref(evals: List[dict]) -> Dict[str, float]:
    """self_sigma_ref's analogue for function cases: per role the median over its answered datasets of sigma_bar."""
    per: Dict[str, List[float]] = {}
    for e in evals:
        if e["f"] is not None and not e.get("pending"):
            per.setdefault(e["spec"]["role"], []).append(_sigma_bar(e["sigma"], _floor(e["spec"])))
    return {r: float(np.median(v)) for r, v in per.items()}


def write_function_points(run_dir: Path, evals: List[dict]) -> None:
    """Every scored case's answer, uncertainty and truth at its evaluation points, with the points' variables, per
    dataset in runs/<seed>/POINTS_FILE (float32; written atomically).  The card bins them into the task's
    function_views, so a new view needs a card rebuild, not a rescore.  A case not answered stores its truth only."""
    import os
    from . import POINTS_FILE
    per: Dict[str, dict] = {}
    for e in evals:
        sp = e["spec"]
        a = per.setdefault(e["seed"], {"_meta": {}})
        role = sp["role"]
        a["_meta"][role] = {"group": sp["group"], "domain": sp["domain"], "tau": float(sp["tau"]),
                            "space": sp["space"], "answered": e["f"] is not None}
        a[f"{role}.truth"] = np.asarray(sp["truth"], dtype=np.float32)
        for k, v in sp["points"].items():
            v = np.asarray(v)
            if v.dtype.kind in "fiub":
                a[f"{role}.point.{k}"] = v.astype(np.float32) if v.dtype.kind == "f" else v
        if e["f"] is not None:
            a[f"{role}.value"] = np.asarray(e["f"], dtype=np.float32)
            a[f"{role}.sigma"] = np.asarray(e["sigma"], dtype=np.float32)
    for seed, a in per.items():
        d = Path(run_dir) / "runs" / seed
        if not d.is_dir():
            continue
        a["_meta"] = np.array(json.dumps(a["_meta"]))
        tmp = d / (POINTS_FILE + ".part")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **a)
        os.replace(tmp, d / POINTS_FILE)


def accuracy_factors(cases: List[dict]) -> Dict[str, dict]:
    """The reference's achieved accuracy per group from the closure run's own cases: its pull RMS r as the factor
    scaling the group's baseline (ACCURACY_RULE), both ways -- stated uncertainties wider than its errors are held to
    the precision reached, narrower ones are taken at the precision reached, not the one stated (which no method is
    known to reach).  Its own case loss is then 1/2 (r^2 - 1) - ln r >= 0 per group."""
    z2: Dict[str, List[float]] = {}
    for c in cases:
        if not c["fallback"]:
            z2.setdefault(c["group"], []).append(c["z2"])
    out = {}
    for g, v in sorted(z2.items()):
        r = math.sqrt(float(np.mean(v)))
        out[g] = {"pull_rms": r, "factor": max(r, ACCURACY_FLOOR)}
    return out


def load_accuracy(suite_dir: Path) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """({group: factor}, None) from closure.json's function_accuracy, or (None, why: none, or another rule)."""
    p = Path(suite_dir) / "closure.json"
    acc = json.loads(p.read_text()).get("function_accuracy") if p.exists() else None
    if not acc:
        return None, "closure.json has no achieved-accuracy factors (CorrectionsTask.baseline_accuracy)"
    if acc.get("rule") != ACCURACY_RULE:
        return None, f"closure.json's achieved-accuracy factors were made under {acc.get('rule')!r}"
    return {g: float(v["factor"]) for g, v in acc["groups"].items()}, None


def function_cases(evals: List[dict], sigma_ref: Dict[str, float], ref_pois: Optional[Dict[str, dict]],
                   accuracy: Optional[Dict[str, float]] = None, multiworld: bool = True) -> dict:
    """The v2 cases, one per (dataset, role) with a reference scale, plus what v2_aggregate needs.  `accuracy`
    ({group: factor}) sets each baseline to _baseline_loss(sigma_ref x factor).  A case's charge is
    sbh.score.case_charge of its pieces: the precision piece ln sigma_bar - ln sigma_b (the geometric-mean width
    ratio) floored at -ln 1.25, the pull piece mean z^2 / 2 - 1/2, capped at C.  A case on a dataset whose run failed
    is failed (no charge; its dataset fails the run, "failed").  A correction not reported on a completed dataset
    (missing, malformed, not evaluating in its domain: the contract's "counts as that correction not reported", which
    "affects that correction only") has no measurement: charged max(F, 0), F its no-answer forecast's charge (the
    fallback's pieces through sbh.score.no_measurement_charge, F <= C), "no_measurement" True, no miss in the bias
    term, its dataset not failed; one
    without a reference scale is left out as provisional ("unscored", sbh.score.NO_REFERENCE_SCALE), and so is one
    whose measurement the post stage has not checked ("pending", function_measurement_gate: its reason).  An answered
    case's miss in the bias term (mw_miss, sbh.score.multiworld) is the mean of its point misses, each clipped at the
    term's clip, its spread (mw_spread) their sd over the points and its width sigma_bar; `multiworld` is the task's
    switch (Task.multiworld)."""
    cases: List[dict] = []
    fits: Dict[str, List[tuple]] = {}
    groups: Dict[str, str] = {}
    roles: List[str] = []
    scored_truths: Dict[str, set] = {}
    fb_width: Dict[str, List[float]] = {}
    failed: Dict[str, List[str]] = {}
    unscored: List[dict] = []
    for e in evals:
        sp, seed = e["spec"], e["seed"]
        role, tau = sp["role"], _floor(sp)
        tv = np.asarray(sp["truth"], dtype=float)
        if role not in groups:
            roles.append(role)
            groups[role] = sp["group"]
        good = e["f"] is not None
        run_failed = bool(e.get("run_failed"))
        no_meas = not good and not run_failed
        if run_failed:
            why = str(e["why"] or "the run failed")
            failed.setdefault(seed, [])
            if why not in failed[seed]:
                failed[seed].append(why)
        fc, fw = sp["fallback"]
        fb_width.setdefault(role, []).append(float(np.mean(fw)))
        if e.get("pending") and not run_failed:
            unscored.append({"seed": seed, "poi": role, "reason": str(e["pending"])})
            continue
        fb_loss = _mean_loss(fc, fw, tv, tau)
        loss = _mean_loss(e["f"], e["sigma"], tv, tau) if good else fb_loss
        rp = (ref_pois or {}).get(seed, {}).get(role)
        factor = float((accuracy or {}).get(sp["group"], 1.0))
        if rp and float(rp.get("sigma", 0.0)) > 0:
            ref_loss, ref_src = _baseline_loss(float(rp["sigma"]) * factor), "seed"
        elif sigma_ref.get(role, 0.0) > 0:
            ref_loss, ref_src = _baseline_loss(sigma_ref[role] * factor), "scalar"
        else:
            if no_meas:
                unscored.append({"seed": seed, "poi": role,
                                 "reason": f"{NO_REFERENCE_SCALE} (no measurement: {e['why']})"})
            continue
        raw = loss - ref_loss
        d_prec, d_pull = _loss_pieces(math.log(_sigma_bar(e["sigma"] if good else fw, tau)), loss, ref_loss)
        delta = (case_charge(d_prec, d_pull) if good else
                 no_measurement_charge(d_prec, d_pull) if no_meas else None)
        if delta is not None and not math.isfinite(delta):
            good, no_meas, delta = False, False, None
            e = dict(e, why=UNBOUNDED_WHY)
            failed.setdefault(seed, [])
            if f"{role}: {UNBOUNDED_WHY}" not in failed[seed]:
                failed[seed].append(f"{role}: {UNBOUNDED_WHY}")
        dcl = str(sp.get("dataset_cluster", seed))
        tbar = float(np.mean(tv))
        case = {"seed": seed, "poi": role, "loss": loss, "ref_loss": ref_loss, "ref_factor": factor, "delta": delta,
                "delta_raw": raw if good else None, "fallback": not good,
                **({"no_measurement": True} if no_meas else {}),
                "delta_precision": d_prec, "delta_pull": d_pull, "cap": None,
                "capped": False, "clipped": False,
                "gated": True, "ref_source": ref_src, "mw_miss": None, "cluster": str(sp["cluster"]),
                "dataset_cluster": dcl,
                "name": sp["name"], "flavour": sp["flavour"], "group": sp["group"], "null": bool(sp["null"]),
                "space": sp["space"], "n_points": int(tv.size), "truth_mean": tbar, "why": e["why"]}
        if not math.isfinite(loss):
            case.update(loss=None, delta_raw=None, delta_pull=None)
        if good:
            se = np.hypot(e["sigma"], tau)
            with np.errstate(over="ignore", invalid="ignore"):
                z = (e["f"] - tv) / se
                z2 = float(np.mean(z * z))
                zbar = float(np.mean(z))
            u = z
            case.update(value_mean=float(np.mean(e["f"])), sigma_eff=_sigma_bar(e["sigma"], tau),
                        zbar=zbar if math.isfinite(zbar) else None,
                        mw_miss=float(np.mean(u)), mw_spread=float(np.std(u)),
                        z2=z2 if math.isfinite(z2) else None,
                        pinned=bool(np.all(e["f"] == sp["nominal"])),
                        impacts=e.get("impacts") or {})
            fits.setdefault(role, []).append((tbar, case["value_mean"], case["sigma_eff"], sp.get("fit_group")))
        cases.append(case)
        scored_truths.setdefault(role, set()).add(round(tbar, 9))
    return {"cases": cases, "fits": {r: fits[r] for r in roles if r in fits}, "unanswered": [],
            "roles": roles, "groups": groups, "scored_truths": scored_truths,
            "multiworld": bool(multiworld), "rule": SCORE_RULE, "failed": failed, "failed_cases": {},
            "unscored": unscored,
            "effort_scale": {r: sigma_ref.get(r) or float(np.mean(fb_width[r])) for r in roles}}


def function_summary(evals: List[dict], cases: List[dict], rule: str) -> dict:
    """The summary's function block (stamped with `rule`): answered counts, and per role and group the mean point pull
    z-bar, the pull RMS sqrt(mean z^2), the response slope of mean answer on mean truth, the cases pinned exactly at
    nominal."""
    roles: Dict[str, dict] = {}
    for c in cases:
        m = roles.setdefault(c["poi"], {"group": c["group"], "flavour": c["flavour"], "n": 0, "n_answered": 0,
                                        "n_null": 0, "n_pinned": 0, "_rows": []})
        m["n"] += 1
        m["n_null"] += c["null"]
        if not c["fallback"]:
            m["n_answered"] += 1
            m["n_pinned"] += c["pinned"]
            m["_rows"].append((c["truth_mean"], c["value_mean"], c["sigma_eff"], c["zbar"], c["z2"]))
    groups: Dict[str, dict] = {}
    for r, m in roles.items():
        rows = m.pop("_rows")
        m["n_truths"] = len({round(t, 9) for t, *_ in rows})
        zb, z2s = [x[3] for x in rows], [x[4] for x in rows]
        m["zbar_mean"] = float(np.mean(zb)) if rows and None not in zb else None
        m["zbar_max_abs"] = float(max(abs(x) for x in zb)) if rows and None not in zb else None
        m["pull_rms"] = float(math.sqrt(np.mean(z2s))) if rows and None not in z2s else None
        slope = se = None
        if m["n_truths"] >= 3:
            b, e = _wls_slope(*(np.array([x[i] for x in rows], dtype=float) for i in range(3)))
            slope, se = (b, e) if np.isfinite(b) else (None, None)
        m["slope"], m["slope_se"] = slope, se
        g = groups.setdefault(m["group"], {"roles": [], "_z2": []})
        g["roles"].append(r)
        g["_z2"] += [x[4] for x in rows]
    for g in groups.values():
        z2 = g.pop("_z2")
        g["pull_rms"] = float(math.sqrt(np.mean(z2))) if z2 and None not in z2 else None
    return {"rule": rule, "n_requested": len(evals),
            "n_answered": sum(e["f"] is not None for e in evals), "roles": roles, "groups": groups}


def function_baseline_problem(suite_dir: Path, task) -> Optional[str]:
    """Why closure.json cannot be this task's baseline (another Task.function_baseline_id), or None."""
    p = Path(suite_dir) / "closure.json"
    if not p.exists():
        return None
    made, want = json.loads(p.read_text()).get("function_baseline"), task.function_baseline_id()
    return None if made == want else f"closure.json's baseline was made under {made!r}, not {want!r}"


def score_functions(run_dir: Path, suite_dir: Path, task, label: str, sigma_ref: Optional[Dict[str, float]],
                    manifest: dict, truths: Dict[str, dict], records: List[dict], making_baseline: bool = False,
                    lost: Tuple[str, ...] = ()):
    """(public, private) of a corrections run: Delta_total = Delta_phys + W_SYST x Delta_syst + W_GOF x Delta_gof +
    W_PEAK x Delta_peak + W_COMP x Delta_compute -- source completeness (when the task requires sources), the
    measurements' GOF (rule E's ensemble), the closure comparison (against closure.json's peak baseline), compute
    (with a reference wall: static or Task.compute_yardstick_k).  The physics term is also the calibration term
    (answers against the injected truths, uncertainty included).  A correction failing the measurement gate counts
    as not reported; an unchecked dataset (not checked yet, or one the post stage failed on: its fault row, an
    evaluator fault, measurement_faults) leaves its gated corrections and the GOF and peak terms pending, its fault
    named (measurement_fault_reasons: seen again on the stage's retry, the run is shown unranked).  S_phys (v1) is
    not computed: its gates assume calibrated scalar pulls.  A closure.json of another baseline id is refused (rerun `sbh closure`);
    the closure run is scored against the baseline it writes.  A dataset whose run failed fails the run (S = 0); a
    correction not reported on a completed dataset has no measurement (charged max(F, 0), function_cases); a closure
    comparison not made within its budget gives no peak case and leaves the score provisional (made again at
    PEAK_RERUN_X or more times it, rerun_remedy; counted as the fallback "peak_not_in_budget"; missed again there, the
    run is shown unranked, sbh.score.REPRODUCED_BUDGET); ``lost``, the datasets without a record.json, are
    no term's cases and leave the score provisional.  v3 "fallbacks" counts the fallbacks taken
    (sbh.score.fallback_block with the kind's FALLBACKS)."""
    from .comparison import pending as pending_note, rule
    from .refit import verify_rule
    evals = function_evaluations(run_dir, records, truths, task)
    rows, faults = measurement_faults(load_measurement_rows(run_dir, records, task))
    if getattr(task, "measurement_required", True):
        function_measurement_gate(evals, rows, faults)
    write_function_points(run_dir, evals)
    accuracy = accuracy_rec = None
    if making_baseline:
        ref_pois = {}
        own = function_cases(evals, function_sigma_ref(evals), None)["cases"]
        for c in own:
            if not c["fallback"]:
                ref_pois.setdefault(c["seed"], {})[c["poi"]] = {"value": c["value_mean"], "sigma": c["sigma_eff"]}
        if getattr(task, "baseline_accuracy", False):
            accuracy_rec = {"rule": ACCURACY_RULE, "groups": accuracy_factors(own)}
            accuracy = {g: v["factor"] for g, v in accuracy_rec["groups"].items()}
        closure_sigma, closure_passed, sigma_ref = {}, True, None
        src = "self (the closure's own per-seed baseline)"
        peak_base, peak_why = peak_baseline_of(rows), None
    else:
        ref_pois, closure_sigma, closure_passed = load_closure_baseline(suite_dir)
        stale = function_baseline_problem(suite_dir, task)
        if stale:
            raise SystemExit(f"{stale}: rerun `sbh closure` on {suite_dir} before scoring against it")
        if getattr(task, "baseline_accuracy", False):
            accuracy, why = load_accuracy(suite_dir)
            if why:
                raise SystemExit(f"{why}: rerun `sbh closure` on {suite_dir} before scoring against it")
        src = "reference closure" if sigma_ref else "self-referenced (no reference sigma; precision term == 1)"
        peak_base, peak_why = load_peak_baseline(suite_dir, task)
    sigma_ref = sigma_ref or function_sigma_ref(evals)
    built = function_cases(evals, closure_sigma or sigma_ref, ref_pois, accuracy,
                           bool(getattr(task, "multiworld", True)))
    v2 = v2_aggregate(built, closure_sigma or sigma_ref, ref_pois, score_gamma(task))
    if v2 and not closure_passed:
        v2["ref_loss_source"] += " (closure not passed — rescore after it does)"
    if v2:
        v2.pop("cases")
    yard = run_yardstick(run_dir, task)
    comp = own_budget_compute(compute_cost(records, task, yard), records, task, yard, label)
    comp = reference_compute(comp, yard, making_baseline)
    systc = function_completeness_cases(evals, truths)
    n_pending = sum(record_ok(r) for r in records) - len(rows)
    n_unchecked = n_pending - len(faults)
    checked = bool(rows) or not any(record_ok(r) for r in records)
    refit = getattr(task, "measurement_refit", True)
    gofc = function_gof_cases(evals, rows, records, truths) if refit and checked else None
    peakc, missed = function_peak_cases(records, rows, peak_base, truths) if checked and peak_base else (None, None)
    peakc = peakc or None
    v3 = None
    if v2:
        pending, gterm = [], None
        if lost:
            pending.append(f"{len(lost)} datasets without a record.json, left out until rerun (infrastructure): "
                           f"{', '.join(lost[:8])}{', ...' if len(lost) > 8 else ''}")
        if refit:
            gterm = function_gof_term(gofc)
            if gterm is None and not faults:
                pending.append("GOF pending (no grader record)")
            elif gterm is not None:
                gterm["n_pending"] = n_pending
                if n_unchecked:
                    pending.append(f"GOF pending on {n_unchecked} of {len(records)} datasets")
        elif not checked and not faults:
            pending.append(f"{pending_note(task)} pending (no post-stage record)")
        elif n_unchecked:
            pending.append(f"{pending_note(task)} pending on {n_unchecked} of {len(records)} datasets")
        pending += measurement_fault_reasons(faults, len(records), {r["seed"] for r in records if record_ok(r)})
        pending += peak_pending_reasons(task, missed)
        terms = [("completeness", V3_W_SYST, completeness_term(systc or None)),
                 *((("gof", V3_W_GOF, gterm),) if refit else ()), ("peak", V3_W_PEAK, peak_term(peakc, rule(task)))]
        v3 = score_v3(v2, terms, len(records), comp=comp, comp_gate=compute_gate_params(task),
                      comp_gate_source=compute_gate_source(task), pending=pending)
        groups = list(dict.fromkeys(built["groups"].values()))
        weights = {"phys": 1.0, **({"completeness": V3_W_SYST} if systc else {}),
                   **({"gof": V3_W_GOF} if v3.get("gof") else {}),
                   **({"peak": V3_W_PEAK} if (v3.get("peak") or peak_base) else {}), "compute": V3_W_COMP}
        v3["fallbacks"] = fallback_block(fallbacks(missed), built["cases"])
        v3.update(rule=function_rule(task), weights=weights, gof_pending=bool(refit and gofc is None),
                  measurement={"rule": MEASUREMENT_RULE, "verify_rule": verify_rule(task), "n_checked": len(rows),
                               "n_pending": n_pending, "n_evaluator_fault": len(faults),
                               "required": getattr(task, "measurement_required", True), "refit": refit},
                  peak_pending=not v3.get("peak"), peak_baseline_problem=peak_why)
        yardstick_weight(v3, yard)
        if v3.get("peak"):
            v3["peak"]["n_pending"] = sum(record_ok(r) and "peak" not in (rows.get(r["seed"]) or {})
                                          for r in records)
            v3["peak"]["n_no_baseline"] = sum(r["seed"] not in peak_base for r in records)
            v3["peak"]["n_not_in_budget"] = len((missed or {}).get("not_in_budget") or [])
            v3["peak"]["n_budget_reproduced"] = len({m["seed"] for m in (missed or {}).get("not_in_budget") or []
                                                     if m.get("reproduced")})
            v3["peak"]["n_not_made"] = len((missed or {}).get("not_made") or [])
        v3["phys"]["label"] = f"Corrections ({', '.join(groups)})"
    done = [r for r in records if record_ok(r)]
    n_seeds = len(records) + len(lost)
    public = {
        "label": label, "suite": manifest["suite"], "task": task.name,
        "n_seeds": n_seeds, "n_completed": len(done),
        "sigma_ref": sigma_ref, "sigma_ref_source": src,
        "wall_s_mean": float(np.mean([r["wall_s"] for r in records if r.get("wall_s") is not None] or [0.0])),
        "isolation": records[0].get("isolation", {}),
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "s_phys": None, "recovery": len(done) / n_seeds, "fdr": None, "n_null": 0, "crosstalk": [], "pois": {},
        "v2": v2, "v3": v3, "cluster_bias": {}, "calibration_response": {"leakage": {}, "declared_yield": None},
        "compute": comp, "discovery_power": None, "nuisances": {}, "tripwire": None, "attackers_for_scale": {},
        "per_seed_status": [{"seed": r["seed"], "completed": record_ok(r), "wall_s": r.get("wall_s"),
                             "errors": r.get("result_errors", [])} for r in records]
        + [{"seed": sd, "completed": False, "wall_s": None, "errors": [PLACEHOLDER_REASON], "missing": True}
           for sd in lost],
        "functions": {**function_summary(evals, built["cases"], function_rule(task)),
                      "baseline": task.function_baseline_id(),
                      "accuracy": ({"rule": ACCURACY_RULE, "factors": accuracy} if accuracy is not None else None)},
    }
    by_seed: Dict[str, List[dict]] = {}
    for c in built["cases"]:
        by_seed.setdefault(c["seed"], []).append(c)
    private = {"label": label, "v2_cases": built["cases"], "reference_pois": ref_pois if making_baseline else None,
               "v3_calibration_cases": None,
               "v3_completeness_cases": systc or None, "v3_gof_cases": gofc, "fit_rule": None, "fit_info": None,
               "v4_truth_cases": None, "v4_claims": None, "function_rule": function_rule(task),
               "v4_peak_cases": peakc, "peak_baseline": peak_base if making_baseline else None,
               "v4_peak_missed": missed if (missed or {}).get("not_in_budget") or (missed or {}).get("not_made")
               else None,
               "peak_baseline_missing": ([{"seed": sd, "flavour": fl, "why": v.get("why")}
                                          for sd, row in sorted(rows.items())
                                          for fl, v in sorted((row.get("peak") or {}).items()) if v.get("loss") is None]
                                         if making_baseline else None),
               "function_accuracy": accuracy_rec,
               "measurement_faults": {sd: fault_reason(r) for sd, r in sorted(faults.items())} or None,
               "function_unanswered": [{"seed": e["seed"], "role": e["spec"]["role"], "name": e["spec"]["name"],
                                        "flavour": e["spec"]["flavour"], "why": e["why"]}
                                       for e in evals if e["f"] is None],
               "per_seed": [
        {"seed": r["seed"], "truth": truths[r["seed"]], "completed": record_ok(r),
         "scoring_truth": {c["poi"]: c["truth_mean"] for c in by_seed.get(r["seed"], [])},
         "pois": {c["poi"]: {"value": c["value_mean"], "stat": 0.0, "syst": 0.0, "total": c["sigma_eff"]}
                  for c in by_seed.get(r["seed"], []) if not c["fallback"]},
         "pulls": {c["poi"]: c["zbar"] for c in by_seed.get(r["seed"], []) if not c["fallback"]},
         "discovery": None, "significance_obs": None} for r in records]}
    return public, private


def scoreboard_rows(summaries: List[dict]) -> List[str]:
    """The scoreboard's function table: per run and role the cases, answered, mean charge, the bias term B_p, mean
    pull, pull RMS, response slope, pinned; an unranked run (sbh.score.unranked_reason) shows no S."""
    from ...score import _f, unranked_reason
    fns = [s for s in summaries if s.get("functions")]
    if not fns:
        return []
    L = ["", "| system | S_v3 | role | n | answered | mean Δ | B_p | mean z̄ | pull RMS | slope ± err | pinned |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in fns:
        v3, per = s.get("v3"), (s.get("v2") or {}).get("per_poi") or {}
        for r, m in s["functions"]["roles"].items():
            sv3 = "unranked" if unranked_reason(v3) else _f(v3['s_v3'] if v3 else None, '{:.3f}')
            L.append(f"| {s['label']} | {sv3} | {r} | {m['n']} "
                     f"| {m['n_answered']} | {_f((per.get(r) or {}).get('delta_mean'), '{:+.2f}')} "
                     f"| {_f(((per.get(r) or {}).get('multiworld') or {}).get('bias'), '{:+.3f}')} "
                     f"| {_f(m['zbar_mean'], '{:+.2f}')} | {_f(m['pull_rms'])} "
                     f"| {_f(m['slope'], '{:+.2f}')} ± {_f(m['slope_se'])} | {m['n_pinned']} |")
    return L


def score_run(run_dir: Path, suite_dir: Path, task, label: str, sigma_ref: Optional[Dict[str, float]] = None,
              making_baseline: bool = False) -> dict:
    """Score a corrections run against the suite's truths, written by sbh.score.write_scores."""
    run_dir, suite_dir = Path(run_dir), Path(suite_dir)
    manifest = read_json_maybe_shielded(suite_dir / "manifest.json")
    truths = {s["seed"]: s["truth"] for s in manifest["seeds"]}
    records = load_scored_records(run_dir, manifest)
    if all(r.get("placeholder") for r in records):
        raise SystemExit(f"no run records under {run_dir}/runs")
    lost = tuple(r["seed"] for r in records if r.get("placeholder"))
    public, private = score_functions(run_dir, suite_dir, task, label, sigma_ref, manifest, truths,
                                      [r for r in records if not r.get("placeholder")], making_baseline, lost)
    write_scores(run_dir, suite_dir, task, public, private, records, None)
    return public
