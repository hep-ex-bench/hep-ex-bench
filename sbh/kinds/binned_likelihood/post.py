"""The binned_likelihood kind's post-score steps (`sbh post-score`) -- EVALUATOR-ONLY -- and the per-dataset model
verification cap (opt-in, Task.model_verify_cap): the grader stage's rule, twice the closure reference's slowest dataset;
a model that does not evaluate within it is a failed measurement.

Order: `sbh closure` -> `sbh post-score --label reference` (verifies the reference uncapped, records walls, writes the
cap into closure.json) -> later runs verify within the cap, except datasets graded through combine's battery (not
the path the closure measured), which run uncapped."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Optional, Sequence

from ...suite import CLOSURE_LABEL


CAP_RULE = "2026-10-06: twice the closure reference's slowest dataset (its model verification's wall)"
CAP_FACTOR = 2.0
CAP_KEY = "model_verify_cap"


def verify_toys(task) -> int:
    """The model verification's toys per dataset: the task's model_verify_toys, SBH_VERIFY_TOYS when set (a quicker
    check)."""
    return int(os.environ.get("SBH_VERIFY_TOYS") or task.model_verify_toys)


def post_score(run_dir, suite_dir, task, label) -> List[str]:
    """Post-score steps, from the repo root: (1) model validation (sbh.kinds.binned_likelihood.verify: refits,
    coverage toys, GOF; docs/MODEL_VALIDATION.md) on the JAX engine (SBH_TOY_ENGINE=pyhf to compare) with the task's
    model_verify_toys toys (SBH_VERIFY_TOYS overrides them for a quicker check; MODEL_COMBINE/ seeds go to combine),
    within verify_cap -- on the closure's reference of a task scored on the mass-marginal mu also the prior curve
    (--prior-curve); (1a) the same verification once more on every completed dataset whose record carries an
    evaluator error on a scored block (verify --stage retry, the same flags: a one-off clears, a deterministic fault is
    reproduced before the run is first scored, never shown with its cases left out as if a rerun could clear them; a
    fault whose places all lie where a systemic one sits (the same place on more than half of the completed datasets)
    is not verified again and is counted apart, verify.SYSTEMIC_FAULT, so the job's wall never doubles; after a
    verification that failed, and on a task kept on the rule it shipped with, not run), and then (1b) its mu
    baseline into closure.json (refresh_closure_baseline) BEFORE anything is rescored against it; (2) sbh/rescore.py
    folds GOF into the score.  Every other step runs even after one fails.  Returns the required steps (1, 1a, 1b, 2)
    that failed.  The previous verification is moved to <file>.prev first, so a verification that fails never leaves
    an older one to be scored as this attempt's."""
    import subprocess
    import sys
    from ... import ROOT
    from ...score import SHIPPED_RULE, score_rule
    from ...suite import marginal_gate
    from .scoring import VERIFY_FILE
    env = {**os.environ, "SBH_TOY_ENGINE": os.environ.get("SBH_TOY_ENGINE") or "jax"}
    baseline = label == CLOSURE_LABEL and marginal_gate(task)
    verify = [sys.executable, "-m", "sbh.kinds.binned_likelihood.verify", "--run-dir", str(run_dir),
              "--toys", str(verify_toys(task)), "--suite", str(suite_dir), "--out", VERIFY_FILE]
    verify += ["--prior-curve"] if baseline else []
    retry = () if score_rule(task) == SHIPPED_RULE else \
        (("fault re-verification", True, verify + ["--stage", "retry"], None),)
    steps = (("model verification", True, verify, None), *retry,
             ("rescore", True,
              [sys.executable, "-m", "sbh.rescore", "--suite", str(suite_dir), "--task", task.name,
               "--work", str(Path(run_dir).parent.parent), "--glob", label], None))
    prev = Path(run_dir) / VERIFY_FILE
    if prev.exists():
        os.replace(prev, prev.with_name(prev.name + ".prev"))
    failed = []

    def report(name, required, why):
        print(f"post-score: {name} failed ({why})" + ("" if required else "; optional, not failing the step"),
              flush=True)
        if required:
            failed.append(f"{name} ({why})")

    for name, required, cmd, limit in steps:
        if name == "fault re-verification" and failed:
            print(f"post-score: {name} not run (the model verification failed)", flush=True)
            continue
        if name == "rescore" and baseline:
            why = refresh_closure_baseline(suite_dir, run_dir, task, verified=not failed)
            if why:
                report("closure baseline", True, why)
        budget = float(os.environ.get(limit[0]) or limit[1]) if limit else None
        try:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, timeout=budget).returncode
            why = f"exit {rc}" if rc else None
        except subprocess.TimeoutExpired:
            why = f"stopped after its {budget:.0f} s budget ({limit[0]})"
        except OSError as e:
            why = f"not run: {type(e).__name__}: {e}"
        if why:
            report(name, required, why)
    return failed


def refresh_closure_baseline(suite_dir, run_dir, task, verified=True) -> Optional[str]:
    """Write the grader's signal-POI baseline into the suite's closure.json from the reference run's VERIFY_FILE: the
    kind's closure_fields (reference_pois[seed][mu] = the posterior's mean, sd and H with its prior curve, the
    suite-average curve, mu_rule, scan_version, baseline_source "grader"), every dataset's earlier mu entry and the
    earlier baseline's keys dropped first (sbh.suite.update_closure).  post_score calls it between the verify and the
    rescore steps, so the reference's own rescore and every later run are scored against the baseline this evaluator
    just made, never a stale closure's and never RESULT.json's (`sbh closure` writes none for the gated POI).
    ``verified`` False (the verification failed) or fields without a grader baseline (no scan in the file) write the
    pending state instead -- no mu entry, baseline_source "pending" -- so no run is scored until a verification
    succeeds (sbh.suite.closure_gate).  A closure.json of no or another mu rule is left as it is: `sbh closure` remakes
    it first (its criteria and summary are another scorer's).  Returns why no grader baseline was written (printed by
    the caller), None when it was."""
    from ...kinds import get_kind
    from ...suite import (BASELINE_GRADER, BASELINE_PENDING, baseline_keys, baseline_pois, closure_gate, mu_rule,
                          update_closure)
    stale = closure_gate(suite_dir, task, CLOSURE_LABEL)
    if stale:
        return stale
    if verified:
        fields = get_kind(task).closure_fields(run_dir, suite_dir, task)
    else:
        fields = {"mu_rule": mu_rule(), "baseline_source": BASELINE_PENDING,
                  "baseline_note": "the reference's model verification failed: sbh post-score --label "
                                   f"{CLOSURE_LABEL} again"}
    src = fields.get("baseline_source")
    if src != BASELINE_GRADER:
        fields = {**fields, "baseline_source": BASELINE_PENDING}
    if not update_closure(suite_dir, fields, replace_pois=baseline_pois(task), replace_keys=baseline_keys(task)):
        return f"no closure.json in {Path(suite_dir).name} to hold it (sbh closure first)"
    if src != BASELINE_GRADER:
        why = fields.get("baseline_note") or fields.get("baseline_why")
        return (f"{Path(run_dir).name}'s verification gives no grader baseline, closure.json says pending "
                f"(baseline_source {src!r}{': ' + str(why)[:200] if why else ''})")
    sp = baseline_pois(task)[0]
    n = sum(1 for e in (fields.get("reference_pois") or {}).values() if sp in (e or {}))
    excl = (fields.get("mu_baseline") or {}).get("excluded") or {}
    print(f"closure baseline: the grader's {sp} on {n} dataset(s)"
          + (f"; {len(excl)} without one, excluded from every run's {sp} cases:" if excl else ""), flush=True)
    for s, why in sorted(excl.items()):
        print(f"  {s}: {str(why)[:160]}", flush=True)
    return None


def closure_cap(run_dir, mv: Optional[dict] = None) -> dict:
    """closure.json's CAP_KEY block: CAP_FACTOR x the slowest ok dataset's verify_wall_s in the reference's
    VERIFY_FILE, rounded up to 0.1 s, with its toys and engine, stamped CAP_RULE; cap_s None plus why if untimed.
    ``mv``: the reference's verification in that file's form ({"verify_cap", "per_seed"}), from a kind whose grader
    writes one record per dataset."""
    from .scoring import VERIFY_FILE
    none = {"rule": CAP_RULE, "cap_s": None}
    try:
        mv = json.loads((Path(run_dir) / VERIFY_FILE).read_text()) if mv is None else mv
    except FileNotFoundError:
        return dict(none, why="the closure's reference run has no model verification yet (its job after the score: "
                              f"sbh post-score --label {CLOSURE_LABEL})")
    except (OSError, ValueError) as e:
        return dict(none, why=f"its {VERIFY_FILE} is unreadable ({type(e).__name__})")
    blk = mv.get("verify_cap")
    if not isinstance(blk, dict):
        return dict(none, why=f"its {VERIFY_FILE} records no dataset walls (verified before the task stated the cap): "
                              "rerun the reference's model verification")
    rows = [r for r in mv.get("per_seed") or [] if isinstance(r, dict)]
    walls = {r["seed"]: float(r["verify_wall_s"]) for r in rows
             if r.get("status") == "ok" and isinstance(r.get("verify_wall_s"), (int, float))}
    left_out = sorted(str(r.get("seed")) for r in rows if r.get("seed") not in walls)
    if not walls:
        return dict(none, why="no dataset of the closure's reference run was verified", left_out=left_out)
    slowest = max(walls, key=walls.get)
    cap = math.ceil(round(CAP_FACTOR * walls[slowest] * 10, 6)) / 10
    return {"rule": CAP_RULE, "cap_s": cap, "slowest_s": walls[slowest],
            "slowest_seed": slowest, "n_datasets": len(walls), "walls_s": walls, "left_out": left_out,
            "toys": blk.get("toys"), "engine": blk.get("engine")}


def verify_cap(run_dir, suite_dir, task, toys: int, engine: Optional[str],
               combine: Sequence[str] = ()) -> Optional[dict]:
    """Per-dataset cap for verifying `run_dir`: {"rule", "cap_s", "source", "why"[, "combine_exempt"]}.  cap_s is
    closure.json's cap, or None with why (this is the reference run; closure.json missing, pre-rule, other rule,
    unmeasured, or measured with other toys/engine).  `combine` datasets run uncapped.  None if the task has no cap."""
    if not getattr(task, "model_verify_cap", False):
        return None
    src = Path(suite_dir) / "closure.json"
    out = {"rule": CAP_RULE, "cap_s": None, "source": str(src), "why": None,
           **({"combine_exempt": sorted(combine)} if combine else {})}
    if Path(run_dir).name == CLOSURE_LABEL:
        return dict(out, why="the closure's reference run: its datasets' walls measure the cap (none capped)")
    try:
        cl = json.loads(src.read_text())
    except FileNotFoundError:
        return dict(out, why="no closure.json in the suite (sbh closure, then sbh post-score --label "
                             f"{CLOSURE_LABEL}, measure the cap)")
    except (OSError, ValueError) as e:
        return dict(out, why=f"closure.json is unreadable ({type(e).__name__})")
    blk = cl.get(CAP_KEY) if isinstance(cl, dict) else None
    if not isinstance(blk, dict):
        return dict(out, why=f"closure.json records no {CAP_KEY} (a closure made before the rule)")
    if blk.get("rule") != CAP_RULE:
        return dict(out, why=f"closure.json's {CAP_KEY} was made under another rule: {str(blk.get('rule'))[:160]}")
    if not isinstance(blk.get("cap_s"), (int, float)):
        return dict(out, why=f"closure.json's {CAP_KEY} is not measured: {str(blk.get('why'))[:300]}")
    if (blk.get("toys"), blk.get("engine")) != (toys, engine):
        return dict(out, why=f"closure.json's {CAP_KEY} was measured with {blk.get('toys')} toys on "
                             f"{blk.get('engine')}, this verification runs {toys} on {engine}")
    return dict(out, cap_s=float(blk["cap_s"]))
