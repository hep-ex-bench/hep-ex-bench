"""The corrections kind's post stage -- EVALUATOR-ONLY (sbh/harbor_pack.py EVALUATOR_ONLY): the check of the declared
measurements (MEASUREMENT.json) and the closure comparison (comparison.py) of a run's corrections.

Per completed dataset, at the points its corrections are scored at: refit.verify (the measurement checks and, under
CorrectionsTask.measurement_refit, the refit, its goodness of fit and its comparison with the reported corrections),
then the closure comparison, written to runs/<seed>/measurement_verify.json (VERIFY_FILE, its "peak" entry whatever
the comparison).  The scorer reads it: the closure term (Delta_peak), the gate (measurement_required), the GOF term
(measurement_refit); the rest is shown.
- Each seed is isolated as in the binned_likelihood verifier: an evaluator-side error leaves that seed pending.  It is
  written as the file's row (fault_row): status "verify error: <error>", the error in "evaluator_error", stamped
  "errors_marked" as every row, under the current rules (so the scorer reads it and leaves the dataset's terms out as
  an evaluator fault); every row counts the post stage's attempts on the dataset (grader_attempts,
  binned_likelihood verify.count_attempt, carried from the file it replaces).  Before a dataset is checked its
  placeholder row is written (unfinished_row: "verify error: grading did not finish", the attempt counted), so a post
  stage process that dies (killed, out of memory) leaves an evaluator fault, counted and retried as any other, never a
  dataset with no row.  Once every asked dataset is checked, a dataset whose fault has not recurred is checked once
  more (verify.to_retry; a fault seen twice is reproduced), unless the fault is systemic (verify.is_systemic over the
  run's completed datasets' files as they then stand: its attempts counted apart, verify.mark_systemic, never
  retried).  With several datasets in one process that decision is over all of their first attempts, whatever their
  order (sbh.cli's --phase post and all check theirs as one batch, see batch); one dataset per job (the post array)
  decides from the files as they stand, and the score phase settles the marks over the run's final files
  (settle_systemic).
- A file made under the current rules is skipped (force redoes it), unless it records a fault; one made under other
  rules is redone.
- Each step runs within its own budget (budget_of): a measurement not verified in time fails as one failing a check
  and the comparison is still made; a comparison not made in time fails.  The file keeps the comparison's attempts on
  the dataset under its rule (peak_attempts: each one's budget and whether it ran out): a comparison that runs out
  again on a rerun at a larger budget is escalated by the scorer (scoring.budget_reproduced).

Run per dataset between the runs and the score: `sbh eval|closure|stress --phase post` (Kind.post_dataset),
or by hand through measurement_verify.py.
"""
from __future__ import annotations

import contextlib
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Set

from . import comparison
from . import peak as P
from . import refit as M
from ...budget import BudgetExceeded, budget
from ...score import _default, load_records, read_json_maybe_shielded, record_ok


def cases_of(task, truth: dict) -> dict:
    """{(name, flavour): {"points", "space"}} of a dataset's scored cases (Task.function_cases)."""
    return {(sp["name"], sp["flavour"]): {"points": sp["points"], "space": sp["space"]}
            for sp in task.function_cases(truth)}


PEAK_NOT_IN_BUDGET = "not computed within the evaluation's budget"


def budget_factors(task, run_dir: Optional[Path] = None):
    """(verify_budget_y, peak_budget_y) in units of the reference's wall, 0 -> None; times evaluator_budget_x for a
    further reference's run that sets its own (Task.own_budget, keyed by the run directory's name)."""
    own = task.own_budget(Path(run_dir).name) if run_dir is not None and hasattr(task, "own_budget") else {}
    x = float(own.get("evaluator_budget_x", 1.0))
    return (float(getattr(task, "verify_budget_y", 0) or 0) * x or None,
            float(getattr(task, "peak_budget_y", 0) or 0) * x or None)


def reference_wall_of(run_dir: Path, task) -> Optional[float]:
    """The reference's wall on a dataset as this evaluation timed it (the run's compute yardstick: Y x k), or None."""
    from ...score import run_yardstick
    from ...yardstick import reference_wall
    yard = run_yardstick(run_dir, task)
    return reference_wall(yard) if yard is not None else None


def budget_of(run_dir: Path, task):
    """({"verify": s, "peak": s}, the record): budget_factors x reference_wall_of per step, or None where the run has
    no usable yardstick (recorded; refit.current redoes the stage once it has one) or the task states none -- then
    only the job's time limit bounds the step."""
    vy, py = budget_factors(task, run_dir)
    ref = reference_wall_of(run_dir, task)
    rec = {"verify_y": vy, "peak_y": py, "reference_wall_s": ref}
    if not ref:
        return {"verify": None, "peak": None}, dict(rec, why="no usable yardstick record: no budget applied")
    out = {"verify": vy * ref if vy else None, "peak": py * ref if py else None}
    return out, dict(rec, verify_s=out["verify"], peak_s=out["peak"])


def _ran_out(peak) -> bool:
    """Whether a comparison (peak_closure's {flavour: ...}) ran out of its budget on any flavour."""
    return isinstance(peak, dict) and any(str((pk or {}).get("why") or "").startswith(PEAK_NOT_IN_BUDGET)
                                          for pk in peak.values() if isinstance(pk, dict))


def peak_attempts(prev: Optional[dict], peak, peak_s: Optional[float], task) -> list:
    """The comparison's attempts on this dataset, this one last: [{"peak_s": its budget in seconds (None: none
    applied), "not_in_budget": whether it ran out}], carried on from the file this one replaces (``prev``) where that
    file was made under the same comparison rule (comparison.rule); a file written before the attempts were kept counts
    as one attempt, read from its own budget and comparison."""
    rule = comparison.rule(task)
    before = []
    if isinstance(prev, dict) and prev.get("peak_rule") == rule:
        before = [a for a in prev.get("peak_attempts") or [] if isinstance(a, dict)]
        if not before and isinstance(prev.get("peak"), dict):
            before = [{"peak_s": (prev.get("budget") or {}).get("peak_s"), "not_in_budget": _ran_out(prev["peak"])}]
    return before + [{"peak_s": peak_s, "not_in_budget": _ran_out(peak)}]


def peak_closure(task, out_dir: Path, ds: Path, cache: Path, index: int, checked: dict,
                 answered: Dict[str, Dict[str, bool]], over: Optional[str] = None) -> dict:
    """{flavour: the closure comparison (comparison.module)} for every requested flavour the task's peak_spec covers.
    A correction not
    reported (`answered`: peak_answers) or, under measurement_required, whose measurement failed (`checked`:
    refit.verify's corrections) enters at the request's no-answer forecast; "loss_fallback" is the loss with every
    one at its forecast (the case's cap).  {"why"} -- a failed comparison for the scorer -- where the corrections are
    unreadable or the budget ran out (`over`: already out, nothing computed).  peak.py writes its candidate cache with
    the alarm held, so a budget never leaves a part-written file (peak.save_cached)."""
    from . import outputs as C
    from ...contract import read_output
    if not hasattr(task, "peak_spec"):
        return {}
    request = json.loads((ds / "request.json").read_text())
    cset, why = C.read_set(out_dir / C.FILE)
    try:
        result = json.loads(read_output(out_dir / "RESULT.json", max_bytes=C.MAX_BYTES))
    except (OSError, ValueError, RecursionError):
        result = None
    out = {}
    gate = getattr(task, "measurement_required", True)
    flavours = {fl for e in (request.get("corrections") or {}).values() for fl in ((e or {}).get("domain") or {})}
    X = comparison.module(task)
    for fl in sorted(flavours):
        spec = task.peak_spec(fl, request)
        if spec is None:
            continue
        if over:
            out[fl] = {"why": over}
            continue
        if cset is None:
            out[fl] = {"why": f"{C.FILE}: {why}"}
            continue
        try:
            cand = X.candidates(ds, spec, fl, cache)
            fb = {n: v for n, v in spec["fallback"].items()
                  if not (answered.get(fl) or {}).get(n, False)
                  or (gate and (checked.get(f"{n}|{fl}") or {}).get("status") != "ok")}
            r = X.closure(spec, cand, cset, result, request, fl, index, fb)
            r["fallback"] = sorted(fb)
            r["loss_fallback"] = X.fallback_loss(spec, cand, request, fl, index, cache)
        except BudgetExceeded as e:
            over = f"{PEAK_NOT_IN_BUDGET} ({e})"
            out[fl] = {"why": over}
            continue
        if "id" in spec:
            r["spec_id"] = spec["id"]
        out[fl] = r
    return out


def peak_answers(task, out_dir: Path, ds: Path) -> Dict[str, Dict[str, bool]]:
    """{flavour: {name: answered}} for the closure comparison's corrections (peak_spec's fallback) by the physics term's
    rule (_answered); made before the comparison's budget starts, so that budget is the comparison's own."""
    from . import outputs as C
    from ...contract import read_output
    if not hasattr(task, "peak_spec"):
        return {}
    request = json.loads((ds / "request.json").read_text())
    cset, _ = C.read_set(out_dir / C.FILE)
    try:
        result = json.loads(read_output(out_dir / "RESULT.json", max_bytes=C.MAX_BYTES))
    except (OSError, ValueError, RecursionError):
        result = None
    out: Dict[str, Dict[str, bool]] = {}
    flavours = {fl for e in (request.get("corrections") or {}).values() for fl in ((e or {}).get("domain") or {})}
    for fl in sorted(flavours):
        spec = task.peak_spec(fl, request)
        if spec is not None:
            out[fl] = {n: cset is not None and _answered(C, cset, result, request, n, fl) for n in spec["fallback"]}
    return out


def _answered(C, cset: dict, result, request: dict, name: str, flavour: str) -> bool:
    """Whether the run reports `name` for `flavour` as the contract requires (outputs.flavour_problems on the domain
    grid, the scorer's function_evaluations rule)."""
    corr, _ = C.find(cset, name)
    dom = ((C.requested(request).get(name) or {}).get("domain") or {}).get(flavour)
    if corr is None or dom is None:
        return False
    try:
        return not C.flavour_problems(corr, result, {flavour: dom}, signed=name in P.RESOLUTIONS, cset=cset)[flavour]
    except C.CorrectionError:
        return False


def fault_row(seed: str, task, e: BaseException, budget_rec: dict, seconds: float, made: Optional[dict]) -> dict:
    """The file's row for a dataset the post stage itself failed on (an exception of the evaluator's code, ``e``):
    {"seed", "status": "verify error: <error>", "evaluator_error": "<error> (at <where>)", "rule", "peak_rule",
    "n_toys", "budget", "seconds", "peak_attempts" (the comparison's earlier attempts, carried from ``made`` under the
    same rule; none made this time)}, stamped (verify.mark_errors).  It is current (refit.current), so the scorer
    reads it, and it holds no measurement, correction or comparison: the dataset is not checked."""
    from ..binned_likelihood.verify import mark_errors
    tb = traceback.extract_tb(e.__traceback__)[-3:]
    where = " <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in reversed(tb))
    msg = f"{type(e).__name__}: {e}"
    return mark_errors({"seed": seed, "status": f"verify error: {msg}"[:600],
                        "evaluator_error": f"{msg} (at {where})"[:600], "rule": M.verify_rule(task),
                        "peak_rule": comparison.rule(task), "n_toys": M.N_TOYS, "budget": dict(budget_rec),
                        "seconds": seconds, "peak_attempts": peak_attempts(made, None, None, task)[:-1]})


def unfinished_row(seed: str, task, budget_rec: dict, made: Optional[dict], before: dict) -> dict:
    """The placeholder row written before a dataset is checked (binned_likelihood verify.unfinished_record, its attempt
    counted on from ``before``), under the current rules as fault_row's (refit.current: the scorer reads it, and a later
    post stage redoes it); it stands where the process dies before the dataset's own row is written."""
    from ..binned_likelihood.verify import unfinished_record
    return unfinished_record(seed, before, rule=M.verify_rule(task), peak_rule=comparison.rule(task),
                             n_toys=M.N_TOYS, budget=dict(budget_rec), seconds=0.0,
                             peak_attempts=peak_attempts(made, None, None, task)[:-1])


def check_one(run_dir: Path, suite_dir: Path, task, s: dict, budgets: dict, budget_rec: dict,
              made: Optional[dict]) -> dict:
    """The post stage on one completed dataset (``s``: its manifest entry; ``made``: the file it replaces, or None) ->
    its row: the measurement checks and the closure comparison, each within its budget, status "ok" and stamped; an
    exception of the evaluator's own code there is the dataset's fault_row."""
    from ..binned_likelihood.verify import mark_errors
    seed = s["seed"]
    ds = suite_dir / ("public" if s["public"] else "sealed") / seed
    out_dir, cache = run_dir / "runs" / seed / "out", suite_dir / "cache" / seed
    t0 = time.time()
    res = peak = None
    exceeded = {}
    log = (lambda m: print(f"[{seed} {time.time() - t0:5.0f} s] {m}", flush=True))
    try:
        try:
            with budget(budgets["verify"], f"the measurement checks of {seed}"):
                try:
                    res = M.verify(out_dir, ds, cases_of(task, s["truth"]), seed=s["index"], log=log,
                                   refit=getattr(task, "measurement_refit", True))
                except BudgetExceeded as e:
                    res = dict(M.unverified(ds, f"{M.NOT_IN_BUDGET} ({e})"), seconds=time.time() - t0)
        except BudgetExceeded as e:
            if res is None:
                res = dict(M.unverified(ds, f"{M.NOT_IN_BUDGET} ({e})"), seconds=time.time() - t0)
            exceeded["verify"] = res.pop("budget_exceeded", None) or str(e)
        t1 = time.time()
        answered = peak_answers(task, out_dir, ds)
        t_peak = time.time()
        try:
            with budget(budgets["peak"], f"the {comparison.noun(task)} comparison of {seed}"):
                peak = peak_closure(task, out_dir, ds, cache, s["index"], res["corrections"], answered)
        except BudgetExceeded as e:
            if peak is None:
                peak = peak_closure(task, out_dir, ds, cache, s["index"], res["corrections"], answered,
                                    over=f"{PEAK_NOT_IN_BUDGET} ({e})")
            exceeded["peak"] = str(e)
        t2 = time.time()
    except Exception as e:  # noqa: BLE001
        row = fault_row(seed, task, e, budget_rec, time.time() - t0, made)
        print(f"{seed}: verify error: {row['evaluator_error'][:400]}", flush=True)
        return row
    if exceeded:
        log("budget: " + "; ".join(f"{k}: {v}" for k, v in exceeded.items()))
    res.update(status="ok", peak=peak, seed=seed, rule=M.verify_rule(task), peak_rule=comparison.rule(task),
               n_toys=M.N_TOYS,
               budget=dict(budget_rec, verify_seconds=t1 - t0, peak_seconds=t2 - t_peak, exceeded=exceeded or None),
               peak_attempts=peak_attempts(made, peak, budgets["peak"], task))
    return mark_errors(res)


_BATCH = []


@contextlib.contextmanager
def batch(run_dir: Path, suite_dir: Path, task):
    """A batch of one run's post stage in this process (sbh.cli's --phase post and all): check on that run's asked
    seeds within it only names them, and the batch's datasets are checked as one when it closes (check: every one
    once, then the retries).  An exception inside it checks none."""
    entry = [str(Path(run_dir)), set()]
    _BATCH.append(entry)
    try:
        yield
    except BaseException:
        _BATCH.remove(entry)
        raise
    _BATCH.remove(entry)
    if entry[1]:
        check(run_dir, suite_dir, task, entry[1])


def check(run_dir: Path, suite_dir: Path, task, seeds: Optional[Set[str]] = None, force: bool = False) -> int:
    """The post stage on a run's completed datasets (`seeds`, e.g. {"seed_003"}; None: all): check_one per dataset after
    its placeholder row (unfinished_row), its attempts counted (count_attempt); then, once every one is checked, each
    whose evaluator error has not recurred (to_retry) once more, unless its fault is systemic over the run's rows as
    they then stand (is_systemic; marked, mark_systemic), a retried row whose fault turns out systemic by the time its
    retry ends marked too (module docstring).  Returns the number checked (a fault row included)."""
    from ..binned_likelihood.verify import (RETRIED_KEY, SYSTEMIC_FAULT, attempts_of, completed_seeds, count_attempt,
                                            mark_systemic, seed_records, systemic_places, to_retry)
    run_dir, suite_dir = Path(run_dir), Path(suite_dir)
    for entry in reversed(_BATCH):
        if entry[0] == str(run_dir) and seeds is not None and not force:
            entry[1] |= set(seeds)
            return 0
    if getattr(task, "kind", None) != "corrections":
        raise SystemExit(f"measurement_verify: task {task.name} has no corrections (nothing to check)")
    manifest = read_json_maybe_shielded(suite_dir / "manifest.json")
    entries = {s["seed"]: s for s in manifest["seeds"]}
    budgets, budget_rec = budget_of(run_dir, task)
    print(f"post stage budgets per dataset: {budget_rec}", flush=True)
    recs = (load_records(run_dir) if seeds is None else
            [json.loads(f.read_text()) for f in (run_dir / "runs" / s / "record.json" for s in sorted(seeds))
             if f.is_file()])
    first = {}
    for rec in recs:
        seed = rec["seed"]
        target = run_dir / "runs" / seed / M.VERIFY_FILE
        made = None
        if target.exists():
            try:
                made = json.loads(target.read_text())
            except (OSError, ValueError):
                made = None
            if not force:
                if M.current(made, task, run_dir) and made.get("status") == "ok":
                    print(f"{seed}: {M.VERIFY_FILE} exists, made under the current rules (--force to redo)")
                    continue
                print(f"{seed}: {M.VERIFY_FILE} "
                      + ("records a fault of the post stage" if M.current(made, task, run_dir)
                         else "was made under other rules") + ": redone")
        if not record_ok(rec):
            print(f"{seed}: the run did not complete; nothing to check")
            continue
        t0 = time.time()
        before = {seed: attempts_of(made)} if isinstance(made, dict) else {}
        _write_row(target, unfinished_row(seed, task, budget_rec, made, before))
        row = count_attempt(check_one(run_dir, suite_dir, task, entries[seed], budgets, budget_rec, made), before)
        _write_row(target, row)
        first[seed] = (row, made, target, t0)
    done = 0
    for seed, (row, made, target, t0) in first.items():
        if to_retry(row):
            systemic = systemic_places(seed_records(run_dir, M.VERIFY_FILE), completed_seeds(run_dir))
            if mark_systemic(row, systemic):
                print(f"{seed}: {SYSTEMIC_FAULT} ({', '.join(f'{p} on {n}' for p, n in systemic.items())} of the "
                      "run's completed datasets): not checked again", flush=True)
            else:
                print(f"{seed}: checked once more (the evaluator's error has not recurred)", flush=True)
                row = count_attempt(check_one(run_dir, suite_dir, task, entries[seed], budgets, budget_rec, made),
                                    {seed: row["grader_attempts"]})
                row[RETRIED_KEY] = True
                if mark_systemic(row, systemic_places(seed_records(run_dir, M.VERIFY_FILE, {seed: row}),
                                                      completed_seeds(run_dir))):
                    print(f"{seed}: {SYSTEMIC_FAULT}, shown by the run's rows once its retry ended", flush=True)
            _write_row(target, row)
        if row.get("status") == "ok":
            bad = sorted(k for k, c in row["corrections"].items() if c["status"] != "ok")
            print(f"{seed}: {len(row['measurements'])} measurements, {len(row['corrections']) - len(bad)} cases "
                  f"(correction|flavour) verified, {len(bad)} not{': ' + ', '.join(bad) if bad else ''} "
                  f"({row['seconds']:.0f} s) -> {target}")
        else:
            print(f"{seed}: not checked, an evaluator fault (attempts {row['grader_attempts']}, "
                  f"{time.time() - t0:.0f} s) -> {target}")
        done += 1
    print(f"measurement_verify: {done} seed(s) checked in {run_dir}")
    return done


def settle_systemic(run_dir: Path, task=None) -> list:
    """The score phase's step before the score (sbh.cli): every row of the run whose fault is systemic over the run's
    final rows marked (binned_likelihood verify.settle_systemic) and written; nothing checked.  Returns the seeds
    marked."""
    from ..binned_likelihood.verify import completed_seeds, seed_records, settle_systemic as settle
    run_dir = Path(run_dir)
    rows = seed_records(run_dir, M.VERIFY_FILE)
    marked = settle(rows, completed_seeds(run_dir))
    for seed in marked:
        _write_row(run_dir / "runs" / seed / M.VERIFY_FILE, rows[seed])
    return marked


def _write_row(target: Path, row: dict) -> None:
    """Write one dataset's row atomically."""
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, indent=1, default=_default))
    tmp.replace(target)
