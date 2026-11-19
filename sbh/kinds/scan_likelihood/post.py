"""The scan_likelihood kind's post stage -- EVALUATOR-ONLY: the grader (verify.py) on one completed dataset of a run,
between the runs and the score (`sbh eval|closure|stress --phase post`, Kind.post_dataset), written to
runs/<seed>/VERIFY_FILE, which the scorer reads.  The task's model_verify_toys toys (SBH_VERIFY_TOYS overrides them
for a quicker check); the dataset's truths from the suite's manifest pin the GOF toys.

A capped task (Task.model_verify_cap; binned_likelihood/post.py's rule): the closure's reference is graded uncapped
and its datasets' walls give closure.json's cap, twice the slowest (closure_cap, written by `sbh closure`); every
other run's dataset is graded within it, and a model not graded in time is failed on that dataset ("fit failed",
"ran_out").

Each record counts the grader's attempts on its dataset (grader_attempts, binned_likelihood verify.count_attempt,
carried from the record it replaces; one written before them counts an evaluator error only where it is stamped,
verify.attempts_of).  Before a dataset is graded its placeholder is written (verify.unfinished_record: "verify error:
grading did not finish", the attempt counted), so a grader process that dies (killed, out of memory) leaves an
evaluator fault, counted and retried as any other, never a dataset with no record.  A record whose evaluator error has
not recurred is graded once more before the score (verify.to_retry, as binned_likelihood's post stage does): a one-off
clears, a deterministic fault is reproduced (the scorer names it apart and shows the run unranked) before the run is
first scored; the first attempt's record stands if the job ends during the retry.  A systemic fault (verify.is_systemic:
every place the dataset faults lies where a fault sits on more than half of the run's completed datasets,
verify.systemic_places) is not retried: its attempts are counted apart (verify.mark_systemic), and the run stays
provisional until the evaluator is fixed and the run graded again.

Several datasets graded in one process (post_datasets; `--phase all`, or `--phase post` on several --seeds, grade
theirs as one batch, see batch) are graded in two passes, every dataset once and then the retries, so whether a fault is systemic is decided over all of their
first attempts, whatever their order.  One dataset per job (the Slurm post array) decides from the run's records as
they stand when its first attempt ends, and again after its retry; the score phase settles the marks over the run's
final records (settle_systemic), so the marks, as the scorer's reading (which decides over the final records itself),
never depend on the order the datasets were graded in.  Only an attempt's cost does: in a parallel array the earliest
faults of a systemic one may be graded once more before the majority shows.
"""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

from . import VERIFY_FILE

POST_SHAPE = "-c 1 --mem 8000M -t 4:00:00"
ENGINE = "jax"


def closure_cap(run_dir) -> dict:
    """closure.json's model-verify cap from the reference run's per-dataset records (binned_likelihood/post.py
    closure_cap on them); cap_s None plus why where none carries the cap's block."""
    from ..binned_likelihood.post import CAP_RULE, closure_cap as cap_of
    from .scoring import load_rows
    rows = list(load_rows(run_dir).values())
    blk = next((r["verify_cap"] for r in rows if isinstance(r.get("verify_cap"), dict)), None)
    if blk is None:
        return {"rule": CAP_RULE, "cap_s": None,
                "why": "the closure's reference run has no grader record yet (sbh closure --phase post)"}
    return cap_of(run_dir, {"verify_cap": blk, "per_seed": rows})


def previous_attempts(path: Path, seed: str) -> dict:
    """{seed: the grader attempts the record at ``path`` counts} (binned_likelihood verify.attempts_of: an unstamped
    record without grader_attempts counts one attempt and no evaluator error), {} without a readable record there."""
    from ..binned_likelihood.verify import attempts_of
    try:
        prev = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return {seed: attempts_of(prev)} if isinstance(prev, dict) else {}


def _grader(run_dir: Path, suite_dir: Path, task):
    """(grade(seed, before) -> the dataset's new record, write(seed, record)) of one post stage on ``run_dir``:
    verify.verify_dataset within the task's cap, its wall and cap recorded; a placeholder (verify.unfinished_record,
    its attempt counted on from ``before``) written before each dataset is graded."""
    from ...budget import BudgetExceeded, budget
    from ...score import _default, read_json_maybe_shielded, write_text_atomic
    from ..binned_likelihood.post import verify_cap, verify_toys
    from ..binned_likelihood.verify import mark_errors, unfinished_record
    from .verify import SCAN_VERSION, verify_dataset
    man = read_json_maybe_shielded(Path(suite_dir) / "manifest.json")
    truths = {s["seed"]: s["truth"] for s in man["seeds"]}
    n_toys = verify_toys(task)
    cap = verify_cap(run_dir, suite_dir, task, n_toys, ENGINE)
    cap_s = cap["cap_s"] if cap else None
    cap_blk = None if cap is None else {**cap, "toys": n_toys, "engine": ENGINE}

    def write(seed, r):
        write_text_atomic(Path(run_dir) / "runs" / seed / VERIFY_FILE, json.dumps(r, indent=1, default=_default))

    def grade(seed, before, placeholder=True):
        if placeholder:
            write(seed, unfinished_record(seed, before, scan_version=SCAN_VERSION,
                                          **({"verify_cap": cap_blk} if cap_blk else {})))
        t0 = time.monotonic()
        try:
            with budget(cap_s, f"the model verification of {seed}"):
                rec = verify_dataset(Path(run_dir) / "runs" / seed, task, truths[seed], n_toys)
        except BudgetExceeded:
            rec = mark_errors({"seed": seed, "scan_version": SCAN_VERSION, "ran_out": True,
                               "status": f"fit failed: the model verification's cap of {cap_s:g} s per dataset ran "
                                         f"out before this model was verified ({time.monotonic() - t0:.0f} s)"})
        rec["verify_wall_s"] = round(time.monotonic() - t0, 2)
        if cap_blk is not None:
            rec["verify_cap"] = dict(cap_blk)
        return rec
    return grade, write


_BATCH = []


@contextlib.contextmanager
def batch(run_dir: Path, suite_dir: Path, task):
    """A batch of one run's post stage in this process (sbh.cli's --phase post and all): post_dataset on that run within
    it only names its dataset, and the batch's datasets are graded as one when it closes (post_datasets: every one once,
    then the retries).  An exception inside it grades none."""
    entry = [str(Path(run_dir)), []]
    _BATCH.append(entry)
    try:
        yield
    except BaseException:
        _BATCH.remove(entry)
        raise
    _BATCH.remove(entry)
    if entry[1]:
        post_datasets(run_dir, suite_dir, task, entry[1])


def post_dataset(run_dir: Path, suite_dir: Path, task, seed, seeds=None) -> None:
    """The post stage on ``seed`` (within a batch of that run, only named: graded when the batch closes), or on several
    datasets as one stage (``seeds``, post_datasets): verify.verify_dataset on runs/<seed> of each completed dataset
    (none for one the run did not complete), within the task's cap, its record, wall and cap written to
    runs/<seed>/VERIFY_FILE with its grader_attempts counted on from the record there before (count_attempt), after its
    placeholder.  Then, once every one of them is graded, a record whose evaluator error has not recurred (to_retry) is
    graded once more, within the cap again, unless its fault is systemic over the run's records as they then stand
    (systemic_places, is_systemic; marked, mark_systemic); a retried record whose fault turns out systemic by the time
    its retry ends is marked too (module docstring).  The grader runs on one CPU; the caller's CPU affinity is restored
    after it, so the datasets a single-process evaluation runs next keep theirs."""
    import os
    from ...score import record_ok
    from ..binned_likelihood.cpu_pin import pin_single_cpu
    from ..binned_likelihood.verify import (RETRIED_KEY, SYSTEMIC_FAULT, completed_seeds, count_attempt,
                                            mark_systemic, seed_records, systemic_places, to_retry)
    run_dir = Path(run_dir)
    if seeds is None:
        for entry in reversed(_BATCH):
            if entry[0] == str(run_dir):
                entry[1].append(seed)
                return
        seeds = [seed]
    todo = []
    for s in seeds:
        try:
            rec = json.loads((run_dir / "runs" / s / "record.json").read_text())
        except (OSError, ValueError):
            continue
        if record_ok(rec):
            todo.append(s)
    if not todo:
        return
    cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    pin_single_cpu()
    try:
        grade, write = _grader(run_dir, suite_dir, task)
        first = {}
        for s in todo:
            before = previous_attempts(run_dir / "runs" / s / VERIFY_FILE, s)
            first[s] = count_attempt(grade(s, before), before)
            write(s, first[s])
        for s, out in first.items():
            if not to_retry(out):
                continue
            systemic = systemic_places(seed_records(run_dir, VERIFY_FILE), completed_seeds(run_dir))
            if mark_systemic(out, systemic):
                print(f"{s}: {SYSTEMIC_FAULT} ({', '.join(f'{p} on {n}' for p, n in systemic.items())} of the "
                      "run's completed datasets): not graded again", flush=True)
            else:
                out = count_attempt(grade(s, {}, placeholder=False), {s: out["grader_attempts"]})
                out[RETRIED_KEY] = True
                if mark_systemic(out, systemic_places(seed_records(run_dir, VERIFY_FILE, {s: out}),
                                                      completed_seeds(run_dir))):
                    print(f"{s}: {SYSTEMIC_FAULT}, shown by the run's records once its retry ended", flush=True)
            write(s, out)
    finally:
        if cpus is not None:
            os.sched_setaffinity(0, cpus)


def post_datasets(run_dir: Path, suite_dir: Path, task, seeds) -> None:
    """The post stage on several datasets of a run as one stage: every one graded once, then the retries
    (post_dataset)."""
    post_dataset(run_dir, suite_dir, task, None, seeds=list(seeds))


def settle_systemic(run_dir: Path, task=None) -> list:
    """The score phase's step before the score (sbh.cli): every record of the run whose fault is systemic over the
    run's final records marked (verify.settle_systemic), and written; no dataset graded.  Returns the seeds marked."""
    from ...score import _default, write_text_atomic
    from ..binned_likelihood.verify import completed_seeds, seed_records, settle_systemic as settle
    run_dir = Path(run_dir)
    rows = seed_records(run_dir, VERIFY_FILE)
    marked = settle(rows, completed_seeds(run_dir))
    for seed in marked:
        write_text_atomic(run_dir / "runs" / seed / VERIFY_FILE, json.dumps(rows[seed], indent=1, default=_default))
    return marked
