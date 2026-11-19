"""Run the corrections kind's post stage (post.py) by hand on one run directory.

Per completed dataset: the measurement checks and the closure comparison (the Z peak or the jet balance), written to
runs/<seed>/measurement_verify.json; a dataset the stage itself fails on gets its fault row (post.fault_row), checked
once more where that fault has not recurred (post.check).
The evaluation runs the same stage through `sbh <cmd> --phase post`.

--peak-budget-x [X] (rerun_peak): only the closure comparisons a current file records as not made within the post
stage's budget, made again at X times that budget (the larger evaluator budget no contract states), X at least
scoring.PEAK_RERUN_X, the factor at which a miss again counts as reproduced, and its default; a smaller X is refused,
since a miss there would reproduce nothing.  The measurement checks are kept as made.
The file's peak_attempts then say whether the comparison ran out again: the scorer leaves a comparison made at the
larger budget scored as any other, and escalates one that ran out again at PEAK_RERUN_X or more times the budget it
missed (the run unranked, scoring.budget_reproduced).

    python -m sbh.kinds.corrections.measurement_verify --run-dir work/<suite>/<label> --suite <suite_dir> \\
        [--task T] [--seeds 1,3] [--force | --peak-budget-x [X]]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sbh.kinds.corrections.post import check  # noqa: E402
from sbh.suite import suite_task  # noqa: E402


def rerun_peak(run_dir: Path, suite_dir: Path, task, x: float, seeds=None) -> int:
    """The closure comparisons not made within the post stage's budget (post.PEAK_NOT_IN_BUDGET on a flavour of a file
    refit.current accepts), made again within ``x`` times that budget (refused below scoring.PEAK_RERUN_X: a miss there
    would reproduce nothing, scoring.budget_reproduced), every flavour of the dataset as the post
    stage makes them (the candidates cached on disk by then).  The measurement checks stay as made: their budget is the
    contract's.  In the file: "peak" the new comparison, "peak_attempts" this attempt appended after the earlier ones
    (post.peak_attempts: its budget in seconds and whether it ran out again), "peak_rerun" {"budget_x", "peak_s",
    "seconds", "exceeded"}; the rule and the budget factors are kept, so refit.current still holds and the scorer reads
    it.  A dataset whose comparison already ran out at a budget at least this large is left as it is (said so); an
    exception of the evaluator's own code leaves its file as it was.  Returns the number of datasets made again."""
    import json
    import time
    import traceback
    from sbh.budget import BudgetExceeded, budget
    from sbh.kinds.corrections import post as P
    from sbh.kinds.corrections import refit as M
    from sbh.score import _default, load_records, read_json_maybe_shielded, record_ok
    from sbh.kinds.corrections.scoring import PEAK_RERUN_X
    if not float(x) >= PEAK_RERUN_X:
        raise SystemExit(f"--peak-budget-x {x:g}: a rerun is at PEAK_RERUN_X = {PEAK_RERUN_X:g} or more times the post "
                         "stage's budget (scoring.budget_reproduced: a comparison that ran out again below it would "
                         "reproduce nothing and stay pending)")
    run_dir, suite_dir = Path(run_dir), Path(suite_dir)
    entries = {s["seed"]: s for s in read_json_maybe_shielded(suite_dir / "manifest.json")["seeds"]}
    budgets, budget_rec = P.budget_of(run_dir, task)
    if budgets["peak"] is None:
        raise SystemExit(f"no closure-comparison budget applies to {run_dir.name} ({budget_rec.get('why') or budget_rec}):"
                         " nothing ran out of one")
    peak_s = float(budgets["peak"]) * float(x)
    recs = (load_records(run_dir) if seeds is None else
            [json.loads(f.read_text()) for f in (run_dir / "runs" / s / "record.json" for s in sorted(seeds))
             if f.is_file()])
    done = 0
    for rec in recs:
        seed = rec["seed"]
        target = run_dir / "runs" / seed / M.VERIFY_FILE
        try:
            row = json.loads(target.read_text())
        except (OSError, ValueError):
            row = None
        if not record_ok(rec) or not M.current(row, task, run_dir):
            if record_ok(rec):
                print(f"{seed}: no {M.VERIFY_FILE} made under the current rules: run the post stage first")
            continue
        if row.get("status") != "ok":
            print(f"{seed}: its {M.VERIFY_FILE} records a fault of the post stage (no comparison was made): run the "
                  "post stage again first")
            continue
        if not P._ran_out(row.get("peak")):
            continue
        before = max((float(a["peak_s"]) for a in row.get("peak_attempts") or []
                      if isinstance(a, dict) and a.get("not_in_budget") and isinstance(a.get("peak_s"), (int, float))),
                     default=float((row.get("budget") or {}).get("peak_s") or 0.0))
        if not peak_s > before:
            print(f"{seed}: its comparison already ran out within {before:.0f} s: rerun above that, not {peak_s:.0f} s")
            continue
        s = entries[seed]
        ds = suite_dir / ("public" if s["public"] else "sealed") / seed
        out_dir, cache = run_dir / "runs" / seed / "out", suite_dir / "cache" / seed
        t0 = time.time()
        peak, exceeded = None, None
        try:
            answered = P.peak_answers(task, out_dir, ds)
            t_peak = time.time()
            try:
                with budget(peak_s, f"the comparison of {seed} at {float(x):g} x its budget"):
                    peak = P.peak_closure(task, out_dir, ds, cache, s["index"], row["corrections"], answered)
            except BudgetExceeded as e:
                if peak is None:
                    peak = P.peak_closure(task, out_dir, ds, cache, s["index"], row["corrections"], answered,
                                          over=f"{P.PEAK_NOT_IN_BUDGET} ({e})")
                exceeded = str(e)
        except Exception as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)[-3:]
            print(f"{seed}: verify error: {type(e).__name__}: {e} (at "
                  + " <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in reversed(tb)) + ")", flush=True)
            continue
        row["peak_attempts"] = P.peak_attempts(row, peak, peak_s, task)
        row["peak"] = peak
        row["peak_rerun"] = {"budget_x": float(x), "peak_s": peak_s, "seconds": time.time() - t_peak,
                             "exceeded": exceeded}
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(row, indent=1, default=_default))
        tmp.replace(target)
        again = P._ran_out(peak)
        print(f"{seed}: comparison made again within {peak_s:.0f} s ({float(x):g} x its budget): "
              + ("ran out again" if again else "made") + f" ({time.time() - t0:.0f} s) -> {target}")
        done += 1
    print(f"measurement_verify: {done} comparison(s) made again in {run_dir}")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, help="a scored or completed run directory (work/<suite>/<label>)")
    ap.add_argument("--suite", required=True, help="the suite directory (its manifest, sealed and public datasets)")
    ap.add_argument("--task", default=None, help="the task the suite is scored for (default: the manifest's)")
    ap.add_argument("--seeds", default="", help="comma-separated seed indices to check (default: every completed one)")
    ap.add_argument("--force", action="store_true", help="re-check a seed whose file is already current")
    ap.add_argument("--peak-budget-x", type=float, nargs="?", const=-1.0, default=None, metavar="X",
                    help="only the closure comparisons not made within the post stage's budget, made again at X "
                         "times it (rerun_peak; X at least scoring.PEAK_RERUN_X, the factor at which a miss again "
                         "counts as reproduced, and its default)")
    a = ap.parse_args(argv)
    want = {f"seed_{int(x):03d}" for x in a.seeds.split(",") if x.strip()} if a.seeds else None
    task = suite_task(Path(a.suite), a.task)
    if a.peak_budget_x is not None:
        if a.force:
            ap.error("--peak-budget-x makes only the comparisons again: not with --force")
        from sbh.kinds.corrections.scoring import PEAK_RERUN_X
        rerun_peak(Path(a.run_dir), Path(a.suite), task, PEAK_RERUN_X if a.peak_budget_x == -1.0 else a.peak_budget_x,
                   want)
        return 0
    check(Path(a.run_dir), Path(a.suite), task, want, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
