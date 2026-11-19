"""Re-score stored runs of a suite with the current scorer and print S_v3 with its components.

No submission is re-executed: stored run records (and model_verify_toys.json for GOF) are re-scored and summary.json
rewritten.  The task is sbh.suite.suite_task's (--task, else the suite manifest's).  The closure's run is scored as
`sbh closure` scores it, against its own baseline; `sbh post-score --label reference` then records its score in
closure.json, which every other run's relative score reads (a scoring change that moves the baseline needs
`sbh closure --phase score`, which rewrites closure.json).  A task scored on the mass-marginal mu is rescored only
against a closure of this evaluator's mu rule, and its other runs only once the grader's baseline is in it
(sbh.suite.closure_gate); a refused run is skipped and fails the command.

    python -m sbh.rescore --suite DIR [--task T] [--work work] [--glob 'agent_*']
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sbh.cli import _sigma_ref  # noqa: E402
from sbh.score import score_run  # noqa: E402
from sbh.suite import CLOSURE_LABEL, closure_gate, suite_task  # noqa: E402


SUITE = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=SUITE, required=SUITE is None)
    ap.add_argument("--task", default=None, help="the task the suite is scored for (default: the manifest's)")
    ap.add_argument("--work", default="work")
    ap.add_argument("--glob", default="*", help="run directories under work/<suite>/ to rescore")
    a = ap.parse_args()
    suite = Path(a.suite)
    task = suite_task(suite, a.task)
    sigma_ref = _sigma_ref(suite)
    rows, skipped = [], []
    for run_dir in sorted((Path(a.work) / suite.name).glob(a.glob)):
        if not (run_dir / "runs").exists():
            continue
        closure = run_dir.name == CLOSURE_LABEL
        why = closure_gate(suite, task, run_dir.name)
        if why:
            print(f"{run_dir.name}: skipped ({why})")
            skipped.append(run_dir.name)
            continue
        try:
            pub = score_run(run_dir, suite, task, run_dir.name, sigma_ref=None if closure else sigma_ref,
                            making_baseline=closure)
        except SystemExit as e:
            print(f"{run_dir.name}: skipped ({e})")
            skipped.append(run_dir.name)
            continue
        v2, v3 = pub.get("v2") or {}, pub.get("v3") or {}
        if not v3:
            continue
        cal, gof = v3.get("calibration") or {}, v3.get("gof")
        pe = cal.get("per_effect", {})
        tr, cl = v3.get("truthfulness"), v3.get("claims")
        prov = (v3.get("provisional") or {}).get("n_cases")
        rows.append((run_dir.name, v3["s_v3"], v2.get("delta_mean"), cal.get("delta"),
                     " ".join(f"{k[:5]}={m['delta']:+.2f}({m['n_reported']})" for k, m in pe.items()),
                     "pending" if gof is None else f"{gof['delta']:+.2f} ({gof['n_charged']} charged)",
                     "" if v3.get("score_source") != "model" else
                     f"truth {tr['delta'] if tr else 0:.2f} claims {cl['delta'] if cl else 0:.2f}"
                     + (f" provisional {prov}" if prov else "")))
    rows.sort(key=lambda r: -(r[1] or 0))
    print(f"\n{'run':30s} {'S':>6s} {'Δphys':>6s} {'Δcal':>6s}  calibration by effect (n reported)          GOF")
    for r in rows:
        print(f"{r[0]:30s} {r[1]:6.3f} {r[2] or 0:+6.2f} {r[3] or 0:+6.2f}  {r[4]:44s} {r[5]}  {r[6]}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
