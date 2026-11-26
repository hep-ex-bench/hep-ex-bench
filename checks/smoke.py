"""The gates in tiers (`pixi run check` = quick, `pixi run smoke`): quick is the consistency checks (seconds, after
any edit); smoke adds the batteries and the toy suites' build, closure and stress (minutes, before every push).  The
first step (the base env) runs alone; then each toy's chain runs in order in a lane of its own while the other steps
share the remaining workers.  Each step's output goes to work/smoke_logs/<step>.log (the tail is printed if it fails);
a table of verdicts and wall times ends the run.  A failure starts no further step unless --keep-going.

    python checks/smoke.py                # the smoke tier, -j 4 by default
    python checks/smoke.py --tier quick   # the consistency checks only
    python checks/smoke.py -j 1           # one at a time, output streamed
    python checks/smoke.py --keep-going   # run the rest after a failure (a toy's chain stops at its first failure)
    python checks/smoke.py --only NAME[,NAME]  /  --from NAME
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOY = "work/smoke/toy_dial"
RUN = "work/smoke/run"
STRESSED = "reference,attacker_constant,attacker_crasher,attacker_inflator,attacker_memorizer,attacker_nocalib,attacker_symlink"
CALIB, RUN_CALIB = "work/smoke/peak_toy", "work/smoke/run_calib"
BUMP, RUN_BUMP = "work/smoke/bump_v0", "work/smoke/run_bump"
BUMP_STRESSED = ("reference_bump,attacker_bump_constant,attacker_bump_crasher,attacker_bump_inflator,"
                 "attacker_bump_memorizer,attacker_bump_nocalib,attacker_bump_symlink")
WTOY, RUN_WTOY = "work/smoke/width_toy_dial", "work/smoke/run_width"
WSTRESSED = ("reference,attacker_width_constant,attacker_width_crasher,attacker_width_inflator,"
             "attacker_width_deflator,attacker_width_misreport,attacker_width_fabricate,attacker_width_token,"
             "attacker_width_misfit,attacker_width_memorizer,attacker_width_nocalib,attacker_width_symlink")

STEPS = [
    ("agentbase", "pixi install -e agentbase", False, "quick"),
    ("sandbox", "python checks/sandbox_check.py", False, "quick"),
    ("contracts", "python -m sbh.contract render --check", False, "quick"),
    ("locks", "python checks/locks_check.py", False, "quick"),
    ("balance", "python checks/balance_check.py", False, "quick"),
    ("task_checks", "python checks/task_checks.py", False, "quick"),
    ("scan_likelihood", "python checks/scan_likelihood_check.py", False, "quick"),
    ("attackers", "python checks/attackers_check.py", False, "quick"),
    ("bundles", "python checks/bundle_check.py", False, "quick"),
    ("mc_bundle", "python checks/mc_bundle_check.py", False, "quick"),
    ("vendored", "python checks/vendored_check.py", False, "quick"),
    ("shield", "python checks/shield_check.py", False, "quick"),
    ("cpu_accounting", "python checks/cpu_accounting_check.py", False, "smoke"),
    ("jax_engine", "python checks/jax_engine_check.py", False, "smoke"),
    ("scan_likelihood_grader", "python checks/scan_likelihood_check.py --grader", False, "smoke"),
    ("seq_toys", "python checks/seq_toys_check.py", False, "smoke"),
    ("agent_run_record", "python scripts/fleet/agent_run_record.py --selftest", False, "quick"),
    ("toy_build", "python -m sbh build-suite --task gauss_peak --name toy_dial --design tasks/gauss_peak/truth_grids/grid_v0.json "
                  "--master-seed 2 --out work/smoke --force", "gauss_peak", "smoke"),
    ("toy_closure", f"python -m sbh closure --suite {TOY} --jobs 4 --work {RUN}", "gauss_peak", "smoke"),
    ("toy_stress", f"python -m sbh stress --suite {TOY} --jobs 4 --work {RUN} --only {STRESSED}", "gauss_peak",
     "smoke"),
    ("calib_build", "python -m sbh build-suite --task peak_calib --name peak_toy --master-seed 3 --out work/smoke "
                    "--design tasks/peak_calib/truth_grids/grid_v0.json --force", "peak_calib", "smoke"),
    ("calib_closure", f"python -m sbh closure --suite {CALIB} --jobs 4 --work {RUN_CALIB}", "peak_calib", "smoke"),
    ("calib_stress", f"python -m sbh stress --suite {CALIB} --jobs 4 --work {RUN_CALIB}", "peak_calib", "smoke"),
    ("bump_build", "python -m sbh build-suite --task bump_model --name bump_v0 --master-seed 1 --out work/smoke "
                   "--design tasks/bump_model/truth_grids/grid_v0.json --force", "bump_model", "smoke"),
    ("bump_closure", f"python -m sbh closure --suite {BUMP} --jobs 4 --work {RUN_BUMP}", "bump_model", "smoke"),
    ("bump_closure_post", f"env SBH_VERIFY_TOYS=200 python -m sbh post-score --suite {BUMP} --work {RUN_BUMP} "
                          "--label reference", "bump_model", "smoke"),
    ("bump_stress", f"python -m sbh stress --suite {BUMP} --jobs 4 --work {RUN_BUMP} --only {BUMP_STRESSED}",
     "bump_model", "smoke"),
    ("bump_post_reference", f"env SBH_VERIFY_TOYS=200 python -m sbh post-score --suite {BUMP} --work {RUN_BUMP} "
                            "--label reference_bump", "bump_model", "smoke"),
    ("bump_post_inflator", f"env SBH_VERIFY_TOYS=200 python -m sbh post-score --suite {BUMP} --work {RUN_BUMP} "
                           "--label attacker_bump_inflator", "bump_model", "smoke"),
    ("bump_stress_post", f"python -m sbh stress --suite {BUMP} --work {RUN_BUMP} --recheck "
                         "--only reference_bump,attacker_bump_inflator", "bump_model", "smoke"),
    ("width_toy_build", "python -m sbh build-suite --task width_toy --name width_toy_dial "
                        "--design tasks/width_toy/truth_grids/width_toy_v0.json --master-seed 2 --out work/smoke --force",
     "width_toy", "smoke"),
    ("width_toy_closure", f"env SBH_VERIFY_TOYS=50 python -m sbh closure --suite {WTOY} --jobs 4 --work {RUN_WTOY}",
     "width_toy", "smoke"),
    ("width_toy_stress", f"env SBH_VERIFY_TOYS=50 python -m sbh stress --suite {WTOY} --jobs 4 --work {RUN_WTOY} "
                         f"--only {WSTRESSED}", "width_toy", "smoke"),
]
TIERS = ("quick", "smoke")
FULL = []


LOGS = ROOT / "work" / "smoke_logs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", "--jobs", type=int, default=4, help="steps at a time (1 = serial, output streamed)")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--from", dest="start", default="")
    ap.add_argument("--tier", choices=TIERS, default="smoke",
                    help="quick: consistency checks, seconds; smoke: + the batteries and the toys' chains, minutes "
                         "(before a push)" + "".join(f"; {t}" for t in TIERS[2:]))
    a = ap.parse_args()
    tier = "full" if getattr(a, "full", False) else a.tier
    steps = [s[:3] for s in STEPS + FULL if TIERS.index(s[3]) <= TIERS.index(tier)]
    names = [s[0] for s in steps]
    for n in [x for x in a.only.split(",") if x] + ([a.start] if a.start else []):
        if n not in names:
            ap.error(f"no step {n!r}; steps: {', '.join(names)}")
    if a.only:
        keep = set(a.only.split(","))
        steps = [s for s in steps if s[0] in keep]
    elif a.start:
        steps = steps[names.index(a.start):]
    order = {s[0]: i for i, s in enumerate(steps)}
    LOGS.mkdir(parents=True, exist_ok=True)
    rows, state = {}, {"failed": False, "broken": set()}
    t_all = time.monotonic()

    def start(name, cmd):
        if a.jobs == 1:
            print(f"\n===== smoke: {name}: {cmd}", flush=True)
            return subprocess.Popen(shlex.split(cmd), cwd=ROOT), None
        log = open(LOGS / f"{name}.log", "w")
        return subprocess.Popen(shlex.split(cmd), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log

    def finish(name, lane, proc, log, t0):
        rc, dt = proc.returncode, time.monotonic() - t0
        if log:
            log.close()
        rows[name] = ("ok" if rc == 0 else f"FAIL ({rc})", dt)
        if a.jobs > 1:
            print(f"  {name:<20} {rows[name][0]:<10} {dt:7.1f} s", flush=True)
            if rc:
                tail = (LOGS / f"{name}.log").read_text(errors="replace").splitlines()[-25:]
                print("\n".join(f"    | {ln}" for ln in tail), flush=True)
        if rc:
            state["failed"] = True
            if lane:
                state["broken"].add(lane)

    def blocked(lane):
        return (state["failed"] and not a.keep_going) or lane in state["broken"]

    first, rest = steps[:1], steps[1:]
    lanes = {}
    for s in rest:
        if s[2]:
            lanes.setdefault(s[2], []).append(s)
    free_q = [s for s in rest if not s[2]]
    for name, cmd, lane in first:
        t0 = time.monotonic()
        proc, log = start(name, cmd)
        proc.wait()
        finish(name, lane, proc, log, t0)
    running = {}
    while any(lanes.values()) or free_q or running:
        for q in (*lanes.values(), free_q):
            while q and blocked(q[0][2]):
                n, _, _ = q.pop(0)
                rows[n] = ("skip", 0.0)
        busy = {v[3] for v in running.values()}
        ready = [q for ln, q in lanes.items() if q and ln not in busy]
        if ready and len(running) < a.jobs:
            name, cmd, lane = ready[0].pop(0)
            running[name] = (*start(name, cmd), time.monotonic(), lane)
            continue
        if free_q and len(running) < max(1, a.jobs - len(ready)):
            name, cmd, lane = free_q.pop(0)
            running[name] = (*start(name, cmd), time.monotonic(), lane)
            continue
        for name, (proc, log, t0, lane) in list(running.items()):
            if proc.poll() is not None:
                del running[name]
                finish(name, lane, proc, log, t0)
        time.sleep(0.2)
    print("\n===== smoke summary")
    for name in sorted(rows, key=order.get):
        verdict, dt = rows[name]
        print(f"  {name:<20} {verdict:<10} {dt:7.1f} s")
    wall = time.monotonic() - t_all
    print(f"  {'total':<20} {'FAIL' if state['failed'] else 'PASS':<10} {wall:7.1f} s wall, "
          f"{sum(r[1] for r in rows.values()):.1f} s summed (-j {a.jobs}; logs in {LOGS.relative_to(ROOT)}/)")
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
