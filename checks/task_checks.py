"""Every task's own consistency checks (Task.quick_checks), well-formed further references, and the Python of its own
submissions (reference, further references, attackers) parsing and binding every name it uses; smoke, seconds.

    pixi run python checks/task_checks.py [--task NAME]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sbh.task import load_task  # noqa: E402


def submissions_python(task):
    """(check, ok, detail): the task's own submissions -- its reference, its further references and its attackers --
    parse and bind every name they use (sbh.harbor_pack.check_python_tree), so a broken one fails here, not in a run."""
    from sbh.harbor_pack import check_python_tree
    dirs = [task.reference_submission, *sorted(task.path("alt_references").glob("*")),
            *sorted(task.path("attackers").glob("*"))]
    bad = []
    for d in (d for d in dirs if d.is_dir()):
        try:
            check_python_tree(d)
        except SystemExit as e:
            bad.append(str(e))
    return ("its submissions' Python parses and binds its names", not bad, "; ".join(bad))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="", help="one task (default: every task)")
    a = ap.parse_args(argv)
    names = [a.task] if a.task else sorted(p.parent.name for p in (ROOT / "tasks").glob("*/__init__.py"))
    fails = []
    for name in names:
        task = load_task(name)
        results = task.quick_checks()
        try:
            task.further_reference_names()
            results = results + [("further references well formed", True, "")]
        except ValueError as e:
            results = results + [("further references well formed", False, str(e))]
        results = results + [submissions_python(task)]
        for check, ok, detail in results:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: {check}" + (f"  ({detail})" if detail else ""))
            if not ok:
                fails.append(f"{name}: {check}")
        print(f"{name}: {len(results)} check(s)")
    print(f"task checks: {'PASS' if not fails else 'FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
