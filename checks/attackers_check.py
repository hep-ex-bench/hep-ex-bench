"""Attackers (runs in `pixi run smoke`): every task attacks or waives each family of its kind, every generated attacker
builds from its reference (sbh/attackgen.py), and submission labels are unique across tasks, submissions/ and the
generated ones.  A participant task stub has no reference and no attackers: it is skipped."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sbh.attackgen import coverage, families, materialize  # noqa: E402
from sbh.task import is_participant_stub, load_task, submission_entries, task_names  # noqa: E402


def main() -> int:
    fails = []
    for name in task_names():
        task = load_task(name)
        if is_participant_stub(task):
            print(f"{name}: a participant stub, no attackers to check")
            continue
        fails += coverage(task)
        for entry in task.attackers:
            d = materialize(task, entry)
            if not (d / "run.sh").is_file():
                fails.append(f"{name}: {entry} has no run.sh")
        own = {f for f, _, _ in task.attackers.values()} | set(task.hand_attackers.values())
        print(f"{name}: families {', '.join(sorted(families(task)))}; attacked {', '.join(sorted(own)) or '-'}; "
              f"waived {', '.join(sorted(task.attack_waivers)) or '-'}")
    try:
        entries = submission_entries()
        print(f"{len(entries)} submission labels, unique")
    except ValueError as e:
        fails.append(str(e))
    for f in fails:
        print("FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
