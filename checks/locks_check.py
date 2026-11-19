"""Every committed pixi.lock is current with its pixi.toml (runs in `pixi run check`).

The runner installs a submission's environment with `pixi install --locked`, which refuses a stale lock; smoke installs
only the toy's reference, so a stale lock of another submission would only show up in its closure.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".pixi", "work", "suites", "dist"}


def tracked_locks():
    """The committed locks (`git ls-files`): never a nested worktree's or an environment's.  Outside a git checkout,
    a walk that does not descend into SKIP directories."""
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", "*pixi.lock"], capture_output=True)
    except OSError:
        r = None
    if r is not None and r.returncode == 0:
        return [ROOT / f for f in r.stdout.decode().split("\0") if f]
    found = []
    for d, subdirs, files in os.walk(ROOT):
        subdirs[:] = [s for s in subdirs if s not in SKIP and not s.startswith(".")]
        found += [Path(d) / "pixi.lock"] if "pixi.lock" in files else []
    return found


def main() -> int:
    pixi = shutil.which("pixi")
    if pixi is None:
        print("FAIL: pixi not on PATH")
        return 1
    locks = sorted(p for p in tracked_locks() if (p.parent / "pixi.toml").exists())
    fails = []
    for lock in locks:
        r = subprocess.run([pixi, "lock", "--check", "--dry-run"], cwd=lock.parent, capture_output=True, text=True)
        if r.returncode:
            fails.append(f"{lock.relative_to(ROOT)}: not current with its pixi.toml (run `pixi lock` there)")
    for f in fails:
        print("FAIL:", f)
    print(f"locks: {len(locks) - len(fails)}/{len(locks)} current")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
