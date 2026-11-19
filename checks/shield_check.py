"""The sealed-truth shield under concurrency (sbh/runner.py): runs that shield and unshield a suite while other
processes read its manifest and truths (sbh.suite.load_manifest, sbh.score.read_json_maybe_shielded) never make a
read fail, and the last run out leaves every mode as it found it.  A file under no shield root stays refused.

    python checks/shield_check.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import stat
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SECONDS = 3.0
MANIFEST = {"task": "gauss_peak", "seeds": [{"seed": "seed_001"}]}
TRUTH = {"mu": 1.25}


def shielder(suite: str, go, errors) -> None:
    from sbh.runner import _shield, _unshield
    s = Path(suite)
    rng = random.Random(os.getpid())
    go.wait()
    until = time.time() + SECONDS
    while time.time() < until:
        try:
            token = _shield([s / "truth", s / "manifest.json"])
            time.sleep(rng.uniform(0, 0.01))
            _unshield(token)
        except Exception as e:  # noqa: BLE001
            errors.put(f"shielder {os.getpid()}: {type(e).__name__}: {e}")
            return


def reader(suite: str, go, errors, reads) -> None:
    from sbh.score import read_json_maybe_shielded
    from sbh.suite import load_manifest
    s = Path(suite)
    n = 0
    go.wait()
    until = time.time() + SECONDS
    while time.time() < until:
        try:
            m, t = load_manifest(s), read_json_maybe_shielded(s / "truth" / "seed_001.json")
        except Exception as e:  # noqa: BLE001
            errors.put(f"reader {os.getpid()}: {type(e).__name__}: {e}")
            return
        if m["seeds"] != MANIFEST["seeds"] or t != TRUTH:
            errors.put(f"reader {os.getpid()}: read {m} / {t}")
            return
        n += 1
    reads.put(n)


def main() -> int:
    import sbh.score  # noqa: F401
    import sbh.suite  # noqa: F401
    from sbh.runner import read_shielded
    fails = []
    ctx = mp.get_context("fork")
    with tempfile.TemporaryDirectory() as tmp:
        suite = Path(tmp) / "suite"
        (suite / "truth").mkdir(parents=True)
        (suite / "manifest.json").write_text(json.dumps(MANIFEST))
        (suite / "truth" / "seed_001.json").write_text(json.dumps(TRUTH))
        os.chmod(suite / "manifest.json", 0o644)
        os.chmod(suite / "truth", 0o755)
        errors, reads, go = ctx.Queue(), ctx.Queue(), ctx.Event()
        procs = [ctx.Process(target=shielder, args=(str(suite), go, errors)) for _ in range(2)]
        procs += [ctx.Process(target=reader, args=(str(suite), go, errors, reads)) for _ in range(2)]
        for p in procs:
            p.start()
        go.set()
        for p in procs:
            p.join(SECONDS + 60)
        errs = []
        while not errors.empty():
            errs.append(errors.get())
        n = []
        while not reads.empty():
            n.append(reads.get())
        modes = (stat.S_IMODE((suite / "manifest.json").stat().st_mode), stat.S_IMODE((suite / "truth").stat().st_mode))
        holders = json.loads((suite / ".shield.json").read_text()).get("holders")
        ok = not errs and len(n) == 2 and min(n) > 0 and modes == (0o644, 0o755) and not holders
        print(f"  {'PASS' if ok else 'FAIL'}  2 shielding runs and 2 readers for {SECONDS:.0f} s: reads {n}, errors "
              f"{errs[:3]}, modes after {oct(modes[0])}/{oct(modes[1])}, holders left {holders}")
        if not ok:
            fails.append("concurrent shield and reads")
        other = Path(tmp) / "other"
        other.mkdir()
        (other / "x.json").write_text("{}")
        os.chmod(other / "x.json", 0)
        try:
            read_shielded(other / "x.json")
            refused = False
        except PermissionError:
            refused = True
        ok = refused and stat.S_IMODE((other / "x.json").stat().st_mode) == 0
        print(f"  {'PASS' if ok else 'FAIL'}  an unreadable file under no shield root stays refused")
        if not ok:
            fails.append("unshielded unreadable file")
        os.chmod(other / "x.json", 0o600)
    print("shield check:", "FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
