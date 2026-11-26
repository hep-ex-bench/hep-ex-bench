"""The host's isolation (`pixi run check`): bwrap must be the runner's mode, and the dev-session wrapper must start.

A host that fails this cannot run `scripts/fleet/run_*_h4l.sh`, and `sbh` would silently fall back to `unshare`
(weaker: the filesystem stays visible) or `none` (NOT isolated).  On Ubuntu 23.10+/24.04, AppArmor restricts
unprivileged user namespaces; without a profile for the bwrap binary the kernel refuses the uid map and bwrap exits
with `setting up uid map: Permission denied`.  `docs/FAQ.md` has the scoped profile and how to undo it.

    python checks/sandbox_check.py
    SBH_SKIP_SANDBOX_CHECK=1 python checks/sandbox_check.py   # static checks in a container without bwrap
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    if os.environ.get("SBH_SKIP_SANDBOX_CHECK") == "1":
        print("sandbox check: SKIP (SBH_SKIP_SANDBOX_CHECK=1)")
        return 0

    # The fleet launchers put the pixi bin first; test the bwrap they actually run.
    fleet_bin = ROOT / ".pixi" / "envs" / "default" / "bin"
    if fleet_bin.is_dir():
        os.environ["PATH"] = f"{fleet_bin}:{os.environ.get('PATH', '')}"

    from sbh import runner as R

    bwrap = shutil.which("bwrap")
    bwrap_ok = R.bwrap_available()
    unshare_ok = R.unshare_available()
    mode = R.sandbox_mode()
    version = ""
    if bwrap:
        try:
            version = subprocess.run([bwrap, "--version"], capture_output=True, text=True, timeout=10
                                     ).stdout.strip().splitlines()[0]
        except Exception:  # noqa: BLE001 - only decoration
            pass
    print(f"  bwrap: {bwrap or '(not found)'}" + (f" ({version})" if version else ""))
    print(f"  unshare: {shutil.which('unshare') or '(not found)'}")
    print(f"  runner: bwrap={bwrap_ok} unshare={unshare_ok} mode={mode}")
    if os.environ.get("SBH_SANDBOX"):
        print(f"  SBH_SANDBOX={os.environ['SBH_SANDBOX']} (overrides the host probes)")

    # A root verifier container is isolated by setpriv; it has no dev sessions to test.
    setpriv_container = os.geteuid() == 0 and mode == "setpriv"
    dev_err = ""
    fails = []
    if setpriv_container:
        print("  setpriv: evaluation runs are isolated; dev sessions need bwrap and are skipped as root")
    else:
        with tempfile.TemporaryDirectory(prefix="sbh-sandbox-") as d:
            try:
                p = subprocess.run([str(ROOT / "scripts/fleet/dev_sandbox.sh"), d, "--", "/bin/true"],
                                   capture_output=True, text=True, timeout=120, cwd=ROOT)
                dev_err = (p.stderr or p.stdout).strip()
                if p.returncode == 0:
                    print("  dev sandbox: PASS")
                else:
                    fails.append(f"dev_sandbox.sh exited {p.returncode}")
                    for line in dev_err.splitlines()[-6:]:
                        print(f"    | {line}")
            except Exception as e:  # noqa: BLE001 - a missing or hung wrapper is a failure
                fails.append(f"dev_sandbox.sh failed: {type(e).__name__}: {e}")
    if not bwrap_ok and not setpriv_container:
        fails.append("bwrap is not usable")

    if fails:
        if not bwrap_ok:
            print("  bwrap is not usable for the runner's sandbox.")
            if "setting up uid map" in dev_err or "RTM_NEWADDR" in dev_err:
                print("  The kernel/AppArmor refused it (Ubuntu 23.10+/24.04: see docs/FAQ.md).")
            if mode == "none":
                print("  sbh would use 'none' (NOT isolated): do not score on this host.")
            elif mode == "unshare":
                print("  sbh would use 'unshare' (weaker: filesystem not isolated).")
            print("  Dev sessions (scripts/fleet/run_*_h4l.sh) cannot start.")
        for f in fails:
            print(f"  FAIL {f}")
        print("sandbox check: FAIL")
        return 1
    if bwrap_ok and mode != "bwrap":
        print(f"  note: SBH_SANDBOX={os.environ.get('SBH_SANDBOX', mode)} selects {mode}; unset it for bwrap isolation")
    print("sandbox check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
