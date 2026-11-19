"""A run's recorded CPU and peak RSS cover every descendant (runs in `pixi run smoke`; about 20 s).

Python 3.14's forkserver outlives the pipeline's main process, so without the reaper (sbh/reaper.py) os.wait4 misses
the pool workers' and orphans' CPU and memory.  In each sandbox mode available (bwrap, unshare, none; setpriv when
root), a probe burning 2 x 0.4 CPU-s in a forkserver pool (64 MB workers) plus 0.3 CPU-s in a double-forked orphan
must record cpu_s >= 0.9 x 1.1 s, maxrss_mb >= 60, a cpu_source naming a finished reaper, and leave nothing behind.

  pidns          a forkserver left in a nested user + PID namespace (its rusage is discarded): bwrap must refuse
                 the namespace or charge the CPU; other modes cannot block it (printed, not failed)
  timeout        a pool burning forever under a 4 s cap: timed_out, SIGKILL exit, the CPU its workers reported
                 using recorded, nothing left
  near deadline  run.sh exits 0 inside the reaper's grace before the deadline, leaving a sleeper: not a timeout
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sbh import runner as R  # noqa: E402

MARK = "sbh_cpu_accounting_check"
PROBE = '''import multiprocessing as mp, os, sys, time
from concurrent.futures import ProcessPoolExecutor

def burn(n):
    b = bytearray(64 << 20)
    for i in range(0, len(b), 4096):
        b[i] = 1
    t = report = time.process_time()
    while n < 0 or time.process_time() - t < n:
        if n < 0 and time.process_time() >= report:     # burning forever: say how much CPU this worker has used
            print(f"burned {os.getpid()} {time.process_time():.3f}", flush=True)
            report += 0.05
    return len(b)

def pool(n):
    with ProcessPoolExecutor(2, mp_context=mp.get_context("forkserver")) as ex:
        list(ex.map(burn, [n, n]))


if __name__ == "__main__":
    n = float(sys.argv[1])
    if sys.argv[2:] == ["pidns"]:
        uid, gid = os.getuid(), os.getgid()
        try:
            os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWPID)
            for fn, txt in (("setgroups", "deny"), ("uid_map", f"0 {uid} 1"), ("gid_map", f"0 {gid} 1")):
                with open(f"/proc/self/{fn}", "w") as fh:
                    fh.write(txt)
            print("nested namespace", flush=True)
        except OSError as e:
            print(f"namespace refused: {e.strerror}", flush=True)
        pid = os.fork()                 # the namespace's init: exits with the pool's forkserver still alive
        if pid == 0:
            pool(n)
            os._exit(0)
        os.waitpid(pid, 0)
        sys.exit(0)
    pool(n)
    pid = os.fork()
    if pid == 0:
        if os.fork() == 0:
            time.sleep(0.2)
            t = time.process_time()
            while time.process_time() - t < 0.3:
                pass
        os._exit(0)
    os.waitpid(pid, 0)
'''
EXPECT_S = 2 * 0.4 + 0.3
PIDNS_S = 2 * 0.4
NEAR_SH = '(setsid sleep 30 &)\nexec python3 -c "import time; time.sleep(max(0.0, {at} - time.time()))"\n'


def tarball(tmp: Path, name: str, arg: str, body: str = "") -> Path:
    tb = tmp / f"{name}.tar.gz"
    run_sh = f'#!/bin/sh\nexport {MARK}={tmp.name}\n' + (body or f'exec python3 "$(dirname "$0")/probe.py" {arg}\n')
    with tarfile.open(tb, "w:gz") as tf:
        for fn, data, mode in (("run.sh", run_sh.encode(), 0o755), ("probe.py", PROBE.encode(), 0o644)):
            ti = tarfile.TarInfo(fn)
            ti.size, ti.mode = len(data), mode
            tf.addfile(ti, io.BytesIO(data))
    return tb


def leftovers() -> list:
    """Processes of this check still alive: their environment carries MARK=<our tmp dir name>."""
    out = []
    for e in os.scandir("/proc"):
        if not e.name.isdigit() or int(e.name) == os.getpid():
            continue
        try:
            with open(f"/proc/{e.name}/environ", "rb") as fh:
                env = fh.read()
        except OSError:
            continue
        if TOKEN in env.split(b"\0"):
            out.append(int(e.name))
    return out


modes = ["bwrap"] if R.bwrap_available() else []
modes += ["setpriv"] if (os.geteuid() == 0 and shutil.which("setpriv")) else []
modes += ["unshare"] if R.unshare_available() else []
modes += ["none"]
fails, lines = [], []
saved = os.environ.get("SBH_SANDBOX")
with tempfile.TemporaryDirectory(dir=ROOT / "work" if os.access(ROOT / "work", os.W_OK) else None) as tmp:
    tmp = Path(tmp)
    tmp.chmod(0o755)
    TOKEN = f"{MARK}={tmp.name}".encode()
    ds = tmp / "dataset"
    ds.mkdir()
    ok_tb, forever_tb = tarball(tmp, "ok", "0.4"), tarball(tmp, "forever", "-1")
    pidns_tb = tarball(tmp, "pidns", "0.4 pidns")
    try:
        for i, mode in enumerate(modes):
            os.environ["SBH_SANDBOX"] = mode
            rec = R.run_one(ok_tb, ds, tmp / f"w{i}", R.Limits(timeout_s=60, memory_mb=0), task=None)
            time.sleep(0.2)
            left = leftovers()
            lines.append(f"{mode}: cpu {rec.cpu_s} s (burned {EXPECT_S:.1f}; cgroup cross-check {getattr(rec, 'cpu_cgroup_s', None)}), "
                         f"maxrss {rec.maxrss_mb} MB, wall {rec.wall_s} s")
            if rec.exit_code != 0 or rec.timed_out:
                fails.append(f"{mode}: probe exited {rec.exit_code} (timed out {rec.timed_out}): {rec.stderr_tail[-200:]}")
                continue
            if not (rec.cpu_s or 0) >= 0.9 * EXPECT_S:
                fails.append(f"{mode}: cpu_s {rec.cpu_s} < 0.9 x {EXPECT_S} CPU-s burned by a forkserver pool + "
                             "an orphan: descendants are missing from the run's CPU")
            if not (rec.maxrss_mb or 0) >= 60:
                fails.append(f"{mode}: maxrss_mb {rec.maxrss_mb} < 60: the 64 MB pool workers are missing")
            src = getattr(rec, "cpu_source", "")
            if "reaper as" not in src or "did not finish" in src:
                fails.append(f"{mode}: cpu_source {src!r}: not the reaper, or it did not finish reaping")
            if left:
                fails.append(f"{mode}: {len(left)} probe process(es) outlived the run")
        for i, mode in enumerate(modes):
            os.environ["SBH_SANDBOX"] = mode
            rec = R.run_one(pidns_tb, ds, tmp / f"p{i}", R.Limits(timeout_s=60, memory_mb=0), task=None)
            ns = (tmp / f"p{i}" / "stdout.txt").read_text().strip().splitlines()[:1]
            charged = rec.exit_code == 0 and (rec.cpu_s or 0) >= 0.9 * PIDNS_S
            lines.append(f"{mode} pidns: {ns[0] if ns else '?'}, exit {rec.exit_code}, cpu {rec.cpu_s} s "
                         f"(burned {PIDNS_S:.1f}; cgroup {rec.cpu_cgroup_s})")
            if mode == "bwrap" and not charged:
                fails.append(f"bwrap pidns: exit {rec.exit_code}, cpu_s {rec.cpu_s} < 0.9 x {PIDNS_S}: a nested "
                             "PID namespace hides the pool's CPU (bwrap must refuse nested user namespaces)")
            elif not charged:
                lines[-1] += " (known: not blockable in this mode)"
        mode = modes[0]
        os.environ["SBH_SANDBOX"] = mode
        for _ in range(3):
            rec = R.run_one(forever_tb, ds, tmp / "wt", R.Limits(timeout_s=4, memory_mb=0), task=None)
            time.sleep(0.2)
            used = {}
            for ln in (tmp / "wt" / "stdout.txt").read_text(errors="replace").splitlines():
                f = ln.split()
                if len(f) == 3 and f[0] == "burned":
                    used[f[1]] = float(f[2])
            burned = sum(used.values())
            if burned >= 0.3:
                break
        left = leftovers()
        lines.append(f"{mode} timeout: exit {rec.exit_code}, cpu {rec.cpu_s} s in {rec.wall_s} s wall (the workers "
                     f"reported {burned:.2f})")
        if not (rec.timed_out and rec.failure == "timeout" and rec.exit_code in R.SIGKILL_EXITS):
            fails.append(f"timeout: timed_out {rec.timed_out}, failure {rec.failure}, exit {rec.exit_code}")
        if burned < 0.3:
            fails.append(f"timeout: the pool's workers reported {burned:.2f} CPU-s in 4 s in three tries, too little to "
                         "tell")
        elif not (rec.cpu_s or 0) >= 0.9 * burned:
            fails.append(f"timeout: cpu_s {rec.cpu_s} < 0.9 x {burned:.2f} CPU-s the pool's workers reported using "
                         "before the kill")
        if left:
            fails.append(f"timeout: {len(left)} probe process(es) outlived the killed run")
        for mode in dict.fromkeys((modes[0], modes[-1])):
            os.environ["SBH_SANDBOX"] = mode
            for _ in range(3):
                near_tb = tarball(tmp, "near", "", NEAR_SH.format(at=time.time() + 2))
                rec = R.run_one(near_tb, ds, tmp / "wn", R.Limits(timeout_s=3, memory_mb=0), task=None)
                time.sleep(0.2)
                if rec.wall_s > 3 or rec.exit_code != 0 or rec.timed_out:
                    break
            left = leftovers()
            lines.append(f"{mode} near deadline: exit {rec.exit_code}, timed out {rec.timed_out}, wall {rec.wall_s} s")
            if rec.exit_code != 0 or rec.timed_out or rec.failure == "timeout":
                fails.append(f"{mode} near deadline: exit {rec.exit_code}, timed_out {rec.timed_out}, failure "
                             f"{rec.failure}: run.sh exited 0 before the deadline; its leftover's grace is not its time")
            elif rec.wall_s <= 3:
                fails.append(f"{mode} near deadline: wall {rec.wall_s} s: the leftover was gone by the deadline in "
                             "three tries, nothing tested")
            if left:
                fails.append(f"{mode} near deadline: {len(left)} process(es) outlived the run")
    finally:
        for pid in leftovers():
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        if saved is None:
            os.environ.pop("SBH_SANDBOX", None)
        else:
            os.environ["SBH_SANDBOX"] = saved

print("cpu accounting:", "; ".join(lines))
for f in fails:
    print("  FAIL ", f)
print("cpu accounting check:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
