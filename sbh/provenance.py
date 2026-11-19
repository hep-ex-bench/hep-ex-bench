"""The evaluator's identity: which harness code prepared, ran and scored a run -- EVALUATOR-ONLY.

identity() digests the evaluator's sources (sbh/ and tasks/ code and data, without the tasks' reference pipelines,
submissions, attackers and authoring files, plus pixi.lock), with the git commit when there is one.  `sbh
eval|closure|stress --phase prepare` (and phase all) records it in <work>/evaluator.json; the run and post phases
refuse to execute under another digest, so an evaluation queued across a checkout change never mixes evaluators
(SBH_ALLOW_EVALUATOR_CHANGE=1 overrides, recorded); the score records its own identity beside the prepare's.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Callable, Optional

from . import ROOT

FILE = "evaluator.json"
ALLOW_ENV = "SBH_ALLOW_EVALUATOR_CHANGE"
SUFFIXES = {".py", ".toml", ".json", ".lock"}
SKIP_PARTS = {"__pycache__", ".pixi", "reference", "alt_references", "submissions", "attackers", "tbs"}


def _sources():
    for top in (ROOT / "sbh", ROOT / "tasks"):
        for p in sorted(top.rglob("*")):
            if p.suffix in SUFFIXES and not SKIP_PARTS & set(p.relative_to(ROOT).parts) and p.is_file():
                yield p
    if (ROOT / "pixi.lock").is_file():
        yield ROOT / "pixi.lock"


def _git() -> Optional[dict]:
    """{"commit", "dirty"} of the checkout, or None (no git, not a checkout: a packed verifier)."""
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10)
        st = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no", "--",
                             "sbh", "tasks", "pixi.lock"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if head.returncode or st.returncode:
        return None
    return {"commit": head.stdout.strip(), "dirty": bool(st.stdout.strip())}


def identity() -> dict:
    """{"digest", "n_files"[, "git"]}: the digest covers each source's path and bytes."""
    h, n = hashlib.sha256(), 0
    for p in _sources():
        h.update(p.relative_to(ROOT).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
        n += 1
    out = {"digest": h.hexdigest(), "n_files": n}
    git = _git()
    if git:
        out["git"] = git
    return out


def _read(work: Path) -> dict:
    try:
        return json.loads((Path(work) / FILE).read_text())
    except (OSError, ValueError):
        return {}


def _update(work: Path, change: Callable[[dict], dict]) -> None:
    """Rewrite FILE as change(its record) under an exclusive lock: the yardstick replicas of one run, on several nodes
    at once, each record an override; each writer goes through its own temporary file."""
    p = Path(work) / FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_name(f".{p.name}.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rec = change(_read(work))
        tmp = p.with_name(f".{p.name}.{socket.gethostname()}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, p)


def record(work: Path, stage: str, ident: Optional[dict] = None) -> dict:
    """Record `stage`'s identity (prepare: a new file; score: beside the prepare's)."""
    ident = ident or identity()
    _update(work, lambda rec: {**({} if stage == "prepare" else rec), stage: ident})
    return ident


def check_unchanged(work: Path, phase: str) -> None:
    """Refuse `phase` when the evaluator differs from the one that prepared `work` (ALLOW_ENV=1: warn, recorded)."""
    prepared = _read(work).get("prepare")
    if not prepared:
        print(f"[{Path(work).name}] note: no {FILE} (prepared before evaluator identities were recorded)")
        return
    now = identity()
    if now["digest"] == prepared.get("digest"):
        return
    desc = lambda i: (i.get("git", {}).get("commit", "?")[:10] + (" dirty" if i.get("git", {}).get("dirty") else "")
                      + f", sources {i['digest'][:12]}")  # noqa: E731
    msg = (f"[{Path(work).name}] the evaluator changed since this run was prepared ({desc(prepared)} -> {desc(now)})")
    if os.environ.get(ALLOW_ENV) != "1":
        raise SystemExit(f"{msg}: its datasets would run under different code.  Prepare it again (sbh eval: with "
                         f"--replace), or set {ALLOW_ENV}=1 to proceed (recorded in {FILE})")
    print(f"WARNING: {msg}; proceeding ({ALLOW_ENV}=1)")
    _update(work, lambda rec: {**rec, "overrides": rec.get("overrides", []) + [{"phase": phase, **now}]})
