"""MC bundles: the simulation a task ships in every dataset's mc/, and the fingerprint that tells which one a directory
holds.

A task with a bundle states where its current one is (Task.mc_bundle(), from the site configuration) and that
bundle's fingerprint (Task.mc_bundle_fingerprint).  Building a suite, packing a participant bundle and preparing an
evaluation refuse a bundle, or a dataset whose mc/, with another fingerprint: a stale copy of the simulation, or a
suite built on one, cannot stand in for the current one.  After a bundle is rebuilt, its new fingerprint is recorded
in the task and the suites built on the old one are rebuilt.

The fingerprint identifies the bundle's version, not its files: sha256 over the sorted relative paths of every regular
file (symbolic links followed), each with its size, except a .json file, which enters with its parsed content without
the entries that only document (DOC_KEYS), so a reformatting or a reworded note (Task.ship_fixup brings the wording up
to date when packing) is the same version.  The bundle's evaluator-side root entries (Task.mc_bundle_evaluator_files),
which datasets do not ship, are left out.  On cold network storage the walk over a suite's datasets can take minutes.

    python -m sbh bundle-fingerprint <directory> [--task NAME]
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List

DOC_KEYS = ("_comment", "convention", "note")


def _json_content(path: Path) -> bytes:
    """A JSON file's content without its documentation entries, in one canonical form (its bytes if it does not
    parse)."""
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except ValueError:
        return raw

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in DOC_KEYS}
        return [strip(v) for v in o] if isinstance(o, list) else o
    return json.dumps(strip(data), sort_keys=True, separators=(",", ":")).encode()


def fingerprint(root: Path, exclude: Iterable[str] = ()) -> str:
    """The fingerprint of the tree under root, without the root entries named in exclude."""
    root = Path(root)
    skip = set(exclude)
    entries = []
    for d, dirs, files in os.walk(root, followlinks=True):
        if Path(d) == root:
            dirs[:] = [x for x in dirs if x not in skip]
            files = [x for x in files if x not in skip]
        for f in files:
            p = Path(d) / f
            if p.is_file():
                entries.append((p.relative_to(root).as_posix(), p))
    h = hashlib.sha256()
    for rel, p in sorted(entries):
        if rel.endswith(".json"):
            content = _json_content(p)
            h.update(f"{rel}\0{len(content)}\0".encode() + content)
        else:
            h.update(f"{rel}\0{p.stat().st_size}\0".encode())
    return h.hexdigest()


def bundle_problems(task) -> List[str]:
    """Why the task's current bundle (Task.mc_bundle()) cannot be used to build its suites (empty: it can)."""
    want = task.mc_bundle_fingerprint
    if want is None:
        return []
    try:
        bundle = task.mc_bundle()
    except RuntimeError as e:
        return [f"{task.name}: {e}"]
    if bundle is None or not Path(bundle).is_dir():
        return [f"{task.name}: no MC bundle at {bundle} (the site configuration names it)"]
    got = fingerprint(bundle, task.mc_bundle_evaluator_files)
    if got != want:
        return [f"{task.name}: {bundle} is not the task's MC bundle (fingerprint {got[:12]}, the task's {want[:12]}): "
                f"point the site configuration at the current bundle, or rebuild it and record its fingerprint in the "
                f"task"]
    return []


def dataset_problems(task, datasets: Iterable[Path]) -> List[str]:
    """The datasets whose mc/ is not the task's current bundle (empty: every one is)."""
    want = task.mc_bundle_fingerprint
    if want is None:
        return []
    bad = []
    for ds in datasets:
        mc = Path(ds) / "mc"
        got = fingerprint(mc) if mc.is_dir() else None
        if got != want:
            bad.append(f"{ds}: its mc/ is not the task's MC bundle ("
                       + (f"fingerprint {got[:12]}" if got else "no mc/") + f", the task's {want[:12]}): rebuild the "
                       "suite with the current bundle")
    return bad


def require(problems: List[str]) -> None:
    if problems:
        raise SystemExit("MC bundle check failed:\n  " + "\n  ".join(problems))
