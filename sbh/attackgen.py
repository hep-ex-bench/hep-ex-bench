"""Generated attackers: each is the task's reference with one attack switched on, built at stress time.

An output kind names the attack families its scoring must catch (Kind.attack_families); a task maps families to its
reference's switches (Task.attackers: {entry: (family, flags, description)}), lists its hand-written attackers
(Task.hand_attackers: {entry: family}) and says why a family is not covered (Task.attack_waivers: {family: why}).
A generated entry is the reference's files plus a run.sh that appends the flags to the reference's exec line; the
core family `symlink` needs no switch: its run.sh moves RESULT.json aside and links it back (output-file hygiene).
Flags of the form `post:<attack> [key=value ...]` name no switch either: the entry is the reference plus a copy of
sbh/kinds/binned_likelihood/grid_attacks.py (grid_attack.py, stdlib only) that its run.sh runs on the output
directory after the pipeline, so the attack is a pure function of the honest outputs (MODEL.json + MODEL_GRID).
The grader must never follow that link: it reads out/ as root under setpriv, so a link to a root-only file would land
the answer key in the run record (confused deputy); this one points at the honest answer, so a grader that follows
links scores it equal to the reference and the stress invariant fails.
Entries are written under work/attackers/<entry>/ and rewritten only where they differ, each entry under a lock
(work/attackers/.<entry>.lock) so concurrent stress jobs see the same bytes and never prune each other's writes.
"""
from __future__ import annotations

import fcntl
import os
import shlex
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import ROOT

OUT = ROOT / "work" / "attackers"
POST_SCRIPT = "grid_attack.py"
GENERATED = ("run.sh", "README.md", POST_SCRIPT)
WIDTH = 120
CORE_FAMILIES = {"symlink": "an honest answer whose RESULT.json is a symlink: the grader must not follow it "
                            "(output-file hygiene; recovery)"}
SYMLINK_TAIL = ['mv "$2/RESULT.json" "$2/result_real.json"', 'ln -s result_real.json "$2/RESULT.json"']
POST_PREFIX = "post:"
README = "# {entry}\n\n{body}\n"
TMP_SUFFIX = ".attackgen-tmp"
GRID_ENTRIES = {
    "wing_inflator_x1p5": ("wing_inflator", "wing_inflator c=1.5",
                           "scales the signal templates of the grid cells outside 117-134 GeV by 1.5"),
    "wing_inflator_x3": ("wing_inflator", "wing_inflator c=3",
                         "scales the signal templates of the grid cells outside 117-134 GeV by 3"),
    "wing_inflator_x10": ("wing_inflator", "wing_inflator c=10",
                          "scales the signal templates of the grid cells outside 117-134 GeV by 10"),
    "grid_far_x1p5": ("grid_far", "grid_far c=1.5",
                      "scales the signal templates of the grid cells more than 3 GeV from MODEL.json's mass by 1.5"),
    "grid_far_x3": ("grid_far", "grid_far c=3",
                    "scales the signal templates of the grid cells more than 3 GeV from MODEL.json's mass by 3"),
    "grid_far_x10": ("grid_far", "grid_far c=10",
                     "scales the signal templates of the grid cells more than 3 GeV from MODEL.json's mass by 10"),
    "grid_allbut_x1p2": ("grid_allbut", "grid_allbut c=1.2",
                         "scales the signal templates of every grid cell but MODEL.json's own by 1.2"),
    "grid_allbut_x1p5": ("grid_allbut", "grid_allbut c=1.5",
                         "scales the signal templates of every grid cell but MODEL.json's own by 1.5"),
    "grid_allbut_x3": ("grid_allbut", "grid_allbut c=3",
                       "scales the signal templates of every grid cell but MODEL.json's own by 3"),
    "grid_allbut_x10": ("grid_allbut", "grid_allbut c=10",
                        "scales the signal templates of every grid cell but MODEL.json's own by 10"),
    "constant_grid": ("constant_grid", "constant_grid",
                      "puts MODEL.json's signal template behind every MODEL_GRID mass label"),
    "end_cut": ("end_cut", "end_cut cut=116", "delivers no grid cell below 116 GeV"),
    "local_gridder": ("local_gridder", "local_gridder half=3",
                      "delivers only the grid cells within 3 GeV of MODEL.json's mass"),
    "grid_less": ("grid_less", "grid_less", "delivers no MODEL_GRID"),
    "dead_point": ("dead_point", "dead_point factor=1e-5",
                   "scales the signal template of the in-range grid cell farthest from MODEL.json's mass by 1e-5"),
    "bound_mover": ("bound_mover", "bound_mover lo=0.2",
                    "declares the POI lower bound 0.2 in MODEL.json and every grid cell"),
    "hi_b_clamp": ("hi_b_clamp", "hi_b_clamp hi=0.001 zmax=3.5",
                   "declares the POI upper bound 0.001 in MODEL.json and every grid cell on the datasets where its "
                   "own significance is below 3.5"),
    "template_sharpen_p2": ("template_sharpen", "template_sharpen p=2",
                            "sharpens the signal template of every grid cell but MODEL.json's own to the power 2 at a "
                            "fixed yield"),
}


def grid_attack_entries(suffix: str, only: Optional[Tuple[str, ...]] = None) -> Dict[str, Tuple[str, str, str]]:
    """GRID_ENTRIES as a task's Task.attackers entries `attacker_<name>_<suffix>` (labels are unique across tasks,
    hence the suffix); `only` restricts to these GRID_ENTRIES names."""
    return {f"attacker_{name}_{suffix}": (fam, POST_PREFIX + spec, what)
            for name, (fam, spec, what) in GRID_ENTRIES.items() if only is None or name in only}


def post_spec(flags: str) -> Optional[List[str]]:
    """[attack, key=value, ...] of a `post:` flags string, None for ordinary switch flags."""
    if not flags.startswith(POST_PREFIX):
        return None
    toks = shlex.split(flags[len(POST_PREFIX):])
    if not toks:
        raise SystemExit(f"attackers: {flags!r} names no post-processing attack")
    return toks


def families(task) -> Dict[str, str]:
    """The attack families the task's kind must catch: {family: what it is}.  The grid-content families attack the
    marginal mu rule (grid_attacks.FAMILIES); a task scoring mu at one mass has nothing of theirs to catch."""
    from .kinds import get_kind
    fam = {**CORE_FAMILIES, **getattr(get_kind(task), "attack_families", {})}
    if not getattr(task, "mu_marginal_over_mass", False):
        from .kinds.binned_likelihood.grid_attacks import FAMILIES as GRID
        fam = {f: w for f, w in fam.items() if f not in GRID}
    return fam


def run_sh(reference: Path, entry: str, family: str, what: str, flags: str) -> bytes:
    """The reference's run.sh with the attacker's header comment and the flags appended to its exec line, which must
    be the only one and end the command (no continuation or trailing comment).  `symlink`: the exec line runs
    without exec, then RESULT.json is moved aside and linked back.  A `post:` spec: the exec line runs without exec,
    then grid_attack.py (next to run.sh) runs the attack on the output directory with the reference's python
    (its PY array when run.sh defines one, else `python`)."""
    lines = (reference / "run.sh").read_text().splitlines()
    body = lines[1:]
    while body and body[0].startswith("#"):
        body = body[1:]
    execs = [i for i, ln in enumerate(body) if ln.lstrip().startswith("exec ")]
    if len(execs) != 1 or body[execs[0]].rstrip().endswith("\\") or " #" in body[execs[0]]:
        raise SystemExit(f"attackers: {reference / 'run.sh'} needs exactly one exec line that ends the command, to "
                         f"add the flags to (found {len(execs)})")
    i = execs[0]
    post = post_spec(flags)
    if flags and post is None:
        body[i] = f"{body[i]} {flags}"
    if family == "symlink":
        body[i] = body[i].replace("exec ", "", 1)
        body[i + 1:i + 1] = SYMLINK_TAIL
    if post is not None:
        body[i] = body[i].replace("exec ", "", 1)
        body[i + 1:i + 1] = [f'"${{PY[@]:-python}}" "$(dirname "$0")/{POST_SCRIPT}" {shlex.quote(post[0])} "$2" '
                             + " ".join(shlex.quote(t) for t in post[1:])]
    head = [lines[0], *textwrap.wrap(f'ATTACKER "{entry}": the reference pipeline ({reference.relative_to(ROOT)}), '
                                     f"but {what}.", WIDTH, initial_indent="# ", subsequent_indent="# ",
                                     break_on_hyphens=False),
            "# Generated by sbh/attackgen.py from the reference at stress time."]
    return ("\n".join(head + body) + "\n").encode()


def expected(task, entry: str) -> Dict[str, bytes]:
    """relative path -> bytes of one generated attacker."""
    family, flags, what = task.attackers[entry]
    ref = Path(task.reference_submission)
    files = {str(p.relative_to(ref)): p.read_bytes() for p in sorted(ref.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and str(p.relative_to(ref)) not in GENERATED}
    files["run.sh"] = run_sh(ref, entry, family, what, flags)
    post = post_spec(flags)
    if post is not None:
        from .kinds.binned_likelihood import grid_attacks
        files[POST_SCRIPT] = Path(grid_attacks.__file__).read_bytes()
    stress = getattr(task, "stress_submissions", None)
    duty = ("The stress test requires it to score below the reference." if stress is None or entry in stress
            else task.review_note or "A review case, not a stress attacker.")
    how = (f": its outputs post-processed by `{POST_SCRIPT} {' '.join(post)}`" if post is not None
           else f": `{flags}`" if flags else "")
    body = "\n".join(textwrap.wrap(f"The reference pipeline (`{ref.relative_to(ROOT)}`), but {what}{how}.  {duty}  "
                                   f"Family `{family}`; generated by `sbh/attackgen.py` at stress time.", WIDTH,
                                   break_on_hyphens=False, break_long_words=False))
    files["README.md"] = README.format(entry=entry, body=body).encode()
    return files


def _write(path: Path, data: bytes, mode: int) -> None:
    """Write `data` atomically unless the file already holds it."""
    if path.is_file() and path.read_bytes() == data and (path.stat().st_mode & 0o777) == mode:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=TMP_SUFFIX)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def materialize(task, entry: str) -> Path:
    """work/attackers/<entry>/, written from the task's reference; files the reference no longer has are removed."""
    d = OUT / entry
    want = expected(task, entry)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f".{entry}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for rel, data in want.items():
            _write(d / rel, data, 0o755 if rel == "run.sh" or os.access(task.reference_submission / rel, os.X_OK)
                   else 0o644)
        for p in sorted(d.rglob("*"), reverse=True):
            rel = str(p.relative_to(d))
            if p.is_file() and rel not in want and "__pycache__" not in p.parts and not p.name.endswith(TMP_SUFFIX):
                p.unlink()
    return d


def generated(tasks) -> Dict[str, Tuple[object, Path]]:
    """{entry: (task, directory)} of every generated attacker of `tasks`, materialized."""
    return {e: (t, materialize(t, e)) for t in tasks for e in getattr(t, "attackers", {})}


def coverage(task) -> List[str]:
    """Problems with the task's attacker families: an entry naming a family its kind lacks, a family neither
    attacked nor waived, a waiver for a family that is attacked."""
    fam = families(task)
    have = {f for f, _, _ in getattr(task, "attackers", {}).values()} | set(getattr(task, "hand_attackers", {}).values())
    waived = getattr(task, "attack_waivers", {})
    out = [f"{task.name}: {e} names family {f!r}, not one of its kind's ({', '.join(sorted(fam))})"
           for e, (f, _, _) in getattr(task, "attackers", {}).items() if f not in fam]
    out += [f"{task.name}: hand-written {e} names family {f!r}, not one of its kind's"
            for e, f in getattr(task, "hand_attackers", {}).items() if f not in fam]
    out += [f"{task.name}: family {f!r} is neither attacked nor waived (Task.attack_waivers)"
            for f in sorted(set(fam) - have - set(waived))]
    out += [f"{task.name}: family {f!r} is waived but attacked" for f in sorted(set(waived) & have)]
    return out
