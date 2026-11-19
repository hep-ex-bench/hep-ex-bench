"""Packing helpers of `sbh pack`: the evaluator-only module list, block-marker stripping, the participant copy
filter, and import-closure pruning of a shipped sbh/ tree."""
from __future__ import annotations

import ast
import builtins
import json
import shutil
from pathlib import Path

from . import ROOT

EVALUATOR_ONLY = ("score.py", "metrics.py", "attackers.py", "tripwire.py", "report_card.py", "harbor_pack.py",
                  "yardstick.py", "attackgen.py", "rescore.py", "provenance.py",
                  "kinds/binned_likelihood/grid_attacks.py", "kinds/binned_likelihood/marginal_check.py",
                  "runstore.py", "hep/calibration.py", "hep/coverage.py",
                  "kinds/binned_likelihood/massgrid.py", "kinds/binned_likelihood/refit.py",
                  "kinds/binned_likelihood/scoring.py", "kinds/binned_likelihood/verify.py",
                  "kinds/binned_likelihood/verify_fit.py", "kinds/binned_likelihood/verify_gof.py",
                  "kinds/binned_likelihood/verify_grid.py", "kinds/binned_likelihood/verify_combine.py",
                  "kinds/binned_likelihood/grader.py", "kinds/binned_likelihood/post.py",
                  "kinds/binned_likelihood/jax_engine.py", "kinds/binned_likelihood/seq_toys.py",
                  "kinds/binned_likelihood/seq_gof.py", "kinds/binned_likelihood/cpu_pin.py",
                  "kinds/binned_likelihood/postfit_export.py", "kinds/binned_likelihood/workspace_to_combine.py",
                  "kinds/binned_likelihood/grid_to_combine_mh.py", "kinds/corrections/measurement_verify.py", "kinds/corrections/README.md",
                  "kinds/corrections/scoring.py", "kinds/corrections/refit.py", "kinds/corrections/peak.py",
                  "kinds/corrections/closure.py", "kinds/corrections/card.py", "kinds/corrections/post.py",
                  "kinds/corrections/comparison.py", "kinds/corrections/balance.py",
                  "kinds/scan_likelihood/verify.py", "kinds/scan_likelihood/post.py",
                  "kinds/scan_likelihood/scoring.py", "kinds/scan_likelihood/density.py",
                  "kinds/scan_likelihood/closure.py")

PARTICIPANT_SBH = ("__init__.py", "__main__.py", "budget.py", "cli.py", "contract.py", "reaper.py", "runner.py", "snapshot.py",
                   "task.py", "hep/__init__.py", "hep/names.py", "hep/nuisances.py", "kinds/__init__.py",
                   "kinds/base.py", "kinds/pois/__init__.py", "kinds/binned_likelihood/__init__.py",
                   "kinds/binned_likelihood/eventlist.py", "kinds/binned_likelihood/model.py",
                   "kinds/corrections/__init__.py", "kinds/corrections/check.py",
                   "kinds/corrections/measurement.py", "kinds/corrections/outputs.py",
                   "kinds/scan_likelihood/__init__.py", "kinds/scan_likelihood/scan.py",
                   "kinds/scan_likelihood/scaling.py")

CONTRACT_SOURCES = ("contract_parts.md",)


DEV_ONLY_BEGIN = ">>> dev-harness only"
DEV_ONLY_END = "<<< dev-harness only"
EVAL_ONLY_BEGIN = ">>> evaluator only"
EVAL_ONLY_END = "<<< evaluator only"


def strip_dev_only(text: str, where="", begin: str = DEV_ONLY_BEGIN, end: str = DEV_ONLY_END) -> str:
    """`text` without its blocks between `begin` and `end` (marker lines included); an unclosed, nested or stray
    marker fails the pack."""
    out, inside = [], False
    for ln in text.splitlines(keepends=True):
        if begin in ln or end in ln:
            opening = begin in ln
            if opening == inside:
                raise SystemExit(f"{where}: {'nested' if inside else 'stray'} block marker: {ln.strip()!r}")
            inside = opening
        elif not inside:
            out.append(ln)
    if inside:
        raise SystemExit(f"{where}: '{begin}' without its '{end}'")
    return "".join(out)


def strip_blocks_tree(root: Path, pairs=((DEV_ONLY_BEGIN, DEV_ONLY_END),)) -> int:
    """strip_dev_only for each (begin, end) pair, in place, on every text file under `root`; returns files changed."""
    root = Path(root)
    n = 0
    for f in ([root] if root.is_file() else sorted(root.rglob("*"))):
        if not f.is_file() or f.is_symlink():
            continue
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue
        new = text
        for begin, end in pairs:
            if begin in new or end in new:
                new = strip_dev_only(new, f, begin, end)
        if new != text:
            f.write_text(new)
            n += 1
    return n




def _python_problems(path: Path):
    """Syntax errors and names used but bound nowhere in the file -- the trace of a stripped block that kept code
    still refers to.  Scopes are not modelled (a tripwire for whole missing definitions, not a linter)."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as e:
        return [f"syntax: {e}"]
    bound = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__", "__package__", "__path__"}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            if not isinstance(n, ast.Lambda):
                bound.add(n.name)
            if not isinstance(n, ast.ClassDef):
                args = n.args
                bound |= {a.arg for a in args.args + args.posonlyargs + args.kwonlyargs}
                bound |= {a.arg for a in (args.vararg, args.kwarg) if a}
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            bound |= set(n.names)
        elif isinstance(n, ast.MatchAs) and n.name:
            bound.add(n.name)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return [f"name {x!r} is used but bound nowhere" for x in sorted(used - bound)]


def check_python_tree(root: Path) -> None:
    """Fail the pack unless every .py under the stripped copy `root` parses and binds every name it uses."""
    bad = []
    for f in sorted(Path(root).rglob("*.py")):
        bad += [f"{f}: {p}" for p in _python_problems(f)]
    if bad:
        raise SystemExit("shipped python after stripping:\n  " + "\n  ".join(bad[:20]))


def _module_imports(path: Path, pkg_root: Path):
    """Paths under `pkg_root` of the sbh modules a file imports (absolute or relative, incl. `from pkg import mod`)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    rel_pkg = path.parent.relative_to(pkg_root.parent).parts
    out = set()

    def add(parts):
        p = pkg_root.parent.joinpath(*parts)
        if p.with_suffix(".py").is_file():
            out.add(p.with_suffix(".py"))
        elif (p / "__init__.py").is_file():
            out.add(p / "__init__.py")

    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                parts = a.name.split(".")
                if parts[0] == pkg_root.name:
                    add(parts)
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                base = list(rel_pkg[:len(rel_pkg) - (n.level - 1)]) if n.level > 1 else list(rel_pkg)
                base = base[:len(rel_pkg) - n.level + 1]
                parts = base + (n.module.split(".") if n.module else [])
            else:
                parts = n.module.split(".") if n.module else []
                if parts[:1] != [pkg_root.name]:
                    continue
            add(parts)
            for a in n.names:
                add(parts + [a.name])
    return out


def prune_to_closure(pkg_dir: Path, roots) -> list:
    """Delete from the sbh/ copy at `pkg_dir` every file outside the import closure of `roots` (plus kept packages'
    __init__.py), non-python files included.  Returns the deleted paths."""
    pkg_dir = Path(pkg_dir)
    keep, todo = set(), [Path(r) for r in roots]
    while todo:
        f = todo.pop()
        for dep in _module_imports(f, pkg_dir):
            if dep not in keep:
                keep.add(dep)
                todo.append(dep)
    keep |= {Path(r) for r in roots if Path(r).is_relative_to(pkg_dir)}
    for f in list(keep):
        for parent in f.relative_to(pkg_dir).parents:
            init = pkg_dir / parent / "__init__.py"
            if init.is_file():
                keep.add(init)
    gone = []
    for f in sorted(pkg_dir.rglob("*")):
        if f.is_file() and f not in keep:
            f.unlink(); gone.append(f)
    for d in sorted((p for p in pkg_dir.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    return gone


def finish_participant_bundle(bundle: Path, task=None) -> dict:
    """Finish a copied bundle: strip both block kinds from sbh/ and dev-only blocks from tasks/, prune sbh/ to the
    self-check's import closure (entry points, the reaper, task stubs, the task's kind, loaded by name), check every
    .py.
    Returns a summary."""
    from .kinds import KINDS
    bundle = Path(bundle)
    stripped = strip_blocks_tree(bundle / "sbh", ((DEV_ONLY_BEGIN, DEV_ONLY_END), (EVAL_ONLY_BEGIN, EVAL_ONLY_END)))
    stripped += strip_blocks_tree(bundle / "tasks")
    roots = [bundle / "sbh" / "__main__.py", bundle / "sbh" / "cli.py", bundle / "sbh" / "reaper.py"] \
        + sorted((bundle / "tasks").rglob("*.py"))
    kind = KINDS[getattr(task, "kind", "pois")].split(".")[2:] if task is not None else []
    roots += [bundle / "sbh" / "kinds" / "/".join(kind) / "__init__.py"] if kind else []
    gone = prune_to_closure(bundle / "sbh", [r for r in roots if r.is_file()])
    check_python_tree(bundle / "sbh")
    check_python_tree(bundle / "tasks")
    kept = sorted(p.relative_to(bundle).as_posix() for p in (bundle / "sbh").rglob("*.py"))
    extra = sorted(p.relative_to(bundle / "sbh").as_posix() for p in (bundle / "sbh").rglob("*")
                   if p.is_file() and p.relative_to(bundle / "sbh").as_posix() not in PARTICIPANT_SBH)
    if extra:
        raise SystemExit(f"participant bundle: sbh/ would ship {extra}, not in harbor_pack.PARTICIPANT_SBH; add them "
                         f"there if participants should have them, else keep them out of the self-check's imports")
    return {"stripped": stripped, "dropped": [p.relative_to(bundle).as_posix() for p in gone], "kept": kept}


def participant_ignore():
    """shutil.copytree ignore for a bundle's sbh/ copy; EVALUATOR_ONLY matches by path relative to sbh/."""
    root = (ROOT / "sbh").resolve()
    junk = shutil.ignore_patterns("__pycache__", "*.pyc")

    def ignore(d, names):
        d = Path(d).resolve()
        if not d.is_relative_to(root):
            return set(junk(d, names)) | {n for n in names if n in EVALUATOR_ONLY}
        rel = d.relative_to(root)
        return set(junk(d, names)) | {n for n in names if (rel / n).as_posix() in EVALUATOR_ONLY
                                      or n in CONTRACT_SOURCES}
    return ignore


