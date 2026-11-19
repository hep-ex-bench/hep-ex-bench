"""The MC bundle fingerprint (sbh/bundle.py; quick tier): a copy anywhere has the bundle's fingerprint, and so does one
whose JSON is reformatted or reworded in its documentation entries; a changed JSON value, file size or file list does
not; the evaluator-side root entries are left out; links are followed; a task's current bundle and a suite's datasets
are refused when they are not the bundle the task records, and nothing is checked for a task that records none.

    python checks/mc_bundle_check.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sbh.bundle import bundle_problems, dataset_problems, fingerprint  # noqa: E402

FAIL = []


def check(name: str, ok: bool) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAIL.append(name)


class FakeTask:
    name = "fake"
    mc_bundle_evaluator_files = ("manifest.json",)

    def __init__(self, bundle, fp):
        self.bundle, self.mc_bundle_fingerprint = bundle, fp

    def mc_bundle(self):
        return self.bundle


def write_bundle(root: Path) -> Path:
    for proc, n in (("DY", 3), ("TT", 2)):
        (root / proc).mkdir(parents=True)
        (root / proc / "meta.json").write_text('{"effective_xsec_pb": 1.0, "n_selected": %d}' % n)
        for i in range(n):
            (root / proc / f"part{i}.root").write_bytes(b"x" * (100 + i))
    (root / "cross_sections.json").write_text('{"DY": 1.0, "TT": 87.58}')
    return root


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        b = write_bundle(tmp / "bundle")
        fp = fingerprint(b)
        copy = tmp / "elsewhere" / "copy"
        shutil.copytree(b, copy)
        check("a copy elsewhere has the bundle's fingerprint", fingerprint(copy) == fp)
        (copy / "TT" / "meta.json").write_text('{"effective_xsec_pb": 1.0, "n_selected": 3}')
        check("a changed JSON value changes it", fingerprint(copy) != fp)
        (copy / "TT" / "meta.json").write_text('{\n "n_selected": 2,\n "effective_xsec_pb": 1.0,\n "note": "reworded"\n}')
        (copy / "cross_sections.json").write_text('{"_comment": "x", "convention": "y", "DY": 1.0, "TT": 87.58}')
        check("a reformatted JSON and reworded documentation entries keep it", fingerprint(copy) == fp)
        shutil.rmtree(copy)
        shutil.copytree(b, copy)
        (copy / "TT" / "part1.root").write_bytes(b"x" * 999)
        check("a changed file size changes it", fingerprint(copy) != fp)
        shutil.rmtree(copy)
        shutil.copytree(b, copy)
        (copy / "TT" / "part2.root").write_bytes(b"x")
        check("an extra file changes it", fingerprint(copy) != fp)
        shutil.rmtree(copy)
        shutil.copytree(b, copy)
        (copy / "manifest.json").write_text('{"digest": "evaluator"}')
        check("an evaluator-side root entry is left out when named",
              fingerprint(copy, ("manifest.json",)) == fp and fingerprint(copy) != fp)
        ds = tmp / "suite" / "sealed" / "seed_001"
        for proc in ("DY", "TT"):
            (ds / "mc" / proc).mkdir(parents=True)
            for f in (b / proc).iterdir():
                os.symlink(f, ds / "mc" / proc / f.name)
        os.symlink(b / "cross_sections.json", ds / "mc" / "cross_sections.json")
        check("a dataset's mc/ of links has the bundle's fingerprint", fingerprint(ds / "mc") == fp)
        ds2 = tmp / "suite" / "public" / "seed_000"
        ds2.mkdir(parents=True)
        os.symlink(b, ds2 / "mc")
        check("a linked mc/ directory is followed", fingerprint(ds2 / "mc") == fp)

        task = FakeTask(b, fp)
        check("the current bundle passes", bundle_problems(task) == [])
        check("the datasets on it pass", dataset_problems(task, [ds, ds2]) == [])
        stale_dir = tmp / "stale"
        shutil.copytree(b, stale_dir)
        (stale_dir / "cross_sections.json").write_text('{"DY": 1.0, "TT": 52.7}')
        check("a stale bundle is refused", len(bundle_problems(FakeTask(stale_dir, fp))) == 1)
        old = tmp / "suite" / "sealed" / "seed_002"
        shutil.copytree(stale_dir, old / "mc")
        (tmp / "suite" / "sealed" / "seed_003").mkdir(parents=True)
        bad = dataset_problems(task, [ds, old, tmp / "suite" / "sealed" / "seed_003"])
        check("a dataset on another bundle and one without mc/ are refused, the current one not",
              len(bad) == 2 and "seed_002" in bad[0] and "no mc/" in bad[1])
        check("a missing bundle is refused", len(bundle_problems(FakeTask(tmp / "nowhere", fp))) == 1)
        free = FakeTask(None, None)
        check("a task that records no fingerprint is not checked",
              bundle_problems(free) == [] and dataset_problems(free, [old]) == [])
    print(f"mc bundle check: {'PASS' if not FAIL else 'FAIL'}")
    return 1 if FAIL else 0



if __name__ == "__main__":
    raise SystemExit(main())
