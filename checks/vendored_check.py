"""Vendored packages: refused unless the task allows them (sbh/snapshot.py; runs in `pixi run check`, offline).

Builds a submission whose pylib/ holds one package installed from a wheel (its files match its RECORD), plus an
edited library file and a console script, and checks that a task's own vendor directory accepts it, and that a task
without a vendor directory (Task.vendor_dir None) refuses installed packages wherever they sit (pylib/, a renamed
directory, site-packages, an egg-info).
"""
from __future__ import annotations

import base64
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sbh import snapshot  # noqa: E402

PKG, VER, TAG = "fakepkg", "1.0", "py3-none-any"
FILES = {"fakepkg/__init__.py": b"X = 1\n" * 300000, "fakepkg/core.py": b"def f():\n    return 2\n" * 60000,
         "fakepkg/edited.py": b"ORIGINAL = True\n"}


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _submission(d: Path) -> None:
    (d / "run.sh").write_text("#!/bin/sh\nexec python3 analyze.py \"$1\" \"$2\"\n")
    (d / "analyze.py").write_text("print('analysis')\n")
    pl = d / "pylib"
    dist = f"{PKG}-{VER}.dist-info"
    for n, data in FILES.items():
        (pl / n).parent.mkdir(parents=True, exist_ok=True)
        (pl / n).write_bytes(data)
    (pl / "fakepkg" / "edited.py").write_bytes(b"ORIGINAL = False  # edited after install\n")
    (pl / "bin").mkdir()
    (pl / "bin" / "fakepkg").write_text("#!/somewhere/python\n")
    (pl / dist).mkdir()
    (pl / dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {PKG}\nVersion: {VER}\n")
    (pl / dist / "WHEEL").write_text(f"Wheel-Version: 1.0\nTag: {TAG}\n")
    rec = [f"{n},sha256={_b64(data)},{len(data)}" for n, data in FILES.items()]
    rec += [f"../../bin/fakepkg,sha256={_b64(b'#!python\n')},9", f"{dist}/RECORD,,"]
    (pl / dist / "RECORD").write_text("\n".join(rec) + "\n")


def main() -> int:
    bad = []
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        sub = t / "sub"
        sub.mkdir()
        _submission(sub)
        for where, files in (("pylib", {"pylib/x-1.0.dist-info/RECORD": b""}),
                             ("deps", {"deps/x-1.0.dist-info/RECORD": b"", "deps/x/__init__.py": b""}),
                             ("lib/python3.14/site-packages", {"lib/python3.14/site-packages/x.py": b""}),
                             (".", {"x.egg-info/PKG-INFO": b""})):
            d = t / f"inst_{where.replace('/', '_')}"
            (d).mkdir()
            (d / "run.sh").write_text("#!/bin/sh\n")
            for n, data in files.items():
                (d / n).parent.mkdir(parents=True, exist_ok=True)
                (d / n).write_bytes(data)
            v = snapshot.hygiene(d, snapshot.list_files(d))
            if not any(x.startswith(f"installed packages committed: {where}/") for x in v):
                bad.append(f"hygiene() without a vendor directory let installed packages in {where}/ through: {v}")
        v = snapshot.hygiene(sub, snapshot.list_files(sub), vendor_dir=snapshot.VENDOR_DIR)
        if v:
            bad.append(f"hygiene() refused a task's own vendor directory: {v}")
    for b in bad:
        print("FAIL:", b)
    print("vendored packages" + (" FAILED" if bad else " ok"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
