"""Participant bundles (runs in `pixi run smoke`): every task's bundle builds within harbor_pack.PARTICIPANT_SBH, its
sbh.cli and task stub import in an isolated interpreter, every file a shipped module opens beside itself
(Path(__file__).with_name(...), e.g. the runner's reaper) is in the bundle, and a module outside the list makes the
pack fail."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sbh import harbor_pack as HP  # noqa: E402
from sbh.task import load_task, task_names, write_participant_task_stub  # noqa: E402


def build(task, dest: Path) -> dict:
    """The bundle's sbh/ and task stub, as the packers build them."""
    shutil.copytree(ROOT / "sbh", dest / "sbh", ignore=HP.participant_ignore())
    (dest / "tasks").mkdir()
    (dest / "tasks" / "__init__.py").write_text("")
    write_participant_task_stub(task, dest / "tasks" / task.name)
    return HP.finish_participant_bundle(dest, task)


def main() -> int:
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in task_names():
            task, b = load_task(name), Path(tmp) / name
            try:
                kept = build(task, b)["kept"]
            except SystemExit as e:
                fails.append(f"{name}: {e}")
                continue
            imp = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, '.'); import sbh.cli, "
                                  f"tasks.{name}"], cwd=b, capture_output=True, text=True)
            if imp.returncode:
                fails.append(f"{name}: the bundle does not import: {imp.stderr.strip().splitlines()[-1]}")
            beside = sorted({(p.parent / m).relative_to(b).as_posix() for p in (b / "sbh").rglob("*.py")
                             for m in re.findall(r"with_name\(\s*[\"']([\w.]+)[\"']\s*\)", p.read_text())
                             if not (p.parent / m).exists()})
            if beside:
                fails.append(f"{name}: the bundle lacks files its modules open beside themselves: {beside}")
            print(f"{name}: {len(kept)} sbh modules, imports {'ok' if not imp.returncode else 'FAIL'}"
                  f"{', missing ' + ', '.join(beside) if beside else ''}")
        saved = HP.PARTICIPANT_SBH
        HP.PARTICIPANT_SBH = tuple(f for f in saved if f != "runner.py")
        try:
            build(load_task("gauss_peak"), Path(tmp) / "mutant")
            fails.append("a bundle shipping a module outside PARTICIPANT_SBH was not refused")
        except SystemExit:
            print("a module outside PARTICIPANT_SBH is refused")
        finally:
            HP.PARTICIPANT_SBH = saved
    for f in fails:
        print("FAIL:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
