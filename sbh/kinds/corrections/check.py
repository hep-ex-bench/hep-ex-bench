"""The corrections kind's run gate and self-check report (participant-facing: `sbh check` ships with the bundle)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


def output_gate(out: Path, result, dataset_dir: Path, self_check: bool) -> Tuple[List[str], List[str]]:
    """(problems, notes): corrections.json must be a CorrectionSet file, else the run failed.  A requested
    correction's own problems are notes (it counts as not reported); without correctionlib only the self-check may
    record the evaluation as NOT VALIDATED HERE."""
    from . import outputs as C
    cset, why = C.read_set(Path(out) / C.FILE)
    if why:
        return [why], []
    try:
        request = json.loads((Path(dataset_dir) / "request.json").read_text())
    except (OSError, ValueError, TypeError) as e:
        raise RuntimeError(f"the dataset's request.json is unreadable ({e}): cannot check a corrections run") from None
    problems, notes = C.check_request(cset, result, request)
    if notes and not self_check:
        raise RuntimeError("task requires corrections validation but correctionlib is not importable in the "
                           "evaluator environment -- refusing to waive the gate")
    return [], notes + [f"corrections: {name}: {'; '.join(p)}" for name, p in problems.items() if p]


def check_report(a, rec, ds: Path, work: Path, info: dict, env_prefix, task, level: str, msg: str) -> int:
    """`sbh check`: every reported correction against the contract -- structure, uncertainty, evaluation on a grid
    spanning its domain, its declared measurement (a failure under measurement_required, else reported); an
    unreported one is noted -- and, with --determinism, a second clean run that must agree on those grids."""
    import numpy as np
    from ...cli import _limits, run_one
    from . import outputs as C
    request = json.loads((ds / "request.json").read_text())
    out = work / "run" / "out"
    cset, _ = C.read_set(out / C.FILE)
    problems, notes = C.check_request(cset, rec.result, request)
    for n in notes:
        print("note:", n)
    cl = C.import_correctionlib()
    if cl is not None:
        print(f"corrections: checked with correctionlib {cl.__version__}"
              + ("" if cl.__version__ == C.GRADER_CORRECTIONLIB else
                 f"; the evaluation uses correctionlib {C.GRADER_CORRECTIONLIB}, so its verdict can differ"))
    reported = {n: p for n, p in problems.items() if p != ["missing"]}
    for name in sorted(problems):
        print(f"  {name}: " + ("not reported" if name not in reported else
                               "OK" if not problems[name] else "PROBLEM -- " + "; ".join(problems[name])))
    bad = sorted(n for n, p in reported.items() if p)
    if bad:
        print(f"corrections: {len(bad)} of the {len(reported)} reported corrections do not satisfy the contract and "
              "count as not reported")
        return 1
    print(f"corrections: {len(reported)} of {len(problems)} requested corrections reported, all satisfying the "
          "contract" + (" (structure only; evaluation NOT VALIDATED HERE)" if notes else ""))
    from . import measurement as Ms
    required = getattr(task, "measurement_required", True)
    if not required and not (out / Ms.FILE).exists():
        print(f"measurements: none declared ({Ms.FILE} is optional for this task)")
    else:
        mres = {k: why for k, why in Ms.check(out, ds, log=lambda m: print("  " + m, flush=True)).items()
                if k.rsplit("|", 1)[0] in reported}
        mbad = sorted(k for k, why in mres.items() if why)
        for k in mbad:
            print(f"  {k}: {'PROBLEM' if required else 'NOTE'} -- {mres[k]}")
        if mbad and required:
            print(f"measurements: {len(mbad)} of the {len(mres)} reported cases (correction|flavour) have no "
                  f"measurement that satisfies {Ms.FILE}'s rules and count as not reported")
            return 1
        if mbad:
            print(f"measurements: {len(mbad)} of the {len(mres)} reported cases (correction|flavour) have no "
                  f"declared measurement that passes the checks (recorded; the declaration is optional for this task)")
        elif reported:
            print("measurements: the measurement of every reported correction passes the checks against this "
                  "dataset")
    if a.determinism:
        rec2 = run_one(Path(info["tarball"]), ds, work / "run2", _limits(a, task), task, env_prefix=env_prefix,
                       self_check=True)
        same = rec2.ok
        if same and not notes:
            cset2, _ = C.read_set(work / "run2" / "out" / C.FILE)
            if cset2 is None:
                same = False
            else:
                v1, v2 = C.grid_values(cset, rec.result, request), C.grid_values(cset2, rec2.result, request)
                same = v1.keys() == v2.keys() and all(
                    np.allclose(v1[k], v2[k], rtol=1e-9, atol=1e-12) for k in v1)
        print(f"determinism (second clean run): {'PASS' if same else 'FAIL'}"
              + (" (both runs completed; their corrections were NOT COMPARED HERE without correctionlib)"
                 if same and notes else ""))
        if not same:
            return 1
    if level == "FAIL":
        print(f"self-check FAILED on memory: {msg}")
        return 1
    return 0
