"""The corrections kind: correction functions (corrections.json, a correctionlib CorrectionSet) answering the
dataset's request.json, each with its uncertainty and its declared measurement (MEASUREMENT.json / .npz).  Contract
text: contract_parts.md; task base: task.CorrectionsTask.

Participant-facing (the bundle's `sbh check`, the runner's gate): outputs.py (reading, checking, evaluating
corrections.json), measurement.py (the measurement checks), check.py.  Evaluator-only, imported where used:
scoring.py (scoring, scoreboard), post.py (per-dataset checks) with refit.py (measurement refits) and the closure
comparison (comparison.py: peak.py, the Z peak; balance.py, the jet balance), closure.py (closure criteria), card.py
(report card, page facts).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

from ..base import Kind
from .outputs import MAX_BYTES, result_problems as _result_problems

VERIFY_FILE = "measurement_verify.json"
POINTS_FILE = "function_points.npz"
POST_REFERENCE_WALL_MAX_S = 1950
POST_SLACK_S = 600


class CorrectionsKind(Kind):
    name = "corrections"
    output_names = ("RESULT.json", "corrections.json", "MEASUREMENT.json", "MEASUREMENT.npz")
    result_max_bytes = MAX_BYTES
    reports_pois = False
    contract_pieces = ("corrections",)
    forbidden_outputs = ("corrections.json",)
    verifiable = False
    card_files = (VERIFY_FILE, POINTS_FILE)
    score_files = ("out/corrections.json",)
    attack_families = {
        "constant": "never reads the data: nominal corrections for every dataset",
        "nocalib": "leaves its measured calibration out of what it reports and of what it measures with it",
        "inflator": "overstates every uncertainty (over-coverage)",
        "shrinker": "understates every uncertainty (under-coverage)",
        "bias": "shifts every answer by a multiple of its own uncertainty",
        "crasher": "fails on part of the datasets (recovery)",
        "fabricator": "declares a measurement the dataset contradicts",
        "pinner": "reports nominal with no uncertainty wherever its measurement is compatible with it",
    }

    def result_problems(self, result: Any, task=None) -> List[str]:
        """RESULT.json: an object whose optional "covariances" and "joint_covariances" have their shapes
        (outputs.result_problems).  The answers are in corrections.json; a nuisance log is recorded unchecked (the
        contract's nuisance section is for a statistical-model analysis, which a corrections task is not)."""
        return _result_problems(result)

    def output_gate(self, out: Path, result: Any, task, self_check: bool = False, pylib: Optional[Path] = None,
                    dataset_dir: Optional[Path] = None) -> Tuple[List[str], List[str]]:
        from .check import output_gate
        return output_gate(out, result, dataset_dir, self_check)

    def check_report(self, a, rec, dataset_dir: Path, work: Path, info: dict, env_prefix, task, level: str,
                     msg: str) -> Optional[int]:
        from .check import check_report
        return check_report(a, rec, dataset_dir, work, info, env_prefix, task, level, msg)

    def closure_criteria(self, pub: dict, priv: dict, task, a, offsets=None, scored_calibration=None):
        from .closure import closure_criteria
        return closure_criteria(pub, priv, task, a, offsets=offsets, scored_calibration=scored_calibration)

    def score_run(self, run_dir: Path, suite_dir: Path, task, label: str, sigma_ref=None, cfg=None,
                  making_baseline: bool = False) -> Optional[dict]:
        from .scoring import score_run
        return score_run(run_dir, suite_dir, task, label, sigma_ref=sigma_ref, making_baseline=making_baseline)

    def build_card(self, run_dir: Path, suite_dir: Path, task, **kw) -> dict:
        from .card import build_card
        return build_card(run_dir, suite_dir, task, **kw)

    def scoreboard_rows(self, summaries: List[dict]) -> List[str]:
        from .scoring import scoreboard_rows
        return scoreboard_rows(summaries)

    def post_shape(self, task) -> Optional[str]:
        """Slurm shape of the post stage of one dataset (post.py): the task's post_cpus / post_memory_mb, wall the
        budgets' sum at POST_REFERENCE_WALL_MAX_S plus POST_SLACK_S; a task stating none of them is refused."""
        cpus, mem = int(getattr(task, "post_cpus", 0) or 0), int(getattr(task, "post_memory_mb", 0) or 0)
        if not (cpus > 0 and mem > 0):
            raise SystemExit(f"{task.name}: a corrections task states its post stage's shape (post_cpus, "
                             "post_memory_mb)")
        if not (float(getattr(task, "verify_budget_y", 0) or 0) > 0 and float(getattr(task, "peak_budget_y", 0) or 0) > 0
                and getattr(task, "compute_yardstick_k", None)):
            raise SystemExit(f"{task.name}: a corrections task states its post stage's budgets (verify_budget_y, "
                             "peak_budget_y, in units of the reference's wall that its compute yardstick times: "
                             "compute_yardstick_k)")
        t = math.ceil((float(task.verify_budget_y) + float(task.peak_budget_y)) * POST_REFERENCE_WALL_MAX_S) + POST_SLACK_S
        return f"-c {cpus} --mem {mem}M -t {t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"

    def post_dataset(self, run_dir: Path, suite_dir: Path, task, seed: str) -> None:
        from .post import check
        check(run_dir, suite_dir, task, seeds={seed})

    def report_extras(self, closure: dict) -> dict:
        from .card import report_extras
        return report_extras(closure)

    def stale_score(self, summary: dict, closure: dict) -> Optional[str]:
        """A run scored against another baseline than the closure's: the summary's baseline facts (evaluation points
        and floors, accuracy factors) against the closure reference's."""
        mine = summary.get("functions") or {}
        theirs = ((closure.get("summary") or {}).get("functions")) or {}
        keys = [k for k in ("baseline", "accuracy") if mine.get(k) != theirs.get(k)]
        return (f"scored against another baseline than the suite's closure ({', '.join(keys)} differ): rescore it"
                if keys and theirs else None)


KIND = CorrectionsKind()
