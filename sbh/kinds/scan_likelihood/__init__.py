"""The ``scan_likelihood`` kind: parameters of interest as intervals, measured by the evaluator on a declared pyhf
model whose samples may scale with monomials in the POIs.

Outputs: RESULT.json (POI entries {"name", "value", "err_minus", "err_plus"} inside the task's physical range,
Task.poi_bounds, stat / syst optional, each {"err_minus", "err_plus"}; the scanned POI, the task's first, may list its
68 % set in several pieces as "intervals"; the nuisance log), MODEL.json (a pyhf workspace with `_meta.binning` and the optional
`_meta.poi_scaling`, scaling.py) and EVENTS.json (binned_likelihood's event list).  The evaluator scans the first
POI's profile over its range and fits the others, on the model with its POI scaling.

A pyhf fit would read the scaled samples as a plain product of normfactors, so the kind fits no model with pyhf
alone: the gate builds MODEL.json without fitting it, `sbh check` refits it with the scaling applied to pyhf's
per-sample rates (scaling.refit), and the evaluator grades a model only with its scaling.

Participant-facing (the bundle's `sbh check`, the runner's gate): this module, scan.py (the interval form, a scan's
q <= 1 pieces), scaling.py.  Evaluator-only, imported where used: verify.py (the grader), post.py (the grader as the
per-dataset post stage), scoring.py (the score from the grader's measurement), density.py (the scan density and the
two-piece interval loss), closure.py (the closure's likelihood pulls and criteria).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base import Kind
from . import scan as SC

VERIFY_FILE = "model_verify.json"


@lru_cache(maxsize=1)
def _interval_poi():
    """IntervalPoi, an interval as the core's PoiResult: total = the error on the truth's side (scan.side_sigma), so
    pulls, bias tests and leakage fits divide by it; err_minus / err_plus kept beside it.  Built on first use:
    sbh.metrics is evaluator-only."""
    from dataclasses import dataclass
    from ...metrics import PoiResult

    @dataclass
    class IntervalPoi(PoiResult):
        err_minus: float = 0.0
        err_plus: float = 0.0
    return IntervalPoi


def poi_bounds(task, name: str):
    """(lower, upper) of a POI's physical range from Task.poi_bounds (None = open)."""
    b = (getattr(task, "poi_bounds", None) or {}).get(name) if task is not None else None
    if not b:
        return None, None
    lo, hi = b
    return (None if lo is None else float(lo)), (None if hi is None else float(hi))


def range_text(task, name: str) -> str:
    """The contract's statement of a POI's range with both ends, from Task.poi_bounds (the one source the gate
    reads)."""
    lo, hi = poi_bounds(task, name)
    return f"from {lo:g} to {hi:g}"


class ScanLikelihoodKind(Kind):
    name = "scan_likelihood"
    output_names = ("RESULT.json", "MODEL.json", "EVENTS.json")
    contract_pieces = ("intervals", "scaled_model", "nuisances")
    model_piece = "scaled_model"
    score_files = ("out/MODEL.json", VERIFY_FILE)
    card_files = (VERIFY_FILE,)
    reports_s_phys = False
    verifiable = False
    attack_families = {
        "constant": "never reads the data: one fixed answer for every dataset",
        "crasher": "fails on part of the datasets (recovery)",
        "inflator": "overstates every uncertainty (over-coverage)",
        "deflator": "understates every uncertainty (under-coverage)",
        "memorizer": "fixes a parameter at a remembered value and fits the rest",
        "nocalib": "skips the in-situ calibration and trusts the nominal simulation (bias)",
        "misreport": "reports numbers its declared model does not give (truthfulness)",
        "fabricate": "declares observed counts that are not the dataset's events (the event list)",
        "token": "declares its systematics at token size (completeness)",
        "misfit": "declares a model that does not describe its data (goodness of fit)",
    }

    def poi_problems(self, p: dict, task=None) -> List[str]:
        """The interval form; "intervals" (several pieces) only on the scanned POI, the task's first."""
        errs = SC.interval_problems(p, *poi_bounds(task, p["name"]))
        if "intervals" in p:
            scanned = task.pois[0] if task is not None else None
            if p["name"] != scanned:
                errs.append(f"poi {p['name']}: 'intervals' only on {scanned} (the POI the evaluator scans)")
            elif not errs:
                errs += SC.pieces_problems(p, *poi_bounds(task, p["name"]))
        return errs

    def poi_result(self, p: dict, truth: Optional[float] = None):
        cls = _interval_poi()
        a = SC.interval(p)
        if a is None:
            v = p.get("value") if isinstance(p, dict) else None
            return cls(float(v) if isinstance(v, (int, float)) else float("nan"))
        return cls(a[0], 0.0, 0.0, SC.side_sigma(*a, truth), err_minus=a[1], err_plus=a[2])

    def poi_sigma(self, p: dict, truth: Optional[float] = None) -> float:
        a = SC.interval(p)
        return SC.side_sigma(*a, truth) if a else float("nan")

    def poi_card(self, p: dict) -> Dict[str, Any]:
        a = SC.interval(p)
        pieces = {"intervals": p["intervals"]} if isinstance(p.get("intervals"), list) else {}
        return {**({"err_minus": a[1], "err_plus": a[2]} if a else {}), **pieces}

    def poi_text(self, p: dict) -> str:
        return f"{p['value']:.4g} -{p['err_minus']:.3g} +{p['err_plus']:.3g}"

    def output_gate(self, out, result, task, self_check=False, pylib=None, dataset_dir=None):
        """MODEL.json must exist and pass the gate: binned_likelihood's model.gate with the task's POIs exempt from
        the naming rules and the POI scaling's rules (scaling.problems)."""
        from ..binned_likelihood import model
        from . import scaling
        return model.gate(Path(out) / "MODEL.json", result, task, self_check, pylib, extra_pois=task.pois,
                          extra_rules=lambda spec: scaling.problems(spec, task))

    def check_outputs(self, out_dir, result, task, dataset_dir, pylib=None) -> int:
        """`sbh check` on MODEL.json (the gate, then the refit with its POI scaling beside RESULT.json's numbers,
        scaling.refit_lines: a diagnostic) and EVENTS.json (against the dev dataset).  Returns 1 if the evaluation
        would fail the dataset, else 0."""
        from ..binned_likelihood.eventlist import self_check
        from . import scaling
        probs, notes = self.output_gate(out_dir, result, task, self_check=True, pylib=pylib)
        if probs:
            print("MODEL.json: FAILS the contract gate —", "; ".join(probs))
            return 1
        print("MODEL.json: " + ("contract rules OK (names, nuisance log, binning, POI scaling); " + notes[0]
                                if notes else "OK — the workspace builds and passes the gate"))
        if not notes:
            for line in scaling.refit_lines(Path(out_dir) / "MODEL.json", result, task, pylib):
                print("  " + line)
        print("  note: the evaluator fits your model itself, with the POI scaling, and RESULT.json must report what "
              "it gives")
        if getattr(task, "event_list_from", None) and not self_check(out_dir, dataset_dir):
            return 1
        return 0

    def post_shape(self, task):
        from .post import POST_SHAPE
        return POST_SHAPE

    def post_dataset(self, run_dir, suite_dir, task, seed) -> None:
        from .post import post_dataset
        post_dataset(run_dir, suite_dir, task, seed)

    def closure_fields(self, run_dir, suite_dir, task) -> dict:
        """closure.json's model-verify cap of a capped task (Task.model_verify_cap; post.closure_cap)."""
        if not getattr(task, "model_verify_cap", False):
            return {}
        from ..binned_likelihood.post import CAP_KEY
        from .post import closure_cap
        return {CAP_KEY: closure_cap(run_dir)}

    def verification(self, run_dir) -> Optional[Tuple[str, dict]]:
        from .scoring import verification
        return verification(run_dir)

    def closure_gate(self, suite_dir, task, label) -> Optional[str]:
        from .scoring import closure_gate
        return closure_gate(suite_dir, task, label)

    def score_inputs(self, run_dir, records, truths, task) -> dict:
        from .scoring import score_inputs
        return score_inputs(run_dir, records, truths, task)

    def case_loss(self, p, poi, truth, task, fallback, ref, ref_sigma) -> dict:
        from .scoring import case_loss
        return case_loss(p, poi, truth, task, fallback, ref, ref_sigma)

    def reference_poi(self, v: dict, scored=None) -> dict:
        """The closure's baseline of the reference on a dataset from the grader's entry, the reported one where the
        grader's is pending: its interval (the two-piece baseline) with its mean half-width as "sigma", and for the
        scanned POI the H (mean -ln p) of its scan density (density.reference_entry, from the scored entry)."""
        from .density import reference_entry
        from .scoring import reference_entry_of
        v = reference_entry_of(v, scored)
        em, ep = v.get("err_minus"), v.get("err_plus")
        if em is None or ep is None:
            return reference_entry(super().reference_poi(v), scored)
        return reference_entry({"value": v["value"], "sigma": 0.5 * (em + ep), "err_minus": em, "err_plus": ep},
                               scored)

    def mark_provisional(self, v2, v2_cases, inputs) -> None:
        from ..binned_likelihood.scoring import mark_provisional
        mark_provisional(v2, v2_cases, inputs["info"])

    def case_status(self, inputs):
        from ..binned_likelihood.scoring import case_status
        return case_status(inputs.get("info"))

    def score_terms(self, run_dir, records, truths, task, inputs, v2, leak, yld) -> dict:
        from .scoring import score_terms
        return score_terms(run_dir, records, truths, task, inputs, v2, leak, yld)

    def closure_criteria(self, pub, priv, task, a, offsets=None, scored_calibration=None):
        from .closure import closure_criteria
        return closure_criteria(pub, priv, task, a, offsets=offsets, scored_calibration=scored_calibration)


KIND = ScanLikelihoodKind()
