"""The ``binned_likelihood`` kind: parameters of interest plus a declared pyhf model the evaluator refits.

Outputs: RESULT.json (POIs, nuisance log), MODEL.json (the pyhf workspace), MODEL_GRID/ (mass grid) and EVENTS.json
(the events behind the observed counts).
"""
from __future__ import annotations

import os
from typing import List

from ..base import Kind


class BinnedLikelihoodKind(Kind):
    name = "binned_likelihood"
    output_names = ("RESULT.json", "MODEL.json", "MODEL_GRID", "EVENTS.json")
    contract_pieces = ("pois", "model", "nuisances")
    model_piece = "model"
    post_score_shape = "-c 16 --mem=64G -t 8:00:00"
    page_files = ("postfit.json",)
    card_files = ("out/MODEL.json",)
    score_files = ("out/MODEL.json",)
    attack_families = {
        "constant": "never reads the data: one fixed answer for every dataset",
        "crasher": "fails on part of the datasets (recovery)",
        "inflator": "overstates every uncertainty (over-coverage)",
        "memorizer": "fixes a parameter at a remembered value and fits the rest",
        "nocalib": "skips the in-situ calibration and trusts the nominal simulation (bias)",
        "wing_inflator": "scales the signal templates of the grid cells outside 117-134 GeV by c (a tilt of the "
                         "units of mu towards the insensitive wings)",
        "grid_far": "scales the signal templates of the grid cells more than 3 GeV from MODEL.json's mass by c",
        "grid_allbut": "scales the signal templates of every grid cell but MODEL.json's own by c",
        "constant_grid": "every MODEL_GRID file holds the template of one mass behind its own label",
        "end_cut": "drops the grid cells below 116 GeV (the insensitive end of the range)",
        "local_gridder": "keeps only the grid cells within 3 GeV of MODEL.json's mass",
        "grid_less": "delivers no MODEL_GRID at all",
        "dead_point": "one grid cell's signal template scaled by 1e-5 (a point that can see nothing)",
        "bound_mover": "declares the POI lower bound lo_b = 0.2 in MODEL.json and every grid cell (mu = 0 "
                       "impossible)",
        "hi_b_clamp": "declares the POI upper bound hi_b = 0.001 wherever its own significance is below 3.5 (a "
                      "null pin by bounds, conditional on its own data)",
        "template_sharpen": "narrows the signal template of every grid cell but MODEL.json's own at a fixed yield "
                            "(each bin raised to the power p and renormalised): more sensitivity per cell, and next "
                            "to the best fit neighbours that see less of a fluctuation",
    }

    def variant_pieces(self, task, pieces):
        """A task with the model piece gets the piece of its signal-strength rule: `mu_marginal` when it scores the
        signal strength on the grid's likelihood marginalised over the mass range (Task.mu_marginal_over_mass; the
        grid's extent and content are then requirements), else `mu_point` (the fit at the reported mass; the grid
        serves the mass alone)."""
        if self.model_piece not in pieces:
            return ()
        return ("mu_marginal",) if getattr(task, "mu_marginal_over_mass", False) else ("mu_point",)

    def post_score(self, run_dir, suite_dir, task, label) -> List[str]:
        """post.post_score (evaluator-only)."""
        from .post import post_score
        return post_score(run_dir, suite_dir, task, label)

    def closure_post_score(self, task) -> bool:
        return True

    def closure_fields(self, run_dir, suite_dir, task) -> dict:
        """closure.json's model-verify cap (post.closure_cap) for a capped task, and for a task scoring mu on the
        mass-marginal density (Task.mu_marginal_over_mass) the grader baseline (scoring.closure_baseline: mu_rule,
        scan_version, baseline_source, reference_pois with the reference's per-dataset H (its density's mean -ln p)
        and prior curve, the suite-average curve); {} otherwise."""
        out = {}
        if getattr(task, "model_verify_cap", False):
            from .post import CAP_KEY, closure_cap
            out[CAP_KEY] = closure_cap(run_dir)
        if getattr(task, "mu_marginal_over_mass", False):
            from .scoring import closure_baseline
            out.update(closure_baseline(run_dir, task, expected_seeds=_sealed_seeds(suite_dir)))
        return out

    def case_loss(self, p, poi, truth, task, fallback, ref, ref_sigma) -> dict:
        """One physics case's losses: under Task.mu_marginal_over_mass the signal POI's density (the POI entry's
        "forecast" as model_records types it, scoring.GraderForecast) is scored by sbh.score.forecast_case_loss
        against the reference's H, its density's mean -ln p -- also when there is no entry (a failed marginal, a
        failed dataset), so the fallback charge uses the same baseline.  A "forecast" of any other type (one a
        RESULT.json entry carried) is never read.  An entry without the grader's forecast (the reported value, while
        the grader's scan is pending) takes the point rule's loss against that same baseline: under the gate a mu case
        never falls back to the scalar sigma_ref (no H: excluded).  Every other POI and every ungated task: the point
        rule."""
        sp = getattr(task, "signal_strength_poi", None) or "mu"
        if not (getattr(task, "mu_marginal_over_mass", False) and poi == sp):
            return super().case_loss(p, poi, truth, task, fallback, ref, ref_sigma)
        from ...score import entropy_ref_loss, forecast_case_loss
        from .scoring import GraderForecast
        fc = p.get("forecast") if p is not None else None
        if p is None or isinstance(fc, GraderForecast):
            return forecast_case_loss(fc, truth, fallback, ref, ref_sigma)
        cl = super().case_loss(p, poi, truth, task, fallback, None, 0.0)
        rl = entropy_ref_loss(ref)
        return dict(cl, ref_loss=rl, ref_source="seed" if rl is not None else None)

    def check_outputs(self, out_dir, result, task, dataset_dir, pylib=None) -> int:
        """`sbh check` on MODEL.json and EVENTS.json (event list against the dev dataset), printed for the
        participant.  Returns 1 if the evaluation would fail the dataset, else 0."""
        import sys
        from pathlib import Path
        from ...snapshot import VENDOR_DIR
        mp = out_dir / "MODEL.json"
        required = getattr(task, "requires_model", False)
        if not os.path.lexists(mp):
            if required:
                print("MODEL.json: MISSING — the contract requires a pyhf-buildable workspace next to RESULT.json")
                return 1
        else:
            from . import model as BLM
            from .eventlist import self_check
            sp = getattr(task, "signal_strength_poi", None) or "mu"
            mu = next((p for p in result["pois"] if p["name"] == sp), None)
            probs = BLM.check_model(mp, mu, result=result, task=task, pylib=pylib)
            unvalidated = BLM.PYHF_MISSING_NOTE in probs
            probs = [p for p in probs if p != BLM.PYHF_MISSING_NOTE]
            gate = [p for p in probs if p.startswith(BLM.GATE_PREFIXES)]
            used = None if unvalidated else sys.modules.get("pyhf")
            if used is not None:
                src = ("vendored in your pylib/" if VENDOR_DIR in Path(used.__file__).parts
                       else f"from {Path(used.__file__).parent.parent}")
                print(f"MODEL.json: checked with pyhf {used.__version__} ({src})"
                      + ("" if used.__version__ == BLM.GRADER_PYHF else
                         f"; the grader builds the model with pyhf {BLM.GRADER_PYHF}, so its verdict can differ"))
            if gate:
                print("MODEL.json: FAILS the contract gate —", "; ".join(gate))
            elif unvalidated:
                print("MODEL.json: contract rules OK (names, nuisance log, binning);", BLM.PYHF_MISSING_NOTE)
            elif probs:
                print("MODEL.json: builds (gate OK); refit diagnostics:", "; ".join(probs))
            else:
                print("MODEL.json: OK — the workspace builds, fits, and its MLE mu matches the report")
            print("  note:", BLM.SELF_CHECK_REFIT_NOTE)
            if gate and required:
                return 1
            if getattr(task, "event_list_from", None) and not self_check(out_dir, dataset_dir):
                return 1
        return 0

    def post_stage(self):
        from . import grader
        return grader

    def output_gate(self, out, result, task, self_check=False, pylib=None, dataset_dir=None):
        """For a task that requires a model, model.gate: MODEL.json must exist and build under the contract rules."""
        if not getattr(task, "requires_model", False):
            return [], []
        from . import model
        return model.gate(out / "MODEL.json", result, task, self_check, pylib)


    score_inputs_takes_suite = True

    def score_inputs(self, run_dir, records, truths, task, suite_dir=None):
        """With Task.model_poi_sources, score the grader's POIs from the declared model (scoring.model_records);
        otherwise the reported ones.  A gated task (Task.mu_marginal_over_mass) names its mu rule ("mu_rule", the
        summary's) and hands model_records the closure's prior curves (sbh.score.load_closure_prior_curves)."""
        if not getattr(task, "model_poi_sources", None):
            return {"records": records, "source": "reported", "info": None, "verify_rows": None}
        from . import scoring as S
        vrows = S.load_verify_rows(run_dir)
        gated = bool(getattr(task, "mu_marginal_over_mass", False))
        closure = None
        if gated and suite_dir is not None:
            from ...score import load_closure_prior_curves
            closure = load_closure_prior_curves(suite_dir, getattr(task, "signal_strength_poi", None) or "mu")
        scored, info = S.model_records(records, vrows, task, closure=closure)
        return {"records": scored, "source": "model", "info": info, "verify_rows": vrows,
                "signal_poi": getattr(task, "signal_strength_poi", None),
                "mu_rule": S.MU_RULE if gated else None, "closure_curves": closure,
                "mu_baseline_id": (closure or {}).get("baseline_id") if gated else None}

    def mark_provisional(self, v2, v2_cases, inputs):
        from .scoring import mark_provisional
        mark_provisional(v2, v2_cases, inputs["info"])

    def case_status(self, inputs):
        from .scoring import case_status
        return case_status(inputs.get("info"))

    def declared_yield(self, run_dir, records, truths, task):
        from .scoring import declared_yield_response
        return declared_yield_response(run_dir, records, truths, task)

    def background_flags(self, run_dir, records, task):
        from .scoring import background_flags
        return background_flags(run_dir, records, task)

    def measured_discovery(self, inputs):
        from .scoring import Z_DISCOVERY
        info = inputs["info"] or {}
        sp = inputs.get("signal_poi") or "mu"

        def z_model(r):
            return (info.get(r["seed"], {}).get(sp) or {}).get("z_obs")
        return {"claim": lambda r: (z_model(r) or 0.0) >= Z_DISCOVERY, "has": lambda r: z_model(r) is not None}

    def score_terms(self, run_dir, records, truths, task, inputs, v2, leak, yld):
        """The kind's S_v3 terms, and the fallbacks the grader and the scorer took on the run's completed datasets
        ("fallbacks", scoring.fallbacks: counted; none under the shipped rule, whose summary counts none)."""
        from . import scoring as S
        from ...hep.calibration import calibration_cases
        from ...score import SHIPPED_RULE, record_ok, score_rule
        from_fit = inputs["source"] == "model"
        vrows = inputs.get("verify_rows")
        strict = bool(getattr(task, "grader_strict", False))
        shipped = score_rule(task) == SHIPPED_RULE
        ungraded = ({rec["seed"] for rec in records if record_ok(rec)
                     and str(((vrows or {}).get(rec["seed"]) or {}).get("status") or "") != "ok"
                     and (shipped or ((vrows or {}).get(rec["seed"]) is not None
                                      and not S.record_fault(vrows[rec["seed"]])))} if strict else None)
        cov = None
        if getattr(task, "requires_model", False) and (getattr(task, "calibration_scored", None)
                                                       or getattr(task, "completeness_units", None)):
            cov = {rec["seed"]: S.model_coverage(run_dir, rec, task, ungraded) for rec in records}
        cal_cases = calibration_cases(records, truths, task, coverage=cov)
        systc = S.completeness_cases(run_dir, records, truths, task, ungraded=ungraded, coverage=cov)
        left = None if shipped else {}
        gofc = S.gof_cases(run_dir, records, combined=from_fit, strict=strict,
                           rows=({k: S._as_shipped(v) for k, v in vrows.items()} if shipped and vrows else None),
                           shipped=shipped, left_out=left)
        truth = claims = tcases = crows = gate = None
        if from_fit:
            tcases = S.truth_cases(records, inputs["info"], truths, task)
            crows = S.claims_rows(records, vrows, inputs["info"], truths, task)
            truth, claims = S.truth_term(tcases), S.claims_term(crows)
            gate = S.calibration_gate(leak, yld, task)
        frows = None if shipped else vrows if vrows is not None else (
            S.load_verify_rows(run_dir) if getattr(task, "requires_model", False) else None)
        done = {rec["seed"] for rec in records if record_ok(rec)}
        repro, systemic = (None, None) if shipped else S.fault_split(frows, done)
        cfaults = S.claims_stage_faults(vrows, done) if (from_fit and not shipped) else None
        terms, pending, extra = S.v3_terms(v2, cal_cases, gofc, truths, len(records), syst_cases=systc, truth=truth,
                                           claims=claims, cal_gate=gate, rule=S.FIT_RULE if from_fit else None,
                                           task=task, gof_left_out=left, reproduced=repro, claims_faults=cfaults,
                                           systemic=systemic)
        return {"terms": terms, "pending": pending, "extra": extra,
                "fallbacks": None if shipped else S.fallbacks(records, frows, inputs.get("info"), gofc, task),
                "private": {"v3_calibration_cases": cal_cases, "v3_completeness_cases": systc, "v3_gof_cases": gofc,
                            "v3_model_coverage": cov,
                            "fit_rule": S.FIT_RULE if from_fit else None, "fit_info": inputs["info"],
                            "v4_truth_cases": tcases, "v4_claims": crows,
                            **({"gof_left_out": left or None, "reproduced_faults": repro or None,
                                "systemic_faults": systemic or None} if not shipped else {})},
                "gof_cases": gofc, "calibration_response": {"gate": gate} if from_fit else {}}


def _sealed_seeds(suite_dir):
    """The suite's sealed datasets (manifest.json, its public seed left out): every one needs a row in the reference's
    verification before the grader baseline is written.  None (no completeness test) without a readable manifest."""
    if suite_dir is None:
        return None
    from pathlib import Path
    from ...score import read_json_maybe_shielded
    try:
        man = read_json_maybe_shielded(Path(suite_dir) / "manifest.json") or {}
        return [s["seed"] for s in man.get("seeds") or [] if not s.get("public")]
    except (OSError, ValueError, KeyError, TypeError):
        return None


KIND = BinnedLikelihoodKind()
