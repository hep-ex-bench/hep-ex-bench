"""The output-kind interface (see sbh/kinds/__init__.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class Kind:
    name: str = "base"
    # Files/directories a run may leave in its output directory besides scratch.
    output_names: Tuple[str, ...] = ("RESULT.json",)
    # RESULT.json size cap in bytes, checked before reading; None = no cap.
    result_max_bytes: Optional[int] = None
    # False: answers live elsewhere and result_problems is the whole RESULT.json check.
    reports_pois: bool = True
    # Further output files a submission may not ship (refused at freeze).
    forbidden_outputs: Tuple[str, ...] = ()
    # Whether `sbh verify` has a rule for the kind.
    verifiable: bool = True
    # Whether a run's summary carries the v1 S_phys (sbh.metrics, linear pulls); the stress invariant orders the
    # attackers by it besides S.  False: s_phys None.
    reports_s_phys: bool = True
    # The pieces of the unified contract its tasks receive by default (Task.contract_pieces; sbh.contract): the
    # template's `<!-- if: <piece> -->` blocks they keep.
    contract_pieces: Tuple[str, ...] = ("pois",)
    # The contract piece of the declared model the evaluator refits (None: the kind declares none); a task holding it
    # states its GOF window and the luminosity encoding (Task.contract_values)
    model_piece: Optional[str] = None
    # Per-dataset files (runs/<seed>/<name>) the report card reads; part of the card's staleness key.
    card_files: Tuple[str, ...] = ()
    # Run-level files the results page reads beside the card (the run directory's <name>); published with the run
    # (sbh/runstore.py).
    page_files: Tuple[str, ...] = ()
    # Per-dataset outputs (runs/<seed>/<name>) the score stage reads besides the record; published with the run, so
    # a published run rescores from the store and its suite.
    score_files: Tuple[str, ...] = ()
    # The attacks its scoring must catch, {family: what it does}; every task of the kind attacks or waives each
    # (Task.attackers, hand_attackers, attack_waivers; sbh/attackgen.py adds the core families).
    attack_families: Dict[str, str] = {}

    def scoreboard_rows(self, summaries: List[dict]) -> List[str]:
        """The kind's own scoreboard table as Markdown lines; [] = none."""
        return []

    def report_extras(self, closure: dict) -> Dict[str, Any]:
        """Extra campaign facts for the results page (closure criteria, baselines, definitions); {} = none."""
        return {}

    def stale_score(self, summary: dict, closure: dict) -> Optional[str]:
        """Why a run's stored score was made against another baseline than the suite's closure now holds (the page
        flags it: the run needs a rescore), or None.  The default: no kind-specific baseline to compare."""
        return None

    def check_outputs(self, out_dir: Path, result: Any, task, dataset_dir: Path, pylib: Optional[Path] = None) -> int:
        """`sbh check` on outputs beyond RESULT.json: 1 if the evaluation would fail the dataset, else 0."""
        return 0

    def build_card(self, run_dir: Path, suite_dir: Path, task, **kw) -> dict:
        """The run's report card.  Default: the POI card (sbh.report_card.build); other kinds build their own from
        the same shell helpers."""
        from ..report_card import build
        return build(run_dir, suite_dir, task, **kw)

    def post_stage(self):
        """The kind's TB grader stage module (options, prepare, child, finish, settings, summary_entry, file_fields,
        VERIFY_FILE; run per dataset by sbh/tbgrade.py), or None when the score reads the run records directly."""
        return None

    # ---- evaluator steps beside the runs and the score ----------------------------------------------------------------
    # Optional evaluator steps: a per-dataset post stage (`--phase post`) and per-run steps after scoring
    # (`sbh post-score`).  A kind without them adds no job.
    # sbatch resources of the `sbh post-score` job; None = no such job.
    post_score_shape: Optional[str] = None

    def post_shape(self, task) -> Optional[str]:
        """sbatch resources of one dataset's post-stage task, "-c <cpus> --mem <MB>M -t <wall>"; None = no post
        stage."""
        return None

    def post_dataset(self, run_dir: Path, suite_dir: Path, task, seed: str) -> None:
        """The per-dataset post stage on one dataset of a run; skips datasets the run did not complete."""
        return None

    def post_score(self, run_dir: Path, suite_dir: Path, task, label: str) -> List[str]:
        """Steps after a run is scored (`sbh post-score`; the report card is refreshed after).  Returns the required
        steps that failed (`sbh post-score` then exits 1); [] = none, or no steps."""
        return []

    def closure_post_score(self, task) -> bool:
        """Whether the closure's reference run also takes the post-score steps (its closure fields come from them)."""
        return False

    def closure_fields(self, run_dir: Path, suite_dir: Path, task) -> Dict[str, Any]:
        """Extra closure.json fields measured on the closure's reference run (updated by `sbh post-score`); {} =
        none."""
        return {}

    def closure_gate(self, suite_dir: Path, task, label: str) -> Optional[str]:
        """Why a run of ``label`` may not be scored on the suite (sbh.suite.closure_gate: a baseline of the closure
        the kind's score needs is missing or another rule's), or None."""
        return None

    def verification(self, run_dir: Path) -> Optional[Tuple[str, dict]]:
        """(file name, the run's model verification in the run-level file's form, {"per_seed": [record, ...],
        "n_seeds", "n_with_model"}) for a kind whose grader writes one record per dataset (the report card's checker
        block and per-dataset rows); None = the run-level file itself (sbh.report_card.verify_file)."""
        return None

    def variant_pieces(self, task, pieces: Tuple[str, ...]) -> Tuple[str, ...]:
        """The contract pieces appended to the task's ``pieces`` (sbh.contract.contract_pieces): the template's
        blocks of a variant of the kind's text, which cannot nest in the blocks they belong to; () = none."""
        return ()

    def contract_parts_file(self) -> Optional[Path]:
        """contract_parts.md in the kind's package ({{kind:<part>}} in a contract template); None = none."""
        import importlib
        mod = importlib.import_module(type(self).__module__)
        f = Path(mod.__file__).parent / "contract_parts.md"
        return f if f.exists() else None

    # ---- one RESULT.json POI entry (the core's readers of `pois` go through these) ---------------------------------
    def poi_problems(self, p: dict, task=None) -> List[str]:
        """Problems of one RESULT.json POI entry.  Default: value, stat, syst, total finite; total > 0 and stat (+)
        syst within 5 %."""
        from ..contract import symmetric_poi_problems
        return symmetric_poi_problems(p)

    def poi_result(self, p: dict, truth: Optional[float] = None):
        """A RESULT.json POI entry as sbh.metrics.PoiResult, whose sigma() the pulls, bias tests, leakage fits and
        report card divide by; ``truth``: the injected value, for a kind whose error depends on the side of it."""
        from ..metrics import PoiResult
        return PoiResult(p["value"], p.get("stat", 0.0), p.get("syst", 0.0), p.get("total", 0.0))

    def poi_sigma(self, p: dict, truth: Optional[float] = None) -> float:
        """The error a pull of one reported POI against ``truth`` divides by (participant side: `sbh check`)."""
        return p["total"]

    def poi_card(self, p: dict) -> Dict[str, Any]:
        """Report-card fields of one reported POI beside its value and total; {} = none."""
        return {}

    def poi_text(self, p: dict) -> str:
        """`sbh check`'s line for one reported POI (after its name)."""
        return f"{p['value']:.4g} ± {p['total']:.3g} (stat {p['stat']:.3g}, syst {p['syst']:.3g})"

    def reference_poi(self, v: dict, scored: Optional[dict] = None) -> dict:
        """What closure.json keeps of the reference's POI on one dataset (scores.json per_seed pois entry v), the
        reference entry case_loss receives; ``scored``: the POI's entry as the physics term scores it (score_inputs'
        records; None where there is none), for a kind whose baseline reads more than RESULT.json."""
        return {"value": v["value"], "sigma": (v["total"] if v.get("total") else
                                               (v.get("stat", 0.0) ** 2 + v.get("syst", 0.0) ** 2) ** 0.5)}

    def case_loss(self, p: Optional[dict], poi: str, truth: float, task, fallback: tuple,
                  ref: Optional[dict], ref_sigma: float) -> dict:
        """One physics case's losses (sbh.score._poi_cases): ``p`` the reported entry (None: no usable answer),
        ``fallback`` the task's (center, width) forecast, ``ref`` the reference's entry on the dataset
        (reference_poi; None: none), ``ref_sigma`` the scalar sigma_ref (0: none).  Returns {"good", "loss",
        "fb_loss", "ref_loss", "ref_source", "value", "sigma", "mw_pull", "fields"}, ref_loss None when there is no
        reference scale (the case is then excluded), mw_pull the case's signed pull (the bias term's,
        sbh.score.multiworld, which takes the case's width from its precision piece and ln sigma_ref = ref_loss - 1/2;
        +-inf allowed, None without an answer or where none exists).  Default: the core's rule
        (sbh.score.point_case_loss)."""
        from ..score import point_case_loss
        return point_case_loss(self.poi_result(p) if p is not None else None, truth, fallback, ref, ref_sigma)

    def result_problems(self, result: Any, task=None) -> List[str]:
        """RESULT.json problems beyond the core fields.  Default: the optional nuisance log (sbh/hep/nuisances.py)."""
        from ..hep.nuisances import check_nuisance_log
        return check_nuisance_log(result, task)

    def check_report(self, a, rec, dataset_dir: Path, work: Path, info: dict, env_prefix, task, level: str,
                     msg: str) -> Optional[int]:
        """`sbh check`'s report for a non-POI kind: the exit code; None = the core's POI report."""
        return None

    def closure_criteria(self, pub: dict, priv: dict, task, a, offsets=None, scored_calibration=None):
        """Kind-specific closure criteria: (criteria, max |pull|, extra closure.json fields); None = POI criteria."""
        return None

    def output_gate(self, out: Path, result: Any, task, self_check: bool = False,
                    pylib: Optional[Path] = None, dataset_dir: Optional[Path] = None) -> Tuple[List[str], List[str]]:
        """Run gate on the other outputs once RESULT.json passed: (problems, notes).  A problem fails the dataset
        (contract failure).  dataset_dir is the input the run read, for outputs that answer a per-dataset request."""
        return [], []

    # ---- scoring (evaluator side: sbh.score.score_run) ------------------------------------------------------------
    def score_run(self, run_dir: Path, suite_dir: Path, task, label: str, sigma_ref=None, cfg=None,
                  making_baseline: bool = False) -> Optional[dict]:
        """Whole-run scoring for a non-POI kind, returning the public summary; None = core POI scoring with the
        hooks below.  making_baseline: this is `sbh closure`'s run, which writes the suite baseline."""
        return None

    def score_inputs(self, run_dir: Path, records: List[dict], truths: Dict[str, dict], task) -> dict:
        """What the physics term scores: {"records", "source": "reported" or the kind's measurement name, "info":
        per-dataset measurement details or None}."""
        return {"records": records, "source": "reported", "info": None}

    def mark_provisional(self, v2: dict, v2_cases: List[dict], inputs: dict) -> None:
        """Flag physics cases scored on the reported value while the kind's measurement is pending."""

    def case_status(self, inputs: dict) -> Dict[Tuple[str, str], Tuple[str, str]]:
        """{(seed, poi): (status, reason)} of the physics cases the kind measures (score_inputs' inputs): "pending"
        (its measurement is not finished: a case without a usable answer is left out as provisional, never failed; a
        reason starting sbh.score.EVALUATOR_FAULT is the evaluator's own failure, and the kind gives no answer for it),
        "failed" (the model cannot give it: the reason its dataset failed), sbh.score.NO_MEASUREMENT (the contract
        counts the completed dataset without a measurement of it: charged max(F, 0), the dataset not failed); {} = every case
        as reported."""
        return {}

    def nuisance_metrics(self, records: List[dict], truths: Dict[str, dict], task, cfg) -> dict:
        """Diagnostics of the logged nuisances against the injected effects (no score effect)."""
        from ..hep.calibration import nuisance_metrics
        return nuisance_metrics(records, truths, task, cfg)

    def declared_yield(self, run_dir: Path, records: List[dict], truths: Dict[str, dict], task) -> Optional[dict]:
        """The declared-yield response diagnostic; None = not measured."""
        return None

    def background_flags(self, run_dir: Path, records: List[dict], task) -> Optional[dict]:
        """Report-only review flags over a run; None = none."""
        return None

    def measured_discovery(self, inputs: dict) -> Optional[dict]:
        """For a search task scored on the kind's measurement: {"claim": rec -> measured claim, "has": rec -> bool};
        None = the reported claim counts."""
        return None

    def score_terms(self, run_dir: Path, records: List[dict], truths: Dict[str, dict], task, inputs: dict,
                    v2: dict, leak: dict, yld: Optional[dict]) -> dict:
        """The weighted S_v3 terms: {"terms": [(name, weight, term)] in add order, "pending": reasons not final,
        "extra": v3 fields, "private": scores.json keys filled (sbh.score.PRIVATE_TERM_KEYS), "gof_cases",
        "calibration_response", and optionally "fallbacks": {kind: sbh.score.fallback_entry} of the fallbacks its
        grader and scorer took (the summary's v3 "fallbacks", sbh.score.fallback_block)}.

        Default: calibration term from the nuisance log only (no declared model, so no GOF or coverage).  Kept
        bit-identical (changing either is a scoring change): gof and completeness weights listed
        with no term, and missing GOF counts as pending, so the compute bonus never applies."""
        from ..hep.calibration import calibration_cases, calibration_term
        from ..score import V3_W_GOF, V3_W_SYST, cal_weight
        if getattr(task, "completeness_units", None):
            raise NotImplementedError(f"{task.name}: completeness_units need a declared model; the {self.name} kind "
                                      "has none")
        cal_cases = calibration_cases(records, truths, task, coverage=None)
        return {"terms": [("calibration", cal_weight(task), calibration_term(cal_cases)), ("gof", V3_W_GOF, None),
                          ("completeness", V3_W_SYST, None)],
                "pending": ["GOF pending (no grader record)"], "extra": {"gof_pending": True},
                "private": {"v3_calibration_cases": cal_cases, "v3_completeness_cases": [], "v3_gof_cases": None},
                "gof_cases": None, "calibration_response": {}}
