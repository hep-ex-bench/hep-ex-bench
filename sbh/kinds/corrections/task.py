"""The task base of the corrections kind: what a task whose answers are correction functions declares besides
sbh.task.Task -- scored function cases, their baseline, closure criteria, post stage shape, synthetic answers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ...task import Task


class CorrectionsTask(Task):
    kind = "corrections"
    result_type = "measurement"
    closure_pull_rms: Optional[Tuple[float, float]] = None
    closure_max_pull: Optional[float] = None
    closure_bias_rho: Optional[Dict[str, float]] = None
    post_cpus: int = 0
    post_memory_mb: int = 0
    closure_comparison: str = "z_peak"
    verify_budget_y: float = 0.0
    peak_budget_y: float = 0.0
    measurement_required: bool = True
    measurement_refit: bool = True
    baseline_accuracy: bool = False
    evaluation_population: str = ""

    def contract_values(self) -> Dict[str, str]:
        """Task.contract_values plus the post stage's verify budget (stated only where it gates), the MEASUREMENT.json
        opening (required or optional) and the evaluation population."""
        import textwrap
        y = float(self.verify_budget_y or 0)
        text = (f"The evaluator checks a dataset's measurements within a wall-clock budget of {y:g} times the "
                "reference analysis's own run time on a dataset, measured on the same machine: a measurement not "
                "checked within it counts as failing its check.") if y > 0 and self.measurement_required else ""
        pop = (f"A reported correction is evaluated at {self.evaluation_population}." if self.evaluation_population
               else "")
        opening = "{{kind.corrections:measurement_%s}}" % ("required" if self.measurement_required else "optional")
        return super().contract_values() | {"measurement_opening": opening,
            "measurement_budget": textwrap.fill(text, 115) + "\n\n" if text else "",
            "correction_population": "\n" + textwrap.fill(pop, 75) + "\n" if pop else ""}

    def function_cases(self, truth: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The scored cases of one dataset, each a dict with
          role      the case's label, the same across datasets (the aggregation unit),
          group     the role's aggregation group (groups weigh equally, roles within one equally),
          name      the requested correction it scores, flavour the one it is evaluated for,
          cluster   key of the role's cross-dataset statistics (datasets sharing it are correlated);
                    dataset_cluster (optional, default the dataset) the dataset's, for the completeness term,
          domain    {pt_min, abs_eta_max} as in request.json, where the contract requires it to evaluate
                    (outputs.py flavour_problems),
          points    {pt, eta, phi, charge} arrays inside it: where it is scored,
          truth     the injected correction at the points, in the comparison space,
          space     "value", or "signed_variance" (a width w compared as w|w|, value and band alike),
          tau       the floor (> 0) added in quadrature to every uncertainty, in the comparison space,
          truth_sigma (optional) the truth's own uncertainty at the points, added in quadrature too,
          fallback  (center, width): the "no answer" forecast,
          nominal   the no-effect value; null: whether the truth is that value at every point,
          fit_group (optional) key within which a role's response slope is fitted (fixed effects) for an effort
                    share, so a value depending only on the request cannot earn it; only the shipped rule has one
                    (sbh.score._effort), which no corrections task takes, so no score reads it,
          required_sources (optional) source classes an answer is expected to state (completeness term;
                    outputs.py SOURCE_CLASSES)."""
        raise NotImplementedError

    def function_views(self, kind: str, points: Dict[str, Any], domain: Dict[str, float]) -> Dict[str, list]:
        """The results page's views of a case of group `kind`, from its evaluation points (the arrays of
        function_cases' "points") and domain: {key: [{"name", "values" at the points, "edges"}, ...]}, one axis (a
        curve) or two (a map); {} none.  No score reads them."""
        return {}

    def function_baseline_id(self) -> Optional[str]:
        """What the per-case reference baselines depend on besides the reference's outputs (evaluation points, floors,
        whatever the truths' uncertainty comes from): closure.json records it, and scoring refuses a baseline made
        under another (rerun `sbh closure`)."""
        return None

    def synthetic(self):
        """Synthetic answers for the kind's scoring invariants (the kind's scoring check): a module with
        design() (the sealed datasets' fixed parameters), answer(truth, rng, mode) -> (corrections set, RESULT) for
        the kind's modes and its EXTRA_MODES ({label: mode}), and checks(cases, pub, check) for task-specific
        invariants.  None: not part of that check."""
        return None
