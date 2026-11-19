"""The task interface: the physics-specific part of a benchmark.

A task says how to draw a sealed truth from a prior, how to generate a dataset from it, which RESULT.json entries are
POIs, which logged nuisances it scores, and its participant-facing documents.  Freezing, execution, isolation,
scoring and packaging are shared.  Tasks live in tasks/<name>/__init__.py and expose a module-level ``TASK``.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class Task:
    name: str = "base"
    # >>> evaluator only
    # The benchmark's name (e.g. HiggsLepMass); None: the task's name.
    bench_name: Optional[str] = None
    # <<< evaluator only
    # The output kind (sbh/kinds/): what a run returns per dataset and how it is checked and scored.
    kind: str = "pois"
    pois: List[str] = []                 # RESULT.json POI names == truth keys
    # The signal-strength POI (0 on a search's null datasets: discovery, FDR, model refits); None = none.
    signal_strength_poi: Optional[str] = None
    # Truth key whose datasets share a draw: the unit the score's band resamples (sbh/band.py) and the clusters of the
    # report-only standard errors; None = each dataset its own.
    cluster_key: Optional[str] = "profile"
    result_type: str = "search"          # "search" | "measurement"
    requires_model: bool = False         # MODEL.json must exist and build as a pyhf workspace (GATE)
    # Contract version from which EVENTS.json (events behind MODEL.json's observed counts) is required; None = never.
    event_list_from: Optional[str] = None
    run_timeout_s: int = 600             # per-run wall clock budget (single source; CLI --timeout overrides)
    run_memory_mb: int = 6144            # RLIMIT_AS per run; 0 = none
    worker_memory_mb: int = 0            # per-dataset RESIDENT memory cap [MB], enforced by the eval's cgroup;
                                         # the runner only measures against it.  0 = not stated.
    run_cpus: int = 0                    # CPUs per run (single source); 0 = host's.
    # Analysis-specific nuisance names are <nuisance_prefix>_<label> (sbh/hep/names.py).
    nuisance_prefix: str = ""
    # Process names the task adds to the grammar's `<proc>` (sbh/hep/names.py PROCESSES; its contract lists them).
    processes: Tuple[str, ...] = ()
    # Pixi environment for a submission that ships no pixi.toml: .pixi/envs/<name>.
    base_env: str = "agentbase"
    # True = render the unified tasks/EVAL_CONTRACT.in.md with contract_values; False = the task's own template or
    # hand-written EVAL_CONTRACT.md (sbh.contract.render_contract).
    unified_contract: bool = False
    # the directory a submission may vendor installed packages in (sbh/snapshot.py hygiene): only a task whose
    # verifier runs offline on a fixed runtime and cannot build a pixi environment; None: dependencies go in
    # pixi.toml + pixi.lock and installed-package trees are refused
    vendor_dir: Optional[str] = None
    # The unified contract's pieces this task receives (the template's `<!-- if: <piece> -->` blocks); None = its
    # kind's (Kind.contract_pieces).  A task that composes others lists them all.
    contract_pieces: Optional[Tuple[str, ...]] = None
    # Entries `sbh stress` runs without --only (None: all not naming another task); `sbh pack`'s stub example.
    stress_submissions: Optional[Tuple[str, ...]] = None
    example_submission: str = "attacker_constant"
    # Attackers generated from the reference at stress time (sbh/attackgen.py): {entry: (family, run.sh flags,
    # description)}, family one of the kind's attack_families (or a core one).  An entry outside stress_submissions
    # is a review case (review_note says why).
    attackers: Dict[str, Tuple[str, str, str]] = {}
    # Hand-written attackers (tasks/<t>/attackers/, or another task's): {entry: family}.
    hand_attackers: Dict[str, str] = {}
    # Families of the kind this task neither generates nor hand-writes: {family: why}.
    attack_waivers: Dict[str, str] = {}
    review_note: str = ""

    def report_page(self) -> Dict[str, Any]:
        """The results page's facts about this task (the page's data collector -> v1.page):
          title, heading, lede   page name (plain), h1 and the paragraph under it (HTML),
          briefing_note          note beside the participant texts' commit,
          profiles               {profile: {column: value}} the datasets tab's detector-profile table,
          cases                  {POI: {name, symbol (HTML), text, unit, digits, unscored}} the run pages' names of a
                                 POI, its unit, its decimals (the response axes one fewer; an injected value shows its
                                 own up to four, a continuous draw two) and why a dataset's case may go unscored; a
                                 POI not listed shows its key, no unit, 2 decimals,
          datasets, seeds_intro  scoreboard dataset line and datasets-tab intro (HTML),
          seed_columns           [[report_truth field, header (HTML), format]] the datasets tab's table, format one of
                                 value (an injected value as stated), f<d>, pct<d> (d decimals), mono, "" (as it is);
                                 default: every field with a value, a POI under its name,
          seeds_note             the datasets tab's note on the scored cases (HTML); default: each POI on every
                                 dataset but those its `cases` entry's unscored names,
          labels                 {key: display name} for case groups, roles, flavours, score terms, curve variables and
                                 the datasets tab's default headers,
          method                 {fallback, scored, beyond, effects, cluster} a POI task's sentences in the method tab
                                 (HTML): what a failed dataset is charged as answering, which cases count, why the
                                 terms past the POIs matter, the calibration effects in parentheses, what datasets of
                                 one profile share; default: fallback, scored and effects from fallback_forecast, the
                                 POIs and the effects, beyond none, cluster a generic sentence,
          glossary               {key: hover text} added to or replacing docs/METRICS.md's glossary entries (its
                                 markup and {...} placeholders),
          tiers                  {"p0", "p1", "p2": what each prompt tier hands over}."""
        return {"title": f"HEP-Ex-Bench {self.name}", "heading": f"HEP-Ex-Bench {self.name}", "lede": ""}

    def report_truth(self, truth: Dict[str, Any]) -> Dict[str, Any]:
        """Per-dataset truth fields shown on the (evaluator-only) results page: the POIs and the profile."""
        return {**{p: truth.get(p) for p in self.pois}, "profile": truth.get("profile")}

    def quick_checks(self) -> List[Tuple[str, bool, str]]:
        """Fast task consistency checks run in smoke (checks/task_checks.py): [(check, passed, detail)]."""
        return []

    # POI -> (center, width): ignorance forecast from PARTICIPANT knowledge only (prompt envelope / shipped priors,
    # never the sealed design).  Evaluator-side scoring input; every POI the task's kind can leave without a
    # measurement needs one (sbh.score._poi_cases).
    fallback_forecast: Dict[str, tuple] = {}
    # Evaluator-side scoring inputs, {effect: nominal, target, prior}: the injected effects (prior = their rms about
    # nominal over the task's truths); Delta_cal value-scores those whose prior is wider than their target
    # (sbh.hep.calibration.value_scored), with tau = the target; empty = the term is off.
    calibration_scored: Dict[str, Dict[str, float]] = {}
    # The measurement's usual nuisances, each unit satisfied by any of its names.  Units with an effect of the
    # calibration spec are the calibration term's; Delta_syst covers the rest (sbh.hep.calibration.syst_units).
    completeness_units: List[Tuple[str, Tuple[str, ...]]] = []
    # Magnitude floors of the nuisance probes (sbh.score.nuisance_probes); an effect below its floor covers nothing.
    # "norm": relative yield change (normsys max|hi - 1|, |lo - 1|; histosys sum|x - nominal| / sum(nominal); lumi
    # its relative width), every effect but scale and resolution; "shape": total-variation distance of the
    # unit-normalised histosys shapes, for scale and resolution.  Parts on one sample add in quadrature.  Empty = none.
    nuisance_floors: Dict[str, float] = {}
    # Standard background processes (sbh.score.background_processes; REPORT-ONLY flag, never a score input):
    # [(label, patterns)], each a regex searched in the sample name lower-cased to letters and digits.
    # _merged: [((labels, ...), patterns)] for a sample carrying several processes (tried only if no single match);
    # _signal: names never counted as background; _min_share: below this share of nominal background = missing.
    standard_backgrounds: List[Tuple[str, Tuple[str, ...]]] = []
    standard_background_merged: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
    standard_background_signal: Tuple[str, ...] = ()
    standard_background_min_share: float = 0.0

    # POI -> (lower, upper) physical range, None = open (scan_likelihood kind: each interval lies inside it).
    # E.g. {"xi": (0.0, None)}.
    poi_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    # Working points a submission may name in an eff_<obj> log entry ("wp"): {object: names, default first}.
    # NAMES only (shipped in the participant stub); their truth meaning stays evaluator-side.
    working_points: Dict[str, Tuple[str, ...]] = {}

    def calibration_truth(self, truth: Dict[str, Any], log: Optional[dict] = None) -> Dict[str, float]:
        """Injected value of every effect, by logged nuisance name.  ``log``: the dataset's nuisance log, for a task
        whose truth follows declared working points; None = defaults."""
        return {k: float(truth[k]) for k in self.calibration_scored if k in truth}

    def calibration_spec(self, log: Optional[dict] = None) -> Dict[str, Dict[str, float]]:
        """The injected effects ({name: nominal, target, prior}) for one nuisance log, of which Delta_cal value-scores
        those whose prior is wider than their target (sbh.hep.calibration.scored_spec); calibration_scored unless the
        truth follows a declared working point."""
        return self.calibration_scored

    # ---- calibration response and compute (diagnostics) --
    # Detector effects are drawn independently of the POIs, so each POI residual is regressed on what an
    # UNcalibrated analysis would show: {poi: {effect: f(truth) -> residual}}.  Leakage k = 0 calibrated, 1 not.
    def leakage_probes(self) -> Dict[str, Dict[str, Any]]:
        return {}

    # Calibration effect whose injected value should scale MODEL.json's signal yield at POI = 1 (None: not probed).
    yield_response_effect: Optional[str] = None

    def event_level_truth(self, truth: Dict[str, Any]) -> Dict[str, float]:
        """One number per effect for the event-level regressions; per-object effects are combined here."""
        return self.calibration_truth(truth)

    # Reference wall seconds per evaluation dataset on the task's worker shape: the compute yardstick.
    # Delta_compute rewards faster and charges slower, up to run_timeout_s.  None: compute reported, not charged.
    compute_reference_wall_s: Optional[float] = None
    # Opt-in: time the reference on the public dataset in every evaluation (sbh/yardstick.py) and use Y x k as the
    # reference wall, k = sealed/public wall ratio measured once.  A failed measurement leaves no compute term.
    compute_yardstick_k: Optional[float] = None
    # Opt-in with compute_yardstick_k, dev only: time the reference on these sealed datasets instead (one replica per
    # node of the evaluation's allocation, Y = their median wall; sbh/yardstick.py), k = its median wall over all
    # sealed datasets / its median over these (one closure).  None: the public dataset.
    compute_yardstick_seeds: Optional[Tuple[str, ...]] = None
    # The reference's median wall per dataset on the hardware run_timeout_s is stated for.  With a yardstick, the run
    # time limit (and the compute term's clip) scale by max(1, Y x k / this) to the evaluation's hardware (cli
    # timeout_scale): a slower node type gets proportionally more time, a faster one never less than stated.
    run_timeout_reference_wall_s: Optional[float] = None
    # Compute bonus gate (sbh/score.py compute_gate): full speed reward while the rest of Delta_total <= rest0,
    # fading linearly to zero at rest0 + width; the slowness charge is never gated.  Set only for the TB verifier
    # (tb_score); None = rest0 from the suite's closure (sbh.score.compute_gate_params), width V3_COMP_GATE_WIDTH.
    compute_gate_rest0: Optional[float] = None
    compute_gate_width: Optional[float] = None
    # Opt-in: the headline's exponent (sbh/score.py score_gamma); None: the scorer's.
    score_gamma: Optional[float] = None
    # The bias term (sbh/score.py multiworld): per POI the mean case charge plus (z_star^2 + s_star^2 - 1) / (2 K),
    # z_star the answered datasets' inverse-variance combination of their misses; False leaves it out (the mean of the
    # cases).
    multiworld: bool = True
    # The physics rule (sbh/score.py score_rule): "standard" (a failed dataset fails the run, S = 0; the precision piece
    # floored per case; the bias term), or "tbs-shipped" for a task kept on the rule it shipped with (a failed case
    # charged its cap + 0.25, completeness, the whole case charge clipped at -(ln 1.25 + 1/2), no bias term).
    score_rule: str = "standard"
    # Set per evaluation by an evaluator that grades and times runs itself, never by a task.
    compute_yardstick: Optional[Dict[str, Any]] = None
    compute_gate_source: Optional[str] = None
    grader_strict: bool = False

    # ---- binned_likelihood kind (sbh/kinds/binned_likelihood/) ---------------
    # Score-from-fit: {POI: "fit" | "grid"}: score the evaluator's refit of the declared model or the declared grid;
    # empty = score the reported values.
    model_poi_sources: Dict[str, str] = {}
    # Range a "grid" POI's interval is evaluated in (e.g. (110, 140) GeV); None = the grid's own span.
    mass_range: Optional[Tuple[float, float]] = None
    # A "grid" POI's grid is linked to MODEL.json when its own workspace at the grid point nearest the reported mass
    # (within the local spacing, model.nearest_point) is MODEL.json's (the same workspace to model.identity_rel, at any
    # mass the grid declares; verify_grid._grid_link, model.grid_identity), else when its fitted point there
    # reproduces MODEL.json's fit; False: the fit comparison alone (a task kept on the grader it shipped with).
    grid_link_identical: bool = True
    # A "grid" POI's grid the grader refuses for its size (its mass_grid block's "over_cap"): True where the contract
    # makes any output file over its cap "a failed run on that dataset" (every case of the dataset failed), False where
    # it states that "a grid above its cap counts as no grid" (no mass measurement, the case charged max(F, 0)).
    grid_over_cap_fails: bool = False
    # Signal-region GOF window [lo, hi) of the model verification (main bins whose centre lies in it, from MODEL.json
    # `_meta.binning`): its deviance p is combined with the global one; the contract states it.  A task with a
    # declared model states it.
    gof_window: Optional[Tuple[float, float]] = None
    # Calibration credit withdrawn where the model shows it is not applied (sbh.score.calibration_gate):
    # {effect: (check, probe)}, check "leakage" with a probe name or "yield" with None.
    calibration_gates: Dict[str, Tuple[str, Optional[str]]] = {}
    # Opt-in: cap model verification per dataset at twice the reference's slowest (recorded in closure.json); a
    # dataset over the cap fails verification.  False: only the job's time limit.
    model_verify_cap: bool = False
    # Toys per dataset of the post stage's model verification (coverage and the scored GOF;
    # sbh/kinds/binned_likelihood/post.py); SBH_VERIFY_TOYS overrides them for a quicker check.
    model_verify_toys: int = 1000
    # Opt-in: score the signal-strength POI on the declared likelihood marginalised over the mass grid (the grader's
    # per-point conditional scans, prior dm dmu / sigma0_ref(m) on mu_support, charged -ln p(mu_true) against the
    # H of the reference's own density (its mean -ln p); sbh/kinds/binned_likelihood/scoring.py marginal_mu,
    # docs/METRICS.md).
    # False: the fixed-mass Gaussian point rule (model_mu).  Switching it re-closes the suite (mu_rule).
    mu_marginal_over_mass: bool = False
    # The evaluator-owned mu support D of that density, every workspace and the reference alike (equals
    # sbh.kinds.binned_likelihood.scoring.MU_SUPPORT, which the kind's checks assert; not imported here so the
    # participant bundle's task stub imports nothing evaluator-side).
    mu_support: Tuple[float, float] = (0.0, 4.0)
    # The task-shipped sigma_SM x BR x A(m) curve the grid's per-point templates are checked against (yield band,
    # slope, dead points; verify_grid.py, `sbh check`): a JSON {"masses": [...], "yield": [...]} relative to the
    # task's package dir (Task.path); None = no content checks on the grid.
    grid_yield_curve_file: Optional[str] = None

    def grid_yield_curve(self) -> Optional[Dict[str, List[float]]]:
        """{"masses": [GeV, ...], "yield": [...]} from grid_yield_curve_file (its other keys, e.g. provenance, are
        dropped), or None when the task ships none.  A file that is missing or not of that shape raises: a task that
        names a curve must ship it."""
        if not self.grid_yield_curve_file:
            return None
        import json
        p = self.path(self.grid_yield_curve_file)
        if not p.exists():
            raise FileNotFoundError(f"{self.name}: grid_yield_curve_file {self.grid_yield_curve_file!r} is not at {p}")
        d = json.loads(p.read_text())
        try:
            masses = [float(x) for x in d["masses"]]
            ys = [float(x) for x in d["yield"]]
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{p}: expected {{'masses': [...], 'yield': [...]}} ({type(e).__name__}: {e})") from e
        if len(masses) < 2 or len(masses) != len(ys) or any(y <= 0 for y in ys) \
                or any(b <= a for a, b in zip(masses, masses[1:])):
            raise ValueError(f"{p}: masses must increase and yields be positive, same length, >= 2 points")
        return {"masses": masses, "yield": ys}

    # ---- priors and truths -------------------------------------------------
    def default_priors(self) -> Dict[str, Any]:
        raise NotImplementedError

    def priors_without_response(self, priors: Dict[str, Any]) -> Dict[str, Any]:
        """Priors with optional dataset variations switched off."""
        return dict(priors)

    def public_truth(self, priors: Dict[str, Any]) -> Dict[str, Any]:
        """The disclosed truth of the public development seed (seed_000)."""
        raise NotImplementedError

    def draw_truth(self, rng: np.random.Generator, priors: Dict[str, Any],
                   fixed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """One sealed truth drawn from the prior, ``fixed`` entries (e.g. {"mH": 120}) overriding."""
        raise NotImplementedError

    def default_design(self, priors: Dict[str, Any], rng: np.random.Generator) -> Optional[List[Dict[str, Any]]]:
        """The sealed seeds' fixed entries when no design file is given (one dict per sealed seed; it sets their
        number), or None: n_seeds free draws."""
        return None

    # ---- data --------------------------------------------------------------
    def generate(self, truth: Dict[str, Any], rng: np.random.Generator,
                 dataset_dir: Path) -> None:
        """Write a dataset directory for this truth: NO truth information, same layout for every seed."""
        raise NotImplementedError

    def finish_dataset(self, dataset_dir: Path, suite_dir: Path) -> None:
        """Turn a generated dataset into the distributed one, in place (build_suite calls it after generate, e.g. a
        slim to the format the task's DATASET.md describes).  Default: ship it as generated."""
        return None
    # >>> evaluator only

    # The MC bundle every dataset's mc/ holds (sbh/bundle.py): the fingerprint of the current one (None: the task has
    # no bundle, nothing is checked) and the bundle's root entries kept for the evaluator, which datasets do not ship.
    mc_bundle_fingerprint: Optional[str] = None
    mc_bundle_evaluator_files: Tuple[str, ...] = ()

    def mc_bundle(self) -> Optional[Path]:
        """Where the task's current MC bundle is (from the site configuration), or None."""
        return None
    # <<< evaluator only

    # ---- scoring hooks -----------------------------------------------------
    def poi_applicable(self, poi: str, truth: Dict[str, Any]) -> bool:
        """Whether a POI is scorable on this seed (e.g. no mass without injected signal)."""
        return True

    def poi_scored_v2(self, poi: str, truth: Dict[str, Any]) -> bool:
        """(seed, POI) cases in the v2 log-score; tasks may widen applicability (e.g. mu on null seeds)."""
        return self.poi_applicable(poi, truth)

    # ---- files -------------------------------------------------------------
    # A variant of another task names it here: the files this task's directory lacks (DATASET.md, PROMPT.md,
    # prompts/, RUNTIME.md, reference/) are that task's.
    shares: Optional[str] = None

    @property
    def dir(self) -> Path:
        return ROOT / "tasks" / self.name

    def path(self, rel: str) -> Path:
        """tasks/<task>/<rel>, else the shared task's copy (`shares`)."""
        own = self.dir / rel
        return ROOT / "tasks" / self.shares / rel if self.shares and not own.exists() else own

    @property
    def reference_submission(self) -> Path:
        return self.path("reference")

    # The run label of the reference when `sbh stress` or `sbh eval` scores it as a submission (the closure's run is
    # always `reference`); stored runs carry it.
    reference_label: str = "reference"

    # Further reference analyses (opt-in): {label: {"name": page name, "budget": {...}, "note": ...}},
    # tasks/<task>/alt_references/<label> evaluated like any run and shown beside the task's reference, which stays
    # the yardstick (sigma_ref).  "budget" (optional, OWN_BUDGET_KEYS): its own larger run limits; its compute term is
    # then neutral and shown scaled to the task's CPUs.  evaluator_budget_x scales the evaluator's per-dataset checking
    # budgets.  "note" (optional): what the results page says beside it.
    further_references: Dict[str, Dict[str, Any]] = {}
    OWN_BUDGET_KEYS = {"run_cpus": int, "worker_memory_mb": int, "run_timeout_s": int, "run_memory_mb": int,
                       "evaluator_budget_x": float}

    def further_reference_names(self) -> Dict[str, str]:
        """{label: page name} of the further references; ValueError names a malformed entry."""
        out = {}
        for label, entry in self.further_references.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]:
                raise ValueError(f"{self.name}: further reference {label!r} is not {{'name': <page name>, "
                                 f"'budget': {{...}} (optional), 'note': '...' (optional)}}: {entry!r}")
            unknown = set(entry) - {"name", "budget", "note"}
            if unknown:
                raise ValueError(f"{self.name}: further reference {label!r} has unknown keys {sorted(unknown)}")
            if not isinstance(entry.get("note", ""), str):
                raise ValueError(f"{self.name}: further reference {label!r}'s note is not a string")
            self.own_budget(label)
            out[label] = entry["name"]
        return out

    def own_budget(self, label: Optional[str]) -> Dict[str, Any]:
        """A further reference's own budget; empty for every other label (task budgets apply)."""
        entry = self.further_references.get(label) if label is not None else None
        if entry is not None and not isinstance(entry, dict):
            raise ValueError(f"{self.name}: further reference {label!r} is not a dict: {entry!r}")
        own = (entry or {}).get("budget") or {}
        unknown = set(own) - set(self.OWN_BUDGET_KEYS)
        if unknown:
            raise ValueError(f"{self.name}: the budget of further reference {label} names unknown limits "
                             f"{sorted(unknown)}")
        return {k: self.OWN_BUDGET_KEYS[k](v) for k, v in own.items()}

    def docs(self) -> List[Path]:
        names = ("DATASET.md", "EVAL_CONTRACT.md", "PROMPT.md")
        return [self.path(n) for n in names if self.path(n).exists()]

    def prompt(self) -> str:
        return self.path("PROMPT.md").read_text()

    def extra_terms(self, run_dir: Path, records: List[dict], truths: Dict[str, dict], inputs: dict) -> list:
        """Task-specific weighted S_v3 terms after its kind's: [(name, weight, {"delta", "se", ...} or None)]."""
        return []

    def _gof_window_text(self) -> str:
        """The contract's GOF window ("105–140"); a task whose contract has a declared model must state one."""
        from .contract import contract_pieces
        from .kinds import get_kind
        if not self.gof_window:
            if get_kind(self).model_piece in contract_pieces(self):
                raise ValueError(f"{self.name}: a task with a declared model states its gof_window")
            return ""
        return "{:g}–{:g}".format(*self.gof_window)

    def contract_values(self) -> Dict[str, str]:
        """The {{value:<name>}} placeholders of the task's contract (sbh.contract.render_contract): budgets, nuisance
        prefix, signal POI, RUNTIME.md additions (one paragraph), observable, and optional kind-specific paragraphs
        (datasets_note, selfcheck_note, measurement_budget, correction_population, request_example, request_names,
        measurement_opening, model_verify_budget).  Tasks extend the dict."""
        import textwrap
        from .contract import contract_pieces
        from .hep.names import PROCESSES
        from .kinds import get_kind

        def gib(mb):
            return f"{mb / 1024:g}"

        def article(n: int) -> str:
            # "an" for numbers read with a vowel sound: 8.., 11 / 18 (and their thousands)
            s = str(int(n))
            return "an" if s.startswith("8") or (s[:2] in ("11", "18") and len(s) in (2, 4, 5)) else "a"
        extra = self.path("RUNTIME.md")
        runtime = extra.read_text(encoding="utf-8").strip() if extra.exists() else ""
        if "\n\n" in runtime:
            raise ValueError(f"{extra}: the runtime part sits inside one paragraph (no blank line)")
        # ends the binned_likelihood kind's "measured" contract part
        cap = ("The evaluator checks each dataset's declared model within a wall-clock budget of twice the longest "
               "time the same checks take on the reference analysis's own models: a model not checked within it "
               "counts as failed on that dataset.") if self.model_verify_cap else ""
        return {"cpus": f"{self.run_cpus} CPU{'' if self.run_cpus == 1 else 's'}", "memory_gib": gib(self.worker_memory_mb),
                "wall_s": str(self.run_timeout_s),
                "wall_article": article(self.run_timeout_s), "nuisance_prefix": self.nuisance_prefix,
                # the sentence of the hep part `more` listing the process names (the grammar's and the task's own)
                "processes": textwrap.fill("`<proc>` is one of " + ", ".join(f"`{p}`" for p in (
                    *PROCESSES, *self.processes)) + ".", 77, initial_indent="  ", subsequent_indent="  "),
                "signal_poi": str(self.signal_strength_poi or ""), "runtime": f"  {runtime}" if runtime else "",
                "observable": "the observable", "datasets_note": "", "selfcheck_note": "", "measurement_budget": "",
                "correction_population": "",
                # the model verification's GOF window the binning must cover (tasks with a declared model)
                "gof_window": self._gof_window_text(),
                # the hep part wp, naming the working points of an eff_<obj> entry (tasks with working points only)
                "wp": "{{hep:wp}}" if self.working_points else "\n",
                # how a declared model encodes the luminosity (the hep part lumi_model; tasks with a model only)
                "lumi_model": "{{hep:lumi_model}}" if get_kind(self).model_piece in contract_pieces(self) else "",
                # scan_likelihood kind: the POI table and the scanned POI (the task fills them)
                "interval_pois": "", "scan_poi": "",
                # corrections kind: the task's example request and the names a request can take (none by default)
                "request_example": "", "request_names": "",
                # required-declaration variant; CorrectionsTask overrides where optional
                "measurement_opening": "{{kind.corrections:measurement_required}}",
                "model_verify_budget": "\n" + textwrap.fill(cap, 75) + "\n" if cap else "",
                # the hygiene part's rule on installed packages (vendor_dir: a task that allows them says so)
                "vendoring": ("Installed packages are refused: no `pip install --target` output and no\n"
                              "`site-packages`, `*.dist-info` or `*.egg-info` tree anywhere in the\n"
                              "submission.  Declare what you need in `pixi.toml` + `pixi.lock` (below).")}

    def ship_fixup(self, root: Path) -> int:
        """Update the dataset copies under a pack's `root` to current shipping rules (suites built earlier); never
        the suite itself.  Returns the number of files changed; default none."""
        return 0


# >>> evaluator only
def task_names() -> List[str]:
    return sorted(p.parent.name for p in (ROOT / "tasks").glob("*/__init__.py"))


def submission_entries() -> Dict[str, Tuple[Path, Optional[str]]]:
    """{label: (directory, owning task)} of every runnable entry besides the tasks' references:
    tasks/<task>/attackers/<label>, tasks/<task>/alt_references/<label> and the generated attackers of every task (written under work/attackers/).
    Labels are unique across tasks, so a task's stress_submissions may name another task's entry."""
    from .attackgen import generated
    out: Dict[str, Tuple[Path, Optional[str]]] = {}

    def add(label, d, owner):
        if label in out:
            raise ValueError(f"submission label {label!r} is both {out[label][0]} and {d}")
        out[label] = (d, owner)
    for d in sorted([*(ROOT / "tasks").glob("*/attackers/*"), *(ROOT / "tasks").glob("*/alt_references/*")]):
        if d.is_dir() and (d / "run.sh").exists():
            add(d.name, d, d.parent.parent.name)
    for label, (task, d) in generated([load_task(n) for n in task_names()]).items():
        add(label, d, task.name)
    return out


def submission_dirs() -> Dict[str, Path]:
    """{label: directory} of submission_entries()."""
    return {k: d for k, (d, _) in submission_entries().items()}


def is_participant_stub(task: Task) -> bool:
    """Whether `task` is a participant task stub (write_participant_task_stub) rather than the evaluator's module: it
    has the contract metadata only, no generator, reference or attackers."""
    return type(task).__name__ == "_ParticipantTask"
# <<< evaluator only


# Task names a suite manifest or run record may carry for a task that has another name now: {stored: current}.
RENAMED_TASKS = {"h4l_ntuple": "h4l_tbs"}


def current_name(name: str) -> str:
    return RENAMED_TASKS.get(name, name)


def load_task(name: str) -> Task:
    mod = importlib.import_module(f"tasks.{current_name(name)}")
    return mod.TASK


def tbs_toml(task_name: str) -> Path:
    """tasks/<task>/tbs/tbs.toml: the task's Terminal-Bench-Science variant (may not exist)."""
    return ROOT / "tasks" / current_name(task_name) / "tbs" / "tbs.toml"


PARTICIPANT_TASK_STUB = '''"""Participant-side task metadata: exactly what `sbh check` needs.

The evaluator's full task module (generation, priors, injected-truth
machinery) is not part of the bundle.
"""
from sbh.task import Task


class _ParticipantTask(Task):
    name = {name!r}
    kind = {kind!r}
    pois = {pois!r}
    result_type = {result_type!r}
    requires_model = {requires_model!r}
    event_list_from = {event_list_from!r}
    nuisance_prefix = {nuisance_prefix!r}
    run_timeout_s = {run_timeout_s!r}
    run_memory_mb = {run_memory_mb!r}
    worker_memory_mb = {worker_memory_mb!r}
    run_cpus = {run_cpus!r}
    working_points = {working_points!r}
{extras}

TASK = _ParticipantTask()
'''


def write_participant_task_stub(task: Task, pkg_dir: Path, **limits) -> None:
    """Write a minimal tasks/<name>/ package for participant bundles: contract metadata only, never the
    generator, priors, yield constants or response profiles.  `limits` overrides run budgets where the evaluation
    differs from the task's (e.g. the TB verifier), so `sbh check` measures against what is actually applied."""
    unknown = set(limits) - {"run_memory_mb", "worker_memory_mb", "run_timeout_s"}
    if unknown:
        raise TypeError(f"write_participant_task_stub: unknown limit override(s) {sorted(unknown)}")
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text(PARTICIPANT_TASK_STUB.format(
        name=task.name, kind=str(getattr(task, "kind", "pois")), pois=list(task.pois),
        result_type=task.result_type,
        requires_model=bool(getattr(task, "requires_model", False)),
        event_list_from=getattr(task, "event_list_from", None),
        nuisance_prefix=str(getattr(task, "nuisance_prefix", "")),
        run_timeout_s=int(limits.get("run_timeout_s", getattr(task, "run_timeout_s", 600))),
        run_memory_mb=int(limits.get("run_memory_mb", getattr(task, "run_memory_mb", 6144))),
        worker_memory_mb=int(limits.get("worker_memory_mb", getattr(task, "worker_memory_mb", 0))),
        run_cpus=int(getattr(task, "run_cpus", 0)),
        working_points={str(o): tuple(str(n) for n in ns) for o, ns in (getattr(task, "working_points", None)
                                                                         or {}).items()},
        extras=_stub_extras(task)))
    _stub_yield_curve(task, pkg_dir)


# the keys of a task's grid yield curve file the participant copy keeps: the published curve itself, not how the
# evaluator derived it (its provenance keys name evaluator-side runs and the reference's yields)
STUB_CURVE_KEYS = ("schema", "description", "units", "mass_range_gev", "masses", "yield")


def _stub_yield_curve(task: Task, pkg_dir: Path) -> None:
    """Copy the task's grid yield curve (Task.grid_yield_curve_file) beside the stub, reduced to STUB_CURVE_KEYS, so
    the participant's `sbh check` runs the grid content checks against the curve the evaluator uses (the stub's
    grid_yield_curve() reads it from the stub's own package dir).  Nothing for a task without one."""
    rel = getattr(task, "grid_yield_curve_file", None)
    if not rel:
        return
    import json
    src = json.loads(task.path(rel).read_text())
    dst = Path(pkg_dir) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({k: src[k] for k in STUB_CURVE_KEYS if k in src}, indent=1) + "\n")


def _stub_extras(task: Task) -> str:
    """Stub lines for non-default processes, base_env, measurement_required, poi_bounds and model_verify_cap, and for a
    task scoring mu on the mass-marginal density / shipping a grid yield curve the gate, the curve file and the mass
    range `sbh check`'s grid checks read (none for a task that sets none of these: its stub is the plain one)."""
    lines = []
    if getattr(task, "processes", None):
        lines.append(f"    processes = {tuple(task.processes)!r}")
    if getattr(task, "poi_bounds", None):
        lines.append(f"    poi_bounds = {dict(task.poi_bounds)!r}")
    if getattr(task, "base_env", Task.base_env) != Task.base_env:
        lines.append(f"    base_env = {task.base_env!r}")
    if getattr(task, "measurement_required", True) is False:
        lines.append("    measurement_required = False")
    if getattr(task, "mu_marginal_over_mass", False):
        lines.append("    mu_marginal_over_mass = True")
    if getattr(task, "model_verify_cap", False):
        lines.append("    model_verify_cap = True")
    # the grid checks of `sbh check` need the task's yield curve (copied beside the stub, _stub_yield_curve) and its
    # evaluation mass range (contract text already)
    if getattr(task, "grid_yield_curve_file", None):
        lines.append(f"    grid_yield_curve_file = {task.grid_yield_curve_file!r}")
        if getattr(task, "mass_range", None):
            lines.append(f"    mass_range = {tuple(float(x) for x in task.mass_range)!r}")
    return "".join(line + "\n" for line in lines)
