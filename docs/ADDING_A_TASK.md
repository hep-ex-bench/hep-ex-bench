# Adding a task (or an output kind)

Three layers.  A new task touches only its own layer, plus a kind if it needs a new one.

| layer | where | what it holds |
|---|---|---|
| **core** | `sbh/` | Runner and sandbox, suites and sealed truth, scoring framework and its records, packaging. Knows no POI name and no output format beyond `RESULT.json`. Scoring: a case is (dataset, quantity, estimate, truth), log-scored against the reference's σ; cases are clipped, capped, aggregated; `score_v3` sums the weighted terms; then the compute gate, the error band, closure. |
| **output kind** | `sbh/kinds/<kind>/` | What a run returns per dataset and how the evaluator checks and scores it; shared by every task returning that kind of answer. |
| **task** | `tasks/<name>/` | The physics: truths and generator, POIs, budgets, calibration effects, prompts, its contract part, the reference. |

`sbh/hep/` holds the nuisance-log grammar, its checks and the calibration term, for any kind whose tasks log detector
calibrations.

## Kinds

| kind | module | returns | used by |
|---|---|---|---|
| `pois` | `sbh/kinds/pois/` | `RESULT.json` POIs with uncertainties, scored directly | `gauss_peak` |
| `binned_likelihood` | `sbh/kinds/binned_likelihood/` | `pois` plus `MODEL.json` (pyhf workspace, refitted), `MODEL_GRID/` (declared mass grid), `EVENTS.json` | `h4l`, `bump_model` |
| `corrections` | `sbh/kinds/corrections/` | correction functions (`corrections.json`, correctionlib) answering the dataset's `request.json`, each with uncertainty and declared measurement (`MEASUREMENT.json`, `.npz`); see `sbh/kinds/corrections/README.md` | `lep_calib`, `peak_calib` |
| `scan_likelihood` | `sbh/kinds/scan_likelihood/` | `RESULT.json` POIs as asymmetric intervals inside `Task.poi_bounds` (the first POI's 68 % set may be several pieces, `intervals`), `MODEL.json` (a pyhf workspace whose samples may scale with monomials in the POIs, `_meta.poi_scaling`; never fitted with pyhf alone, which would ignore the scaling) and `EVENTS.json`; the evaluator scans the first POI's profile over its range and fits the others on the model's JAX twin | `h4l_width`, `width_toy` |

A task names its kind in `Task.kind`; `sbh.kinds.get_kind(task)` returns it; `KINDS` in `sbh/kinds/__init__.py` is the
registry.

### The Kind interface (`sbh/kinds/base.py`)

| hook | called by | default |
|---|---|---|
| `poi_problems`, `poi_result`, `poi_sigma`, `poi_text`, `poi_card` | the core's readers of one RESULT.json POI entry: contract check, the scorer's PoiResult (pulls, bias, leakage), `sbh check`'s pull and line, the report card | the symmetric value / stat / syst / total form |
| `reference_poi`, `case_loss` | `sbh closure` (what closure.json keeps of the reference's POI), the physics term's per-case loss, baseline and signed pull (`sbh.score._poi_cases`; the pull feeds the bias term, with the width of the loss's precision piece) | the Gaussian ln σ + pull²/2 against ln σ_ref + ½, pull (value − truth)/σ |
| `output_names` | nobody yet (documentation; hygiene reads `forbidden_outputs`) | `("RESULT.json",)` |
| `result_max_bytes` | runner, before reading RESULT.json | `None`: no cap |
| `reports_pois` | core RESULT.json check: is `pois` required | True |
| `reports_s_phys` | scorer: does the summary carry the v1 S_phys (linear pulls), which the stress invariant also orders by | True (scan_likelihood: False) |
| `result_problems(result, task)` | core RESULT.json check | HEP nuisance-log check (corrections: none) |
| `forbidden_outputs` | freeze: extra files a submission may not ship | none |
| `output_gate(out, result, task, self_check, pylib, dataset_dir)` | runner, after RESULT.json passes | nothing |
| `check_outputs(out_dir, result, task, dataset_dir, pylib)` | `sbh check` (ships to participants) | nothing |
| `check_report(...)` | `sbh check`'s report, for a kind with its own (corrections) | `None`: POI report |
| `post_shape(task)`, `post_dataset(run_dir, suite_dir, task, seed)` | dev chain's per-dataset post stage between runs and score (`sbh <cmd> --phase post`) | `None` |
| `post_score_shape`, `post_score(run_dir, suite_dir, task, label)` | dev chain's job after a run's score (`sbh post-score`); `post_score` returns the required steps that failed (the command then exits 1); `sbh stress --recheck` then checks the stress invariant on the rescored runs (the smoke's bump_model lane runs it) | `None`, `[]` |
| `closure_post_score(task)`, `closure_fields(run_dir, suite_dir, task)` | closure's job after scoring the reference (`sbh post-score --label reference`) and the closure.json fields it measures (`sbh closure`; updated by that job); binned_likelihood: model-verify cap | False, `{}` |
| `score_run(run_dir, suite_dir, task, label, ...)` | `score_run`: a kind scoring the run itself via core pieces (`v2_aggregate`, `score_v3`, `write_scores`); on the closure's run (`making_baseline`) it writes the per-dataset baselines it scored the reference against as private `reference_pois`, which `sbh closure` records | `None`: generic path |
| `closure_gate(suite_dir, task, label)` | `sbh.suite.closure_gate` (every scoring command): why a run may not be scored on the suite; scan_likelihood: the closure holds the grader's baseline of the reference on every dataset | `None` |
| `score_inputs`, `mark_provisional`, `case_status` | `score_run`: what the physics term scores; which cases are pending (a case without a usable answer left out, provisional; on an evaluator fault, `sbh.score.EVALUATOR_FAULT`, left out whatever was reported), failed by the kind's measurement (the model's: its dataset fails the run), or have no measurement on a completed dataset where the contract says so (`sbh.score.NO_MEASUREMENT`: charged max(F, 0), the dataset not failed; each such POI needs a `Task.fallback_forecast`) | reported POIs; none |
| `score_terms(...)` | `score_run`: the kind's weighted S_v3 terms, and the fallbacks its grader and scorer took (`"fallbacks"`: {kind: `sbh.score.fallback_entry`}, counted in `v3.fallbacks`, never charged; the core adds its own, a case with no measurement among them) | calibration term from the log; no fallbacks |
| `declared_yield`, `background_flags`, `measured_discovery`, `nuisance_metrics` | `score_run` diagnostics | none / HEP diagnostics |
| `scoreboard_rows(summaries)` | `sbh.score.format_tables` | none (POI table) |
| `closure_criteria(pub, priv, task, a, ...)` | `sbh closure` | `None`: POI criteria |
| `contract_parts_file()` | contract renderer | `contract_parts.md` in the kind's package |
| `model_piece`, `variant_pieces(task, pieces)` | contract renderer (`sbh.contract.contract_pieces`), `Task.contract_values`, `get_kind` (`requires_model`), the results page's data collector: the declared model's contract piece and the variant pieces appended to the task's | `None` (no declared model), `()` |
`sbh/kinds/base.py` also defines the hooks of the evaluator's own records, the report card and the results page
(`verification`, `build_card`, `card_files`, `page_files`, `report_extras`, `stale_score`), whose defaults a new kind
may keep.

**Dev-chain steps**, the kind's work around a run's score:

- `post_dataset`: one task per completed dataset (`post_shape`), between runs and score.  Corrections: measurement
  checks and the closure comparison (Z peak or jet balance; `sbh/kinds/corrections/post.py`), within the task's
  budget.
- `post_score`: one job per run (`post_score_shape`).  binned_likelihood: model validation and rescore; with
  `model_verify_cap`, each dataset's validation within the closure-measured cap (the closure's own run included,
  `closure_post_score`).

A kind without them adds no job.

Evaluator-only modules (scorers, graders, refits) go into `sbh/harbor_pack.EVALUATOR_ONLY` (paths relative to `sbh/`)
and never ship in participant bundles.  `sbh check` must stay importable without them.  What may ship is pinned:
`sbh/harbor_pack.PARTICIPANT_SBH` lists every `sbh/` file a bundle may hold, and a bundle holding anything else fails
to pack (`checks/bundle_check.py` builds every task's bundle in smoke), so a module reaches participants only by an
edit to that list.

## The Task interface (`sbh/task.py`)

Subclass `sbh.task.Task`; its class attributes carry defaults and meaning.  Main ones:

- **Identity:** `name`, `kind`, `pois`, `result_type` (`search` or `measurement`); `signal_strength_poi` (null
  datasets of a search, a kind's refits); `cluster_key` (truth key whose datasets share a draw: the unit the
  score's band resamples, METRICS.md "Error bands").
- **Budgets:** `run_timeout_s`, `run_cpus`, `worker_memory_mb`, `run_memory_mb`; the contract's Budgets paragraph is
  rendered from them.
- **Physical range:** `poi_bounds` ({POI: (lower, upper)}, None = open; the scan_likelihood kind's intervals live inside it).
- **Process names:** `processes`, the names a task adds to the nuisance grammar's `<proc>` (`QCDscale_<proc>`,
  `cross_section_<proc>`; `sbh/hep/names.py`); its contract's list and its participant stub carry them (h4l_width:
  `EWVV`).
- **Truth and data:** `default_priors`, `draw_truth`, `public_truth`, `generate(truth, rng, dataset_dir)`;
  `poi_applicable` / `poi_scored_v2` (which (dataset, POI) cases are scored).
- **Scoring inputs:**
  - `fallback_forecast`: the "no answer" forecast, from what participants are told: its charge F sets each case's
    cap C = max(F, 4) and the charge of a case with no measurement, max(F, 0).  Required for every POI the task's
    kind can leave without a measurement (`sbh.score.NO_MEASUREMENT`: a mass without a usable grid, a mass-marginal
    μ without a grid, a correction not reported): a case with no measurement on a POI without one stops the score as
    a configuration error, never charged a default.
  - `compute_reference_wall_s`, or `compute_yardstick_k`: the reference timed in each evaluation (`--phase
    yardstick`, before the runs); wall × k is the compute term's reference wall.  Such a task states its worker spec
    (`run_cpus`, `worker_memory_mb`).
  - `calibration_scored` / `calibration_truth` / `calibration_spec` / `working_points`: the HEP calibration term,
    per injected nuisance its nominal, its target precision (the reference's typical) and its prior (the rms of
    the injected values about the nominal: the width of its no-answer forecast, the nominal value; an unlogged
    nuisance costs max(F, 0)).  A nuisance is value-scored only where its prior is wider than its target
    (`sbh.hep.calibration.value_scored`), with τ = the target as the σ_ref of its per-case charge: a measurement at
    the target then says more than the prior, and logging the nominal value never beats measuring.  One whose prior is
    no wider than its target is held out (`held_out`: a measurement at the target says less than the prior, and a
    constant logged without the data would beat it); give a target the reference reaches and that is narrower than
    the prior for every effect the term should score.
    Every valid logged entry is scored on its value whether or not the declared model carries the effect; the model
    enters only through the calibration gate.  docs/METRICS.md "Δ_cal".
  - `completeness_units`, `nuisance_floors`: binned_likelihood completeness term.
  - `model_verify_cap` (binned_likelihood, scan_likelihood; off): dev-chain model verification per dataset within 2× the closure
    reference's slowest; the contract then states it (`contract_values` `model_verify_budget`).
  - `gof_window` (binned_likelihood; required with a declared model): the model verification's signal-region GOF
    window [lo, hi), which the contract asks the binning to cover (`contract_values` `gof_window`).
  - `model_verify_toys` (binned_likelihood; 1000): toys per dataset of the post stage's model verification (coverage
    and the scored GOF); `SBH_VERIFY_TOYS` overrides them for a quicker check.
  - `score_gamma`: exponent in S_quality = exp(−γ Δ); unset = 2 (luminosity-equivalent ratio), 1 = precision ratio
    σ_ref/σ.
  - `multiworld` (True): the bias term of the physics charge (per POI the mean case charge plus
    (z_star² + s_star² − 1)/(2K), z_star the answered datasets' inverse-variance combination of their misses); False keeps the
    mean of the cases; docs/METRICS.md "Per-case loss".
  - `score_rule` ("standard"): the physics rule; "tbs-shipped" keeps a frozen task on the rule it shipped with (a
    failed case charged C + ¼ and completeness, the effort share, no failure of the run, its calibration rule, and
    its grader as shipped: `verify.SHIPPED`), which no new task takes.
  - `grid_link_identical` (True; a task with a "grid" POI): the grid is linked to MODEL.json when its own workspace
    at the grid point nearest the reported mass (within the local spacing) is MODEL.json's, else by the fit
    comparison; False keeps a frozen task on the fit comparison alone (docs/METRICS.md, "scored m_H (grid)").
  - `grid_over_cap_fails` (False; a task with a "grid" POI): what a declared grid over its size cap is, as the task's
    contract states it: True a failed run on that dataset ("a larger file counts as a failed run"), False no grid
    (no mass measurement, charged max(F, 0)).
  - `measurement_required`, `measurement_refit` (corrections): whether the declared measurement gates the
    corrections (else checked and shown only) and whether the post stage refits it (GOF scored);
    `evaluation_population`: where corrections are evaluated, stated in the contract's corrections.json section.
    lep_calib: optional, not refitted.  See `sbh/kinds/corrections/README.md`.
  - `further_references`: see *Files of a task*; `own_budget(label)` gives a label's budget, checked by
    `further_reference_names()` (`checks/task_checks.py`).
- **Extra terms:** `extra_terms(run_dir, records, truths, inputs)`, appended after the kind's terms.
- **Contract:** `unified_contract`; `contract_values()` fills the contract's `{{value:..}}` placeholders.
- **Evaluator-only settings** (attributes the task module may carry, not declared on `sbh.task.Task`, which ships to
  participants): `tripwire_spec` (the point-mass tripwire, `sbh/tripwire.py`: the POI, the points its sealed truths sit
  on, published values, the null and every rule's setting); `sealed_value_steps` ({POI: step} of the sealed truths'
  design grid) and `sealed_value_grid(priors)` ({POI: values} a truth can take), which the sealed-safe results page
  uses to withhold guessable values.  The toys show all three.

## Files of a task

```
tasks/<name>/__init__.py          the Task subclass, exposed as TASK
tasks/<name>/EVAL_CONTRACT.md     rendered contract (`pixi run contracts`)
tasks/<name>/RUNTIME.md           optional: what the task's base environment adds to the base runtime
tasks/<name>/DATASET.md           input mechanics only
tasks/<name>/PROMPT.md, prompts/  prompt tiers: PROMPT.md the baseline, prompts/<tier>.md the others
tasks/<name>/reference/           the reference pipeline (Task.reference_submission; run label Task.reference_label)
tasks/<name>/attackers/<label>/   hand-written attackers (Task.hand_attackers; the others are generated at stress time)
tasks/<name>/alt_references/<label>/  further reference analyses (Task.further_references)
tasks/<name>/truth_grids/         the truths of each sealed dataset (`sbh build-suite --design`; answer key, never shipped)
```

Attackers: the output kind names the attack families its scoring must catch (`Kind.attack_families`; `symlink`,
output-file hygiene, is the core's and applies to every kind).  A task gives its reference a command-line switch per
family and maps them in `Task.attackers` (`{entry: (family, flags, description)}`); `sbh/attackgen.py` builds each
entry at stress time under `work/attackers/<entry>/` as the reference plus a run.sh that adds the flags.  Attackers
that do not run the reference (a constant answer) are hand-written under `tasks/<name>/attackers/` and listed in
`Task.hand_attackers`; a family the task does not attack yet goes in `Task.attack_waivers` with the reason.
`checks/attackers_check.py` (smoke) holds every task to this.

Everything about a task lives in its directory.  Entry labels are unique across tasks, so a task's
`stress_submissions` may name another task's entry.

Further references: `{label: {"name": page name, "budget": own run budget (optional), "note": what the results page
says beside it (optional)}}`.  Each is evaluated as a run
under its label (`sbh eval --label <label>`) and shown beside the reference, which stays the yardstick.  An own budget
applies to its runs and makes its compute term neutral (CPU-scaled walls shown); `evaluator_budget_x` scales the
evaluator's checks of its answers (that reference's alone: a run's comparison that missed its budget is made again
with `measurement_verify --peak-budget-x`, docs/METRICS.md "Function outputs").

## The contract

`EVAL_CONTRACT.md` is rendered from a template plus included parts:

- `{{core:<part>}}`: `sbh/contract_parts.md` (submission form, outputs, hygiene, runtime, budgets, self-check memory).
- `{{hep:<part>}}`: `sbh/hep/contract_parts.md` (nuisance log, names, rules).
- `{{kind:<part>}}`: the task's kind's `contract_parts.md`; `{{kind.<name>:<part>}}`: the named kind's.
- `{{value:<name>}}`: `Task.contract_values()`; a task keeps a longer text of its own as a part of
  `tasks/<task>/contract_parts.md` and returns it there (the calibration benchmarks: `request_example`,
  `request_names`; h4l_width: `wp`, the working points its data carry).

Templates:

- `unified_contract = True` (h4l, lep_calib): `tasks/EVAL_CONTRACT.in.md`, composed per task from pieces: the
  shared submission form, runtime, budgets and self-check, plus the blocks `<!-- if: <piece> -->` of the task's
  `contract_pieces` (default its kind's `Kind.contract_pieces`: binned_likelihood `pois`, `model`, `nuisances`;
  corrections `corrections`), then its kind's variant pieces (`Kind.variant_pieces`: binned_likelihood adds
  `mu_point` or `mu_marginal`, its signal-strength rule's text).  A task with several kinds of result lists every
  piece.  A task whose pieces hold its kind's declared-model piece (`Kind.model_piece`: binned_likelihood `model`)
  states its GOF window and gets the luminosity encoding.  Kept headings
  (`{{sec:<id>}}`) are numbered in order and `{{ref:<id>}}` renders as `§<n>`, so a task's sections are numbered
  from 1 and a reference to a section the task does not receive fails the render.  A task with a statistical
  model gets the whole nuisance section, every effect's encoding, so the contract does not suggest which
  systematics apply.  The template fixes table names for all tasks (`sbh.contract.UNIFIED_VALUES`: signal strength
  `mu`, analysis prefix `ana_`); a task adds requirements through its values (h4l: the observable MODEL.json bins,
  evaluation dataset size).
- Otherwise the task's own `EVAL_CONTRACT.in.md` (`peak_calib`), or a hand-written `EVAL_CONTRACT.md` (`gauss_peak`).

The rendered file is committed and read by every reader and packer; smoke fails when it is stale.  Edit the template
or parts, never the rendered file.

## Inputs

The code that prepares a task's input data lives beside the tasks, under `inputs/<source>/`: one directory per
input source (a production, an open-data release), holding its preparation code, its file catalogs and nothing
task-specific.  One source may feed several tasks, and a task may read more than one source; the task module imports
what it uses from there.

A task whose datasets ship a prepared simulation (an MC bundle, every dataset's `mc/`) says where its current bundle
is (`Task.mc_bundle()`, from the site configuration) and records that bundle's fingerprint
(`Task.mc_bundle_fingerprint`; `sbh/bundle.py`, `python -m sbh bundle-fingerprint --task <name>`), naming in
`Task.mc_bundle_evaluator_files` any root entry the bundle keeps for the evaluator only.  Building a suite, packing a
participant bundle and preparing an evaluation refuse another bundle, so a stale copy, or a suite built on one, is
never used in its place: after a rebuild, record the new fingerprint and rebuild the suites.

## A variant of a task

A variant subclasses the task, sets `Task.shares = "<task>"` and keeps in its own directory only what differs: its
`EVAL_CONTRACT.md` (it names the task) and whatever data it pins.  `Task.path(rel)` resolves the rest
(DATASET.md, PROMPT.md, prompts/, RUNTIME.md, reference/) from the shared task's directory.

## One suite, several tasks

A task may score datasets another task built (e.g. another quantity measured on h4l's seeds).  List both in the suite
manifest's `"tasks"` and score with `--task` (`sbh eval/closure/stress --task <name>`).  Truths are the generator's;
each task scores only its own POIs (`Task.pois`, `poi_scored_v2`).

## Gates (cheap first)

- **`pixi run check`** (~10 s): the quick tier — contract render, lock files, bundles, attackers, task consistency.
  After any edit.
- **`pixi run smoke`** (an hour or so, steps in parallel): the push gate — the quick tier plus the grader batteries
  and each toy's suite build, closure and stress, a lane each.

A new output kind lists the per-dataset outputs its scorer reads in `Kind.score_files`, so that a run rescores from
them.
Its failures are the contract's (docs/METRICS.md "Failures"): what the contract calls a
failed run fails the dataset; a case the contract counts on a completed dataset without a measurement has none
(`sbh.score.NO_MEASUREMENT`, charged max(F, 0); the task gives each such POI a `fallback_forecast`), and nothing
else does; an error of the evaluator's own code leaves the case
pending (`sbh.score.EVALUATOR_FAULT`), never the run's.  A grader of its own marks its errors and stamps its records
(`evaluator_error`, `errors_marked` through binned_likelihood's `verify.mark_errors`), and its post stage writes a
placeholder before it grades a dataset (`verify.unfinished_record`, "grading did not finish": a process that dies
leaves an evaluator fault, never a dataset without a record), counts its attempts per dataset and verifies a dataset
once more before the score where an evaluator error has not recurred (`grader_attempts`, `verify.count_attempt`,
`verify.to_retry`), as scan_likelihood's does, so that a fault it reproduces shows the run unranked from its first
score.  It skips that retry where every place a dataset faults lies where the same fault sits on more than half of
the run's completed datasets (`verify.is_systemic`, `verify.systemic_places`, `verify.mark_systemic`: a systemic
fault, the run provisional), grading the datasets one process is given all once before any retry (a post module's
`batch`) and offering `settle_systemic`, which the score phase runs over the run's final records; its scorer splits
reproduced from systemic faults over those records (binned_likelihood `scoring.fault_split`), whatever the grading
order.  It reads a record it wrote before stamping as no evaluator error among its attempts (`verify.attempts_of`),
and it refuses at the model's boundary what pyhf's fit refuses or no toy can draw, as the model's failure: an init
outside its bounds (`jax_engine.init_outside_bounds`), a constraint width that is not a positive finite number
(`model.constraint_width_problem`) and, before any toy, a rate beyond numpy's Poisson limit
(`verify_fit.check_toy_rates`, `POISSON_RATE_MAX`).  Every secondary procedure its grader or scorer can take is a
fallback, counted through `score_terms` "fallbacks" or recorded, and listed in docs/METRICS.md "Fallbacks".

## Known couplings (remove when a second task needs it)

- **binned_likelihood assumes a signal strength and a mass POI `mH`.**  The verifier's RESULT reader and the mass grid
  (`MODEL_GRID/mH_<value>.json`) use `mH`; the unified contract names `<poi>_<value>.json` for any mass POI of its §2,
  but only `mH` is read.  Claims, Z_obs and consistency checks read `Task.signal_strength_poi`.
- **Two `pois`-kind quirks kept for bit-identity** (changing either is a scoring change): gof and completeness weights
  stay listed with no term; a missing GOF counts as pending, so the compute bonus never applies.
- **Vendored `sbh/metrics.py`** keeps its null rule on `truth["mu"]`; it feeds only the S_phys diagnostic.
