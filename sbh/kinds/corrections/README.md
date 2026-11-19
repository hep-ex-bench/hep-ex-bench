# The `corrections` kind

A run of a corrections task returns correction functions: `corrections.json`, a correctionlib CorrectionSet answering
the dataset's `request.json`, each correction with its uncertainty and its declared measurement (`MEASUREMENT.json`,
`MEASUREMENT.npz`).  The contract's corrections.json, request.json and MEASUREMENT.json sections are this kind's parts
(`contract_parts.md`).  Used by the calibration benchmarks and the toy `peak_calib`.

## Modules

Participant-facing (in every bundle; `sbh check` and the runner's gate):

| module | holds |
|---|---|
| `__init__.py` | `CorrectionsKind`, `KIND` (registered in `sbh.kinds.KINDS`) |
| `outputs.py` | reading, checking and evaluating `corrections.json` (correctionlib imported lazily) and RESULT.json's covariances |
| `measurement.py` | the declared measurement's checks: structure, recount, the pairs against the dataset |
| `check.py` | the runner's output gate and `sbh check`'s report |
| `task.py` | `CorrectionsTask`, the task base |
| `contract_parts.md` | contract sections 5–7: corrections as functions, the request, the declared measurement |

Evaluator-only (`sbh/harbor_pack.py` `EVALUATOR_ONLY`; imported only where used):

| module | holds |
|---|---|
| `scoring.py` | the function cases and the run's score |
| `refit.py` | the checks of a declared measurement and, under `measurement_refit`, its refit (GOF; the refit comparison, display only) |
| `comparison.py` | which closure comparison a task uses (`closure_comparison`) and what records name it by |
| `peak.py` | the closure comparisons' common core (`Comparison`, `compare`, the fallback loss, the candidate cache) and the Z-peak closure |
| `balance.py` | the jet balance closure (Z(μμ)+jet balance and dijet asymmetry) |
| `closure.py` | the closure criteria and the baselines `closure.json` keeps |
| `post.py` | the post stage (below), writing `runs/<seed>/measurement_verify.json`; command line `measurement_verify.py` |

## The Kind hooks it sets (`sbh/kinds/base.py`; every default is the POI kinds' behaviour)

| hook | here |
|---|---|
| `output_names` | `RESULT.json`, `corrections.json`, `MEASUREMENT.json`, `MEASUREMENT.npz` |
| `result_max_bytes` | 20 MB: RESULT.json may carry covariances |
| `reports_pois` | False: `result_problems` checks `covariances` and `joint_covariances`; a nuisance log is recorded unchecked (see [The contract](#the-contract)) |
| `output_gate(..., dataset_dir)` | `corrections.json` against the request: a malformed correction is a note (not reported), not a failed run; without correctionlib only the self-check proceeds (NOT VALIDATED HERE) |
| `check_report` | `sbh check`: each reported correction's structure, inputs, uncertainty and evaluation on a grid over its domain; its measurement against the development dataset; `--determinism` compares two runs |
| `forbidden_outputs` | `corrections.json`: a submission may not ship one (refused at freeze, as RESULT.json) |
| `closure_criteria` | per group the pull-RMS band, the largest mean pull, the response slope and the bias gate (`closure.py`) |
| `score_run` | the kind scores the run itself (`scoring.py`) |
| `post_shape`, `post_dataset` | the per-dataset post stage (`sbh <cmd> --phase post`) in `post_cpus` / `post_memory_mb`; a task stating no shape or budget is refused |
| `scoreboard_rows` | the scoreboard's function table, per run and role (`scoring.py`) |

**Review: scoring through `score_run`.**  A corrections run's cases are functions evaluated at the task's evaluation
points (value, σ, sources) aggregated over groups and roles, not POIs, so the kind owns the orchestration and reuses
the core's pieces: `v2_aggregate` for the physics term; `score_v3` with completeness (0.10), GOF (0.30, under
`measurement_refit`) and the closure comparison (0.5: the Z peak or the jet balance); the compute gate (the closure's
reference is the compute yardstick
itself and scores its own compute term 0); `write_scores`.  No truthfulness term: the physics term's log loss, a
proper score, prices a stated uncertainty both ways.  The POI kinds keep `score_inputs` / `score_terms`;
`score_run` returns None for them.

**Review: the post stage** (`post.py`), between the runs and the score:

- One job per completed dataset in `post_cpus` / `post_memory_mb`: the measurement checks, the refit and its GOF
  toys under `measurement_refit`, the closure comparison; writes `runs/<seed>/measurement_verify.json` with rule stamps
  (a stale stamp is pending).
- Launched by `post_dataset` behind `sbh <cmd> --phase post`, or by hand via `measurement_verify.py`.
- Process pools (refits, dataset readers) run through `measurement.pool_imap`: a worker killed by the memory limit
  has its items rerun in fewer processes instead of hanging the stage.
- Each step runs within its budget (`verify_budget_y`, `peak_budget_y` × the compute yardstick's Y·k;
  `sbh/budget.py`): measurements verified in time keep their results, the others fail as not verified; the closure
  comparison then takes them at their forecast, and a flavour not compared in time gives the dataset no closure case
  and leaves the score provisional until the comparison is made again at `PEAK_RERUN_X` or more times that budget
  (the fallback `peak_not_in_budget`; docs/METRICS.md "Function outputs").
- The file keeps the comparison's attempts on the dataset under its comparison rule (`peak_attempts`: each one's
  budget in seconds and whether it ran out; `post.peak_attempts`, carried over when the stage rewrites the file under
  the same rule).  `measurement_verify.py --peak-budget-x [X]` (`rerun_peak`) makes again only the comparisons the
  current file records as not made within the budget, within X times it, the measurement checks kept as made (their
  budget is the contract's), and records the rerun (`peak_rerun`: `budget_x`, `peak_s`, `seconds`, `exceeded`).  X
  is at least `scoring.PEAK_RERUN_X` = 2, the factor at which a miss again counts as reproduced (twice: the margin
  the grader's cap takes over the reference's slowest dataset, so a comparison that runs out again there runs out
  from its own cost, not a node's load), and defaults to it; a smaller X is refused, since a miss there would
  reproduce nothing (`scoring.budget_reproduced` counts no miss below that factor).  It skips a budget no larger
  than one the comparison already ran out of and a dataset whose file is a fault row or a placeholder (no
  comparison made: run the post stage first).  A comparison that runs out again at `PEAK_RERUN_X` or more times a
  budget it missed (`scoring.budget_reproduced`) is escalated as a reproduced evaluator fault is: the run is shown
  unranked (`sbh.score.REPRODUCED_BUDGET`, `v3.peak.n_budget_reproduced`).
  A further reference's own `evaluator_budget_x` scales the stage's budgets for that reference's run alone, and is
  never the remedy for a comparison that missed its budget.
- A dataset the stage's own code fails on (an exception of the evaluator's, isolated per dataset) is written as its
  fault row (`post.fault_row`): status "verify error: <error>", the message and where it raised in
  `evaluator_error`, stamped `errors_marked` and current under the stage's rules, its earlier comparison attempts
  carried, no measurement, correction or comparison.  Before a dataset is checked its placeholder row is written
  (`post.unfinished_row`, binned_likelihood `verify.unfinished_record`: status "verify error: grading did not
  finish", `evaluator_error`, its attempt counted), so a post stage process that dies (killed, out of memory) leaves
  an evaluator fault, retried and escalated as any other, never a completed dataset without a row.  Every row counts
  the stage's attempts on the dataset (`grader_attempts`, binned_likelihood `verify.count_attempt`, carried from the
  file it replaces).  The stage checks every dataset it is given once, then checks once more each whose fault has
  not recurred (`verify.to_retry`), the first attempt's row standing if the job ends during it, unless the fault is
  systemic: every place it faults lies where the rows of more than half of the run's completed datasets fault, as
  their files then stand (`verify.is_systemic`, `verify.systemic_places`, `verify.mark_systemic`: not checked again,
  `systemic_fault`, `grader_attempts`' `n_systemic`; a retried row, `retried`, found systemic once its retry ends is
  marked too).  The datasets of one process (`post.batch`, under `sbh <cmd> --phase post` and `all`) are checked as
  one stage, so that decision is over all of their first attempts; one dataset per job decides from the files as
  they stand, and the score phase settles the marks over the run's final rows (`post.settle_systemic`).  The scorer
  decides which faults are systemic over the final rows itself (`scoring.fault_split`, the binned_likelihood
  scorer's rule), whatever order the stage checked them in.  A later post stage redoes a fault row and a
  placeholder.  The scorer reads the row as an evaluator fault
  (`scoring.measurement_faults`): the dataset is not checked, so its GOF and closure comparison are left out and,
  under `measurement_required`, its gated corrections are pending (`function_measurement_gate`), never scored as
  answered and never charged, as a completed dataset not checked yet is ("its measurement not checked yet"); a fault
  the retry saw again shows the run unranked (`sbh.score.REPRODUCED_FAULT`), a systemic one leaves it provisional
  (`scoring.measurement_fault_reasons`).  Counted in `v3.measurement.n_evaluator_fault`; each row's message in the
  private scores' `measurement_faults`.

**The refit's cost.**  The refit moves every free number of a reported correction one at a time (restrict(), the
Jacobians, the refit comparison, parameter tables).  correctionlib routes each point to one leaf by the inputs alone
(`outputs.leaves`; a transform has none and is evaluated whole), so a move is evaluated on its leaf alone, at the
points it reaches, many moves per correctionlib call (`refit._Form.moved`), bit for bit the whole correction's
value there.  A lineshape template changes only in the pairs holding those objects (`_Lineshape.predict_moved`:
moves reaching most pairs predicted whole, each in a thread, the rest in one pass); the covariance comes from the
pairs' sparse influences on the moments.  Results agree with whole-correction evaluation to rounding.

A lineshape refit with more free numbers than fitted moments (k > m: a form finer than its measurement) refits in
the moments' space (`refit._moments_mode`, `refit._MomentSpace`): a k × k pseudo-inverse costs too much per step and
its cutoff, in the parameters' units, would decide what counts as constrained.  One thin SVD of the column-scaled
weighted response gives the constrained directions, the damped step, the uncertainties, the rank and the
covariance's projection in O(m² k).  With k ≤ m the pseudo-inverse stays.  Each record names its step (`lm`,
`n_parameters`, `n_moments`); the results page names such refits as forms finer than their measurement.

**Review: the closure term** (Δ_peak) is the kind's (any corrections task with a peak baseline in its closure), not
`Task.extra_terms`.  The task names its comparison (`closure_comparison`: `z_peak`, `peak.py`; `balance`,
`balance.py`) and supplies its selection and histograms (`peak_spec(flavour, request)`); records keep the `peak`
names (`measurement_verify.json` `peak`, closure.json `peak_baseline`, `v4_peak_cases`) whatever is compared, stamped
with the comparison's rule.  A new comparison is a module of the same interface on `peak.py`'s core
(`comparison.py`).


## The task base (`task.py`, `CorrectionsTask`)

`kind = "corrections"`, `result_type = "measurement"`; the attributes' meanings are in `task.py`:

- `function_cases(truth)`: the scored cases of one dataset;
- `function_baseline_id()`: what the per-case baselines depend on besides the reference's outputs;
- `closure_pull_rms`, `closure_max_pull`, `closure_bias_rho`: closure criteria from toys of the reference;
- `post_cpus`, `post_memory_mb`: the post stage's job shape;
- `closure_comparison`, `peak_spec(flavour, request)`: the post stage's closure comparison and its spec per flavour;
- `verify_budget_y`, `peak_budget_y`: the post stage's budgets per dataset (needs `compute_yardstick_k`); the
  contract states the first where the declaration gates (`measurement_budget`); a further reference's
  `evaluator_budget_x` multiplies both (`post.budget_factors`);
- `measurement_required`: True (default), a correction failing the measurement gate counts as not reported; False
  (lep_calib), the declaration is optional, recorded `undeclared` when absent, never a gate; the contract's
  MEASUREMENT.json opening and the self-check follow it;
- `measurement_refit`: True (default), refit and Δ_gof; False (lep_calib), the checks alone
  (`refit.verify_rule` records which);
- `evaluation_population`: where a reported correction is evaluated, for the contract (lep_calib: the simulation's
  Z → ℓℓ legs within the domain);
- `baseline_accuracy`: the ideal-case reference baseline, σ̄ × the reference's pull RMS per role group
  (closure.json `function_accuracy`), instead of reference parity; it nearly cancels in the relative
  headline (`sbh.score.relative_score`; docs/METRICS.md "Function outputs").

A corrections task's own parts: `generate` writes `request.json` into each dataset; `RUNTIME.md` and `base_env`
(lep_calib: correctionlib, ROOT, CMS combine); budgets, `compute_yardstick_k` (the reference timed in each
evaluation), stress submissions and attackers.

## The contract

The parts (`contract_parts.md`) are the unified contract's `corrections` piece (`tasks/EVAL_CONTRACT.in.md`,
`contract_pieces = ("corrections",)`): submission form, RESULT.json's covariances, corrections.json, the request
section (kinds table; the task's example request and the selections and trigger legs its requests can name, from its
contract values `request_example` / `request_names`), MEASUREMENT.json and the self-check.
No POI, model or nuisance-log section: a corrections analysis states its corrections in corrections.json, and a
`nuisances` object in RESULT.json is recorded unchecked (the log's grammar has one `eff_<obj>` per object while a
request names several selections).
