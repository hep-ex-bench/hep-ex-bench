# Model-level validation for `h4l`

How the harness validates a submission's *statistical model*, not just its
point estimates, with trusted tooling.  Metric definitions and how they are
scored: docs/METRICS.md, "## Suite-level metrics" (Δ_gof, Δ_truth,
Δ_claims) and "## Model-level validation (h4l; design in
`MODEL_VALIDATION.md`)".

## The problem

The sealed-truth loop scores point-estimate recovery but cannot see whether
the numbers come from the claimed likelihood, whether it fits the data, or
whether its μ is unbiased with accurate uncertainties.  Self-reported checks
are gameable; black-box reruns cost minutes per toy.  So the agent fits
however it likes but must export the model declaratively, and the evaluator
reruns every trust-critical test on it at histogram level (~ms per toy).

## The three layers

1. **Paired-injection probes** (black-box).  When `Task.generate` draws the
   background before the signal, two suites with one master seed and
   designs differing only in (m_H, μ) share the background byte-for-byte.
   Δμ̂ between them is the differential response (true: the injected
   difference); a constant pipeline gives Δμ̂ = 0 from one pair.
2. **Self-reported GOF** (the optional `quality` block's `gof_pvalue`).
   Judged for uniformity across seeds; clustering or p ≈ 0 flags misfit.
3. **Declared-model validation** (`MODEL.json`, required, contract §3): a
   pyhf/HistFactory workspace (plain JSON, writable without pyhf) with the
   binned m4l channels, templates and modifiers, the selected observations,
   and `mu` as POI.  Internal corrections must be folded into the templates
   so μ_ws ≡ μ_reported.  The evaluator runs:
   - `sbh/kinds/binned_likelihood/verify.py` (pyhf + cabinetry, or the JAX
     engine for all but the nominal fit):
     - **consistency** of the MLE with the reported μ (0.25 σ + 0.05);
     - cabinetry `fit(goodness_of_fit=True)` **saturated GOF**;
     - postfit **impacts** (cabinetry `fit.ranking` definition);
     - `--toys N` **toy ensembles**: toy-calibrated GOF p, bias at the
       fitted μ, toy-spread / quoted-σ ratio;
     - the model's own μ (`mu_fit`: best fit, profile half-width, Hessian and
       stat-only σ, Z_obs = √q0), which the scorer uses instead of
       RESULT.json's; the μ = 0 fit runs with the POI bound widened to
       min(lo_b, 0) on both engines (`mu0_outside_bounds`);
     - the declared `MODEL_GRID/` (`mass_grid`): every point in the task's
       mass range fitted with μ free; for a task that scores μ over the
       mass (`Task.mu_marginal_over_mass`) also each point's conditional
       profile in μ (`points[].mu_scan`, ~25 fits warm-started outward from
       the free fit), its template facts (signal yield at μ = 1, centroid,
       core widths, channel shares),
       the coverage of the range and the content checks against the task's
       yield curve (`coverage`, `content`; docs/METRICS.md, "Signal strength
       marginalised over the mass").  On the closure's reference run
       (`--prior-curve`) each point also gets σ_0,ref, the background-only
       Fisher width of μ on the μ = 0 Asimov data with the nuisances at the
       observed μ = 0 fit: the shape of the rule's mass prior.  The grid's
       link to MODEL.json (`mass_grid.link`), at the grid point nearest
       the reported m_H within the local spacing (`model.nearest_point`):
       the grid's own workspace there, at any mass the grid declares, is
       MODEL.json's to the tolerance MODEL.json's own counts set
       (`model.identity_rel`; `by` "identical workspace", no fit compared,
       and at a fitted point the profile takes the lower of the point's
       and MODEL.json's 2NLL, `link.model_nll`), else the grader's fit of
       that point reproduces its fit of MODEL.json (`by` "fit";
       docs/METRICS.md, "scored m_H (grid)").

     Both toy stages hold auxiliary data at observed values (`--gof-aux
     redrawn` restores combine's --toysFrequentist); the scored p is
     `gof_p_comb` (global and window statistics, the window the task's
     `gof_window`, h4l 105–140 GeV; (k+1)/(N+1)); the task's
     `model_verify_toys` toys (1000 by default) in the post stage.
   - A submission that also declares `MODEL_COMBINE/` (contract §3) is
     **graded in combine** (see "A declared combine model").

## Tooling decision

pyhf/cabinetry is the **eval-path** validator, so ROOT is not an
evaluation dependency.
Exception: a submission's own combine declaration is graded in combine
where combine is available ("A declared combine model" below).

## The JAX engine (`SBH_TOY_ENGINE=jax`)

Rebuilds the declared workspace as a vectorized JAX likelihood, the "twin"
(`sbh/kinds/binned_likelihood/jax_engine.py`: pyhf's default code4/code4p interpolation,
parameter space and constraints read from `model.config`; combine's
treatment of negative and zero yields).  Fitted with iminuit + exact JAX
gradient; uncertainties from the autodiff Hessian (parameters on a bound
held fixed).  Not the evermore library; `SBH_TOY_ENGINE=evermore` is an
alias, records say `"engine": "jax"`.

MC statistics as combine's analytic `autoMCStats`: a staterror gamma with a
Gaussian constraint, alone in its bin, is solved in closed form (with
nu = A + γB the minimum is the larger root of a quadratic) instead of being
handed to Migrad, which fits the other parameters on that profiled NLL.  The
minimum and the uncertainties are the full fit's (profiled Hessian = Schur
complement); a model with ~1400 staterror gammas fits ~40× faster.  Fits
that fix such a gamma, bins with two gammas, a gamma on a sample with a
negative POI-scaling coefficient, and shapesys stay on the full fit;
`SBH_BB_LITE=0` turns it off.

POI scaling (`build(..., poi_scaling=)`, passed by a kind whose model
declares it; binned_likelihood passes none): a sample's rate is
max(nominal + histosys, 1e-9) × c × ∏ P^p, or the least of several such
monomials, × its other modifiers, with c ≠ 0 of either sign and
p ∈ {−1, −½, 0, ½, 1}.  The sample carries a normfactor named P for exactly
each P with a non-zero power, so pyhf creates the parameters and bounds;
the twin replaces their product by the scaling.  A negative power needs
P > 0: at P = 0 its monomial is +∞, which loses a minimum but must be kept
out by the POI's bounds otherwise.  `build` checks the rates against
pyhf's per-sample rates (that sample's markers at 1) times the scaling at
two points, each scaled POI at a value of its own (so a monomial with two
POIs' powers swapped differs).

| stage | pyhf engine | jax engine |
|---|---|---|
| nominal fit, saturated GOF, consistency | cabinetry; pyhf's own MLE where cabinetry is absent (below) | cabinetry, the twin above 200 free parameters (`mu_model`/`mu_model_unc`); after a failure the twin (below) |
| postfit impacts | cabinetry postfit refits (`_fit_model`), pre-fit half dropped | twin refits at pyhf bestfit ± HESSE σ, same `{label: {up, down}}` schema, MC-statistics gammas (staterror, shapesys) skipped (`impacts_info.n_mcstat_skipped`); shifts beyond a bound evaluated at the bound (`impacts_info.clipped`), failed refit `null` (`impacts_info.failed`) |
| Z_exp | pyhf q0 on the Asimov at (bestfit, μ = 1) | twin q0 on the Asimov of the twin's floored rates (`z_exp_info.asimov_floored` when the floor changes it) |
| mass grid | pyhf fit per point | twin compiled once for the grid; a point the twin rejects is fitted by pyhf |
| μ scans per grid point (μ over the mass) | pyhf `fixed_poi_fit` per μ value | the grid's twin, conditional fits warm-started outward; σ_0,ref from one Asimov Hessian per point |
| `--toys` ensembles, truth-pinned coverage | pyhf sampler at the main rates floored at 1e-9 (`GEN_RATE_FLOOR`, `n_gen_floor`) + fitter; statistic = Poisson deviance | twin throughout (generation, toys from floored rates, observed GOF, conditional fit, refits: one likelihood); statistic = combine's saturated q |

### Negative and zero yields follow CMS combine

MODEL.json is read with pyhf semantics (modifiers, code4/code4p,
constraints, bounds); where pyhf and combine fit differently, the twin is
combine.

- pyhf's xlogy Poisson term rewards a negative expectation in an empty bin
  and is NaN in a filled one, so pyhf fits can end at negative rates.
- combine v11.1.0 (`autoMCStats` channels, CMSHistFunc on the raw
  templates) floors every yield at 1e-9: each sample's rate before its
  normalisation factors (nominal + histosys deltas; CMSHistFunc.cc:612) and
  each bin's total (CMSHistErrorPropagator.cc:185).  An empty bin at the
  floor adds 1e-9; a filled one pays −n ln 1e-9 = 20.7 n.
- The twin does exactly this (`sbh/kinds/binned_likelihood/jax_engine.py`, `CROP`); its NLL
  equals combine's CachingSimNLL to < 1e-6 on faithful conversions.

Consequences:

- Where every pre-factor rate stays ≥ 1e-9 the twin IS pyhf (the reference:
  bit-identical).  A zero template bin becomes 1e-9 × its factors.
- Every point is evaluable; start candidates are only a retry ladder.  An
  optimum where a rate crosses zero is a kink: a fit whose only failure
  there is EDM ≤ 10 × Migrad's goal is accepted and counted (`n_kink`).
- Floor occupancy is recorded, never dropped: `n_floor` (fits with a bin at
  the floor), `n_sample_floor` (fits with a negative pre-factor sample rate)
  in toy blocks, `impacts_info`, `z_exp_info`, `mass_grid`; per fit in
  `twin_fit` (plus `min_expected_rate`, and `from`: the observed fit
  restarts from the model's init when the floor is active at pyhf's best
  fit, kept if its NLL is lower by > 1e-3) and `freq_toys.cond_fit`.
- Every record keeps the pyhf best fit's `min_expected_rate` /
  `negative_expected_rate` / `n_negative_rate_bins`; pyhf toy stages count
  `n_negative_rate`.
- Toys (`toy_pdf`) use pyhf's sampler with the twin's floored rates:
  identical draws where the floor is inactive.  On the pyhf engine the
  toys are drawn the same way, at the generation point's main rates
  floored at 1e-9 (`verify.GEN_RATE_FLOOR`, `_pyhf_toy_pdf`; the bins
  raised counted as `n_gen_floor` on the toy block or its `cond_fit`):
  pyhf's own sampler refuses a negative rate, so a best fit with a
  negative rate in a bin gives toys, not an evaluator error, and where
  no rate is below the floor the draws are pyhf's, draw for draw.

### Statistics

- **Saturated statistic** (jax): combine's GoodnessOfFit `--algo saturated`,
  q = 2 (NLL(θ̂) − NLL_sat) with constraint terms in both
  (GoodnessOfFit.cc:122–235): Poisson deviance + Σ((θ̂ − a)/σ)² (+ Poisson
  constraints' excess).  `gof_statistic` = `saturated_combine` or
  `saturated_deviance` (pyhf).  Jax blocks keep the deviance-only p
  (`toys.gof_p_toys_deviance`, `freq_toys.gof_p_saturated_deviance`) and
  `gof_q_obs`, `gof_q_obs_deviance`.  combine's toys are b-only unless
  `--expectSignal` (Combine.cc:850); the battery passes r̂ / the truth to
  match the twin's generation points.
- **GOF p** (both engines): every valid toy fit counts (`n_toys`).  A toy
  counts when q_toy ≥ q_obs − 1e-3 (`GOF_TIE_EPS`, Migrad's residual at tol
  0.1).  `gof_dof` = main bins − rank of ∂(expected main rates)/∂θ over the
  free *unconstrained* parameters, on the pyhf model at the pyhf best fit
  (a POI-scaled model: on the twin's rates, `_gof_dof(rates=)`).
  At `gof_dof` ≤ 0 q is the minimizer's leftover, so every toy p is `null`
  with `gof_p_undefined` (scoring: docs/METRICS.md, Δ_gof).  A stage that
  gives no p says why (`gof_p_missing`): "dof" (`gof_dof` ≤ 0),
  "no_window" (no main bin in the GOF window) or "no_toys" (no toy fit of
  the model valid); each is the model's, charged in Δ_gof, while an
  error of the grader's own code on the stage is its `evaluator_error`.
- **Coverage** (truth-pinned stage, both engines): every valid toy is in
  the denominator (`n_coverage_toys`).  A toy with undefined POI σ (on the
  twin: no positive-definite curvature, POI on its bound) is covered iff μ̂
  is on the truth within 1e-4 (hi − lo) (`poi_bound_window`); recorded as
  `n_sigma_undefined`, `n_sigma_undefined_covered`,
  `n_sigma_undefined_poi_at_bound`, `sigma_undefined_max_poi_bound_dist`.
  Dropping them would bias coverage truth-dependently.  Pulls use toys with
  a σ (`n_pull_toys`); coverage is `null` only when none has one.  The
  interval convention near a bound is an open question.  On the twin,
  further parameters (another POI with a truth) can be held at their values
  in the conditional fit the toys are drawn at (`freq_coverage(extra_pins=)`,
  recorded as `pins`).

### The nominal fit and its attempts

The grader fits the declared model as the contract states, its own fit of
MODEL.json with HistFactory semantics (`verify.py` `verify_seed`,
`_fallback_nominal_fit`):

1. the first attempt: the twin from init above
   `SBH_PYHF_NOMINAL_MAX_FREE` (200) free parameters on the jax engine,
   else cabinetry (pyhf's MLE: MIGRAD at strategy 1, tolerance 0.1, then
   HESSE);
2. only when that failed: the twin from init (jax engine; its build reason
   kept when it cannot be built), then, where cabinetry is absent (a grader
   runtime), pyhf's own MLE.

A model none of them fits is a failed dataset: "fit failed: <the decisive
attempt's error> (decisive attempt: <engine> from <start>; before it <the
earlier attempts' errors>)".  Where MIGRAD reaches a valid minimum and
HESSE then gives no covariance, pyhf's optimizer raises an AttributeError
reading it; the status names that cause instead ("HESSE failed at a valid
MIGRAD minimum: the covariance is singular, so the model leaves its
parameters' uncertainties undefined (the model's failure; …)"), and so
does a mass-grid point's reason (`mass_grid.points[].why`).  Every attempt is in the
record's `fit_attempts` (engine, role, start, strategy, tolerance, steps,
validity, the twin's build reason, the error; `used` on the one whose
result is graded).  RESULT.json is read as reported (`_read_result`): a
file that is not a JSON object, a `pois` that is not a list and an entry
without a string name are not reported, and the consistency check skips
a μ entry whose value or total is not a number it can compare.

### What the grader reads, and whose failure an error is

The grader reads every output as the contract reads it (`contract.read_output`:
never through a symlink) and within its size caps (`model.MODEL_MAX_BYTES`,
`model.grid_cap`: `GRID_MAX_BYTES`, or `GRID_MAX_BYTES_MARGINAL` for a task
that scores μ over the mass; EVENTS.json `eventlist.MAX_BYTES`).  A
MODEL.json over its cap is "invalid workspace" before it is parsed
(`verify._size_gate`).  A grid over its cap is not read: its block is
`mass_grid.over_cap`, and the scorer applies the task's contract
(`Task.grid_over_cap_fails`: a failed run on that dataset, or no grid).
Grid files are listed with `lstat`, so a symlinked one is skipped as a
stray file is.

A fit of the model that raises is the model's (`ModelFitError`, "fit
failed: …": the nominal fit, MODEL.json's μ refit, the pyhf fits of the
toy stages and of the sequential GOF).  Any other exception in the
grader's own code is the evaluator's: the block it hit keeps its `error`
text and gains `evaluator_error`.  Each stage after the measurement (the
impacts, the Asimov Z_exp, the toys, the sequential GOF, the claims'
cross-checks, the grid) runs in its own guard (`verify._stage_error`,
`_guarded_grid_stage`, `_guarded_crosschecks`), so an error there stays
on its block (`impacts`, `z_exp_asimov`, `toys` or `freq_toys` or
`gof_seq`, `quality_crosscheck`, `mass_grid`; the grid link's own code on
`mass_grid.link`, and on `mass_grid` too when no route links;
`link.model_nll` when the profile at MODEL.json fails;
`pyhf_crosscheck` and `combine_model` on a combine-graded dataset) and
`mu_fit` and `mass_grid` are kept.  A "verify error" of the record itself,
carrying `evaluator_error`, is left for an exception before the
measurement exists.

What a submission reports or declares is read at the model's boundary:
in every form the grader's checks fuzz (the claims, the grid's numbers,
inits and bounds, constraint widths, rates up to and beyond the float
range), a value the grader's own code could not use is a claim that
disagrees, a hole in the grid or the model's failure, named before
anything raises, never an evaluator fault.  An exception none of these
guards foresees stays the evaluator's: an evaluator fault, verified once
more before the score (below).  The
optional claims (the `quality` block's values, `significance_exp`, a
pre-v0.4 `gof.p_value`) are read through `verify_fit.finite_number`: a
JSON number that is finite, never a bool; a string (a numeric one too:
the contract asks for a number), a list, an object, null, NaN, an
infinity or an integer beyond the float range is not a number, and the
claim's cross-check records it as disagreeing (`agree` false, `why` "not
a number", `given` a short description of what was there), whether or
not the evaluator has a number of its own to compare.  The record keeps
a bounded copy of the quality block (`self_quality`: at most 50 names,
each value a finite number or null, a note of at most 300 characters).
The grid checks' arithmetic is total (`verify_grid`): a ratio of two of
a grid's numbers (a point's signal yield against MODEL.json's point, a
template's core or sensitivity width against the lineshape's) is taken
only where it is a positive finite float (`_ln_ratio`), and otherwise is
no ratio, so that point is not compared on it (its yield not
band-clipped, its widths not compared; the smoothness check, which
compares logarithms, still sees an extreme yield); the yield band's
stretch exp(ln r − b) is held within the float range (`LN_FLOAT_MAX`,
the logarithm of the largest finite float, derived); a lineshape axis
whose span leaves the float range is none, the window fraction at a mass
that is not positive applies no correction, and a centroid line that
cannot be fitted, or a centroid that is not a finite number, counts as
off the line.  A grid point whose yields are 1e±300 or 1.7e308, whose
sums are subnormal or cancel, whose edges or fitted 2NLL and μ are
extreme, therefore gives an unusable or a usable grid, never an error
of the grader's on a scored block.  Under the grid's deadline
(`GRID_DEADLINE`, 0.7 of a verification cap) no point is started past it,
and each point's fit and scan run within what is left of it
(`_point_budget`, `sbh/budget.py`): a point that would run past the
deadline is stopped there and is a hole in the grid ("wall budget ran
out", counted in `mass_grid.n_budget_cut`), never a dataset that runs out
of its cap, while the dataset's own cap still ends its verification.  A
slow point (extreme modifiers or POI bounds can hold one pyhf fit for
minutes) therefore uses up what is left of the grid's time and may leave
a grid that is unusable (no mass measurement), never a failed dataset.  A shipped-rule task's grid stops between points only,
as it shipped.

Under the standard rule, what pyhf's own fit refuses, or no toy can
draw, is the model's failure on both engines and both grading kinds,
refused at the model's boundary before the grader's code meets it:

- **An init outside its bounds.**  A declared parameter, free or fixed,
  whose init lies outside its bounds, or whose bounds exclude every
  value: pyhf's own fit refuses such a model, and the JAX twin refuses
  it alike (`jax_engine.init_outside_bounds`, read by `jax_engine.build`
  and by the scan_likelihood grader's twin: "fit failed: init outside
  its bounds: <parameter> init <v>, bounds [lo, hi]").  The probe point
  at which the twin's rates are compared with pyhf's
  (`jax_engine.probe_point`) is total: an init outside its bounds is
  first taken to the nearer bound and a step that is not a non-negative
  finite number is 0, the draws of a healthy model unchanged.
- **A constraint width.**  A Gaussian constraint whose width is not a
  positive finite number (a lumi parameter's `sigmas`, a staterror
  bin's relative uncertainty; normsys and histosys have width 1), or a
  Poisson constraint (shapesys) whose factor (nominal/uncertainty)² is
  not finite (a relative width of 0 or not a number), has no likelihood
  a fit can minimise, and pyhf's fit fails on it
  (`model.constraint_width_problem`; a staterror bin of zero uncertainty
  pyhf fixes itself, so that one passes).  It is "fit failed: constraint
  width not a positive finite number: <name> width <w> (<kind>
  constraint) (pyhf's fit fails on it, so no fit is made)", before any
  fit: on MODEL.json (`verify._fit_refusal`, in `verify_seed`, a
  combine-graded dataset's MODEL.json measurement `_combine_record` and
  its regrid `_regrid_combine`), in the twin's build, on each grid point
  (`mass_grid_check`: a failed point) and in the scan_likelihood
  grader's twin.  No Gaussian constraint is then ever drawn at a width
  scipy refuses.
- **A toy rate.**  Every toy is drawn by numpy's Poisson sampler, which
  refuses a rate above `verify_fit.POISSON_RATE_MAX` = 9.2234e18 (numpy's
  own limit, derived as numpy derives it: the largest C long less ten of
  its square roots, so that a draw stays a long), an infinite or NaN
  rate and a negative one.  Before any draw (`verify_fit.check_toy_rates`,
  in `toy_ensembles`, `freq_coverage`, which the scan_likelihood
  grader's GOF runs, and the sequential GOF's `seq_scored_gof`, so the
  verdict does not depend on `SBH_GOF_SEQ`) every main-bin rate the toy
  pdf draws at (`verify._drawn_main_rates`: the twin's floored rates, or
  pyhf's floored at `GEN_RATE_FLOOR`) and every Poisson-constraint rate
  (`verify_fit.poisson_aux_rates`: the parameter times its factor, which
  pyhf draws even where the toys then hold the auxiliary data at its
  observed value) must be a finite number in [0, `POISSON_RATE_MAX`].
  Otherwise `ModelFitError`, "an expected rate of <r> events (<where>,
  at the toys' generation point), beyond any Poisson draw": the toy
  stage's "fit failed", the model's, its GOF charged (`GOF_CHARGE_MAX`;
  the scan_likelihood grader catches it beside a truth-pinned fit that
  does not converge, `gof_failed`).  Such rates come from the declared
  model at its own fit (a junk bin of 1e19 events, a nuisance fixed at
  a pull of 700, a normfactor started at 1e300 or fixed at 1e19, a
  shapesys of relative width 1e-10, whose constraint draws at 1e20); an
  observed GOF statistic of that order has p ≈ 0.  The engines differ
  in severity there (a judgement call: each engine keeps its own fit's
  verdict): pyhf's own nominal fit fails on the fuzzed models with rates
  of order 1e19, so on the pyhf engine their datasets fail; the twin
  fits them, so on the jax engine (the post stage's, and the
  scan_likelihood grader's one engine) only their GOF fails.  Both are
  the model's failure.
- **A likelihood that is not a number.**  The twin's fit is valid only
  at a finite NLL (`JaxPyhfModel.finite_nll`), in both of its branches
  (all parameters fixed: the point evaluated; Migrad), so a likelihood
  that is not a number at its minimum (a NaN observation, rates beyond
  the float range) is no fit (`valid` false), never a minimum a stage
  reads.  A grid point whose fit ends at a −2 ln L that is
  not a finite number is a failed point ("fit failed: its -2 ln L at the
  fit is <v>, not a finite number"), never a NaN in the profile, and the
  grid's tie set ranks only points whose 2NLL on the support is a finite
  number (`verify_grid.grid_coverage`), so it is never empty where one
  ranks.

What the evaluator chooses itself is never the model's: a toy's own
draw and refit (a refit that raises or ends invalid costs only that
toy, `n_failed`) and the twin's probe point (total, above) are guarded
where they are made.  A task kept on the rule it shipped with
keeps its grader as it shipped (`jax_engine.build(..., shipped=True)`,
`verify.SHIPPED`): its twin refuses only a free parameter's init outside
its bounds, where its cross-check point raised (`probe_point(...,
total=False)`, numpy's "high - low < 0", read as the twin not built,
pyhf's fit then refusing the model), so on the jax engine a fixed
parameter outside its bounds is fitted on the twin at its nearer bound
once pyhf's fit refuses the model; it checks no constraint width or toy
rate, counts a fit at an NLL that is not a number as any other, and
ranks its grid's points as it did.

The blocks whose evaluator error leaves a number the score reads out
(`verify.SCORED_BLOCKS`: `mu_fit`, `mass_grid` and its link, `toys`,
`freq_toys`, `z_exp_asimov`, `quality_crosscheck`; `evaluator_errors`)
leave those cases pending: the scorer leaves them out until the
evaluator is fixed and the dataset verified again, never charging the
model (docs/METRICS.md, "Failures").  An error on a report-only block
(`verify.REPORT_ONLY_BLOCKS`: `impacts`, `gof_seq`, `pyhf_crosscheck`,
`combine_model`; `report_only_errors`) is recorded apart, in the file
summary's `evaluator_error_report_only`, and leaves nothing out:
`combine_model` is the combine battery's own block, which no score term
reads (a combine-graded dataset is measured on MODEL.json, and one whose
battery raised is graded on MODEL.json whole).

The grader stamps every record it writes (`ERRORS_MARKED_KEY`,
`"errors_marked": 1`; `verify.mark_errors`, on every exit, a dataset
that ran out of its cap among them; the scan_likelihood grader alike):
each error of its own code carries `evaluator_error`, so an unmarked
error on a stamped record is the model's.  A stored record without the
stamp was graded before the grader marked its own errors, by a grader
that wrote some blocks for any exception, its own code's included; the
scorer reads each such unmarked error there as an evaluator fault to
regrade (`scoring.regrade_marked`; docs/METRICS.md, "Failures"): a
"verify error" status, an error block on a toy stage that is not a
failed fit of the model, a `mu_fit` error block that is not a failed fit
of the model, a `mass_grid` error block not marked unusable (`usable`
false, the model's grid with fewer than three usable points, stays the
model's), and a grid whose link the grader could not check (its `why`
"linkage check failed: …").  The grid and link stages keep a record's stamp as stored:
a record whose toy blocks an earlier grader wrote does not claim its
errors were marked.

Each record counts its verifications and those that ended on an
evaluator error of a scored block (`grader_attempts` = {`n`,
`n_evaluator_error`}, carried from the verification file a run replaces,
`<out>` or `<out>.prev`; `verify.attempts_of`, which counts a record
written before them as one attempt, an evaluator error only where it is
stamped and carries one on a scored block: an unstamped record's error
came from an earlier grader that may since have been fixed, so a fault
in the dataset's first regrade is a first fault), and the file summary
lists, over the records as they stand, the datasets' evaluator errors
(`evaluator_error`), those that recurred over two or more attempts and
are not systemic (`evaluator_error_reproduced`) and those of a systemic
fault (`evaluator_error_systemic`).

Before it grades any dataset, the verification writes every dataset's
placeholder into its file (`verify.unfinished_record`: status
`verify.GRADING_UNFINISHED`, "verify error: grading did not finish",
with `evaluator_error`, stamped, its attempt counted on), and each
dataset's record replaces its placeholder as it finishes.  A job that
dies (killed, out of memory) therefore leaves an evaluator fault on the
datasets it did not finish, counted, retried and escalated as any other
(a death that recurs reproduced, one on most datasets systemic), never a
completed dataset without a record; the scorer's "no grader record"
(`NO_GRADER_RECORD`) is left for a dataset whose grading never started.
The scan_likelihood and corrections post stages write theirs before
each dataset's first attempt (below; the corrections kind's README).  A
task kept on the rule it shipped with writes none.

`verify --stage retry` (`verify.to_retry`, `_retry`) verifies once more,
with the full stage's flags, every completed dataset whose record
carries an evaluator error on a scored block that has not recurred
(`n_evaluator_error` below 2), counting the attempts on and writing the
file in place (its `.prev`, the verification before this post stage,
kept).  The post stage runs it between the verification and the rescore
(`post.post_score` step 1a, "fault re-verification"; not after a
verification that failed): a one-off clears, and a fault of the
evaluator's code that recurs is reproduced before the run is first
scored, so the scorer names it apart at once (the run unranked;
docs/METRICS.md, "Failures") and never publishes a score that leaves the
faulted cases out as if a rerun could clear them.  A fault a rerun
reproduces needs a fix of the evaluator's code.  A systemic fault is not
verified again: a dataset whose every fault place (the record, a scored
block, `mass_grid.link`, `mass_grid.link.model_nll`;
`verify.fault_places`) lies where the same place carries an evaluator
error on more than half of the run's completed datasets
(`verify.is_systemic` over `verify.systemic_places`, whatever `--seeds`
names; `verify.SYSTEMIC_FAULT`) is marked (`verify.mark_systemic`: its
latest attempt moves from `n_evaluator_error` to
`grader_attempts.n_systemic`, both attempts where the record is a post
stage's own retry, `retried`; the places in `systemic_fault`), and the
retry verifies only the others, a dataset with a fault of its own beside
the systemic one among them.  The verification graded every dataset
before the retry stage, so the decision is over the whole run, whatever
the grading order.  A deterministic fault on most datasets is the
evaluator's code or machine; verifying them all again would only double
the post stage's wall, and counting the attempts apart keeps them from
reading as reproduced when the fixed evaluator verifies the run again
(the run stays provisional, never unranked by it; docs/METRICS.md,
"Failures").  The marks are the grader's bookkeeping: the scorer decides
which faults are systemic and which reproduced over the run's final
records itself (`scoring.fault_split`, the same rule over the records as
the score reads them), and `verify.settle_systemic` marks every record
whose fault is systemic over a run's final records, so the marks agree
with it whatever order the datasets were graded in.  A link error counts
only where it leaves the grid unlinked: where the identical-workspace
route linked the grid anyway, nothing is left out.

Record fields of the grid link: `link` = {`linked`, `by` or `why`,
`point`, `distance`, `local_spacing`, the fit comparison's `mu_grid`,
`mu_model`, `mu_tol`, `twice_nll_grid`, `twice_nll_model`, `engine`,
`cross_engine`; the identity route's `identical`, `identical_point`,
`identical_file`, `identical_distance`, `identical_spacing`,
`identical_rel`, `identical_diff` or `identical_why`; `fit_linked`;
`model_nll` = {`point`, `engine`, `twice_nll_grid`, `twice_nll_model`,
`used`} or {`used` false, `why`}; `evaluator_error`}; a grid point whose
2NLL the profile replaced keeps its own fit as `points[].twice_nll_fit`.

### Fallbacks and where it runs

Every fallback of the grader and the scorer, counted in the summary's
`v3.fallbacks` or recorded only, is listed in docs/METRICS.md,
"Fallbacks"; none fails a dataset or is charged.

- A spec outside the supported modifiers (normfactor, normsys, histosys,
  staterror, shapesys, lumi), or whose observed fit does not converge on the
  twin, stays on pyhf with the reason (`engine`/`engine_fallback`, also in
  the summary).  Without jax every seed falls back (`engine_fallback: "jax
  engine unavailable: …"`); the Pool initializer never imports jax.
- Tool default: `pyhf`.  The **evaluator's dev post stage runs `jax`**
  (`sbh post-score` → `sbh/kinds/binned_likelihood/post.py`).

**Threads.**  XLA:CPU sizes its thread pools from the CPU affinity
(XLA_FLAGS cannot limit it), so every jax-engine process is pinned to one
CPU before its first JAX computation (`pin_single_cpu` in the jax-free
`sbh/kinds/binned_likelihood/cpu_pin.py`, all threads; `SBH_JAX_PIN=0` disables).  Workers pick a
CPU by index in their own pool, so two runs sharing one affinity set (an
interactive allocation) collide: give the second
`SBH_JAX_PIN_OFFSET=<first run's worker count>`.

**Engine agreement gate.**  No per-run double fitting.  `pixi run smoke`
runs `checks/jax_engine_check.py`: the engine's stages on tiny synthetic
workspaces, each against pyhf/cabinetry.

## A declared combine model: graded in combine

A seed whose `out/` has `MODEL_COMBINE/` (`datacard.txt` + inputs, or a
text2workspace-format `workspace.root`; contract §3) is graded by
`combine_battery` (`sbh/kinds/binned_likelihood/verify.py`) wherever combine
is available, regardless of `SBH_TOY_ENGINE`: combine (conda-forge
`cms-combine`, also in the `lepcalib` pixi env) is found through
`SBH_COMBINE_BIN`, its `bin/` directory.
Records keep the MODEL.json schema with `"engine": "combine"`, so the scorer
(`sbh/score.py` `gof_cases`) reads `freq_toys.gof_p_saturated` /
`toys.gof_p_toys` unchanged.  MODEL.json is measured on such a dataset
exactly as on the MODEL.json path (`verify._combine_record`): the
nominal fit and its attempts (a MODEL.json none of them fits is the
"fit failed" record, the dataset failed, the combine block kept beside
it), the twin's observed-data fit on the jax engine, the scored μ
(`mu_fit`, its profile width; the twin's reason for leaving it on pyhf
in `mu_fit.engine_fallback`, not the record's `engine_fallback`, which
names the seed's engine) and `pyhf_fit.twice_nll`, so the grid links by
either route.  combine's own fit (`mu_model` ± `mu_model_unc`) and
MODEL.json's nominal fit beside it (`pyhf_crosscheck`: μ̂ and its
Hessian width, r̂ − μ̂, the negative-rate flag; no fit of its own) are
report-only.  The mass is evaluated only from a declared grid: the
declared pyhf grid goes through the grid stage against MODEL.json as on
the MODEL.json path, on every task, and a combine model floating MH keeps
its own MH scan beside it as `combine_mass_grid`; with no declared grid,
its scan stays in `mass_grid` with `engine` "combine", which the scorer
reads as no mass measurement.  Either `mass_grid` is stamped
`declared_grid` (whether a grid was declared) and `combine_grid_rule`
(`verify.COMBINE_GRID_RULE`, 2).

| stage | combine call | record |
|---|---|---|
| build | `text2workspace.py` (datacard form) | `combine_model.t2w_exit`, `timing_s` |
| same data | declared observed data vs MODEL.json `observations` (per channel; renamed channels paired by sums) | `combine_model.data_check`; mismatch → MODEL.json path |
| consistency | `MultiDimFit --algo singles` (r; r and MH when MH floats), `--saveWorkspace` | `mu_model`, `consistency` (MODEL.json tolerance), `combine_model.r_lo/r_hi`, `mh_fit` ± `mh_unc` |
| observed GOF | `GoodnessOfFit --algo saturated` | `gof_q_obs`, asymptotic `gof_p_saturated` (never scored); failure → MODEL.json path |
| σ, shapes, pulls | `FitDiagnostics --skipBOnlyFit --saveShapes` | `mu_model_unc` = r's HESSE error; `combine_model.postfit`, `parameters` (no postfit band: too slow) |
| Z_exp | `Significance -t -1 --toysFrequentist --bypassFrequentistFit --expectSignal 1` from the MultiDimFit snapshot | `z_exp_asimov`: post-fit Asimov (r = 1, nuisances and global observables at best fit) |
| impacts | per nuisance 2 × `MultiDimFit --algo none --setParameters p=θ̂±σ --freezeParameters p` | `impacts` (cabinetry definition; σ = FitDiagnostics HESSE), top 30 by \|corr with r\|, out-of-range shifts clipped |
| toys at the fit | `GenerateOnly -t N --toysFrequentist --expectSignal r̂ --saveToys` (seed 20260923), then `GoodnessOfFit --toysFile … --trackParameters r` | `toys`: `gof_p_toys`, bias / spread / `unc_coverage_ratio` |
| truth-pinned | same at `--expectSignal μ_true` (clipped into r's range; seed 20260924), `GoodnessOfFit` and `FitDiagnostics --skipBOnlyFit` on the saved toys | `freq_toys` (below) |
| mass profile | `MultiDimFit --algo grid -P MH --floatOtherPOIs 1`, 41 points over m̂H ± 4σ (MH-floating models) | summarised like the declared grid (`_interval_summary`), kept as `combine_mass_grid` beside the declared grid's `mass_grid` (in its place where none is declared: no mass measurement), either stamped `declared_grid` and `combine_grid_rule`; the declared MODEL_GRID is graded on MODEL.json's engine |

`freq_toys`: `gof_p_saturated` (scored); `coverage_68` = truth inside
FitDiagnostics' MINOS interval [r − rLoErr, r + rHiErr]; pulls on the
truth's side; valid toy = `fit_status` 0, 1 or 3 (census in
`fit_status_counts`); a truth-side error equal to the distance to r's range
edge (FitDiagnostics' substitute for a failed MINOS side,
FitDiagnostics.cc:511–512) counts as `n_minos_substituted` and is treated as
an undefined σ; `toy_r_gof_vs_fitdiag` checks both fits of each toy agree.

- **Toys**: `--toys N` for both ensembles.
- **Minimizer**: combine's default (Minuit2 cascade, strategy 1) first; if
  the best fit fails, retry once at `--cminDefaultMinimizerStrategy 0` (the
  twin's Migrad setting) and use it for the rest of the seed
  (`combine_model.minimizer`).  Every other fitting call gets the same retry
  when its output is unusable (no output, no finite q, fit status outside
  0/1/3 or no positive HESSE error, no valid toy fit);
  `combine_model.strategy0_retries` lists them.
- **`gof_dof`** = bins − floating parameters no constraint charges (counted
  in the workspace; a constrained nuisance's global observable is
  `<name>_In`).  Combine's shared `CMS_th1x` has the longest channel's bin
  count, so channels of different lengths are over-counted (open).
- **MH floating**: floats (`--floatParameters MH`, started at its fit) in
  every stage but the 2-POI fit; Z_exp freezes it at the fit.
- **No parameter range is passed to combine**: the declared model's ranges
  apply (a datacard's r is text2workspace's [0, 20]; a `workspace.root`
  carries its own).  The only range set is the MH scan window
  (`mass_grid.scan_window`).

**Not gradable → MODEL.json path.**  Combine absent, text2workspace fails,
observed data differ from (or cannot be compared with) MODEL.json's, best
fit fails, or the GOF cannot be established (bins not countable, count
`gof_dof` ≤ 0, observed `GoodnessOfFit` fails at both strategies, or toys
requested and neither ensemble yields a p): the seed is graded on MODEL.json
with `engine_fallback: "combine: <why>"` and the combine block kept
(`combine_model.not_graded`), so breaking combine's GOF cannot escape the
GOF charge.  A count `gof_dof` ≤ 0 is not taken as saturation (floating
rateParams on a dummy process could drive it below zero): the MODEL.json
Jacobian rank decides.

- `SBH_COMBINE_BATTERY=0` sends declaring seeds down the MODEL.json path.
- `SBH_COMBINE_KEEP=<dir>` keeps combine's inputs and outputs.
- `pyhf_crosscheck` records MODEL.json's nominal fit beside combine's
  (μ̂, its Hessian width, Δμ combine − pyhf, negative-rate flag):
  report-only; the scored μ is `mu_fit`.
- `combine_model`, the battery's own block, is report-only too
  (`verify.REPORT_ONLY_BLOCKS`): no score term reads it, and an error of
  the battery's code (its `evaluator_error`) sends the dataset down the
  MODEL.json path whole, leaving nothing out and counting in no attempt.

## The scan_likelihood grader

A task of the `scan_likelihood` kind declares a model whose samples may
scale with monomials in the POIs (`_meta.poi_scaling`), which pyhf alone
reads as a plain product of normfactors.  Its grader
(`sbh/kinds/scan_likelihood/verify.py`) therefore fits only on the twin
built with the scaling, never with pyhf.  What the model causes is its
failure: a parameter, free or fixed, whose init lies outside its bounds
("fit failed", `jax_engine.init_outside_bounds`, as above), a constraint
width that is not a positive finite number ("fit failed",
`model.constraint_width_problem`, as above), a modifier the twin does not
support (the kind's run gate refuses one first,
`model.ENGINE_MODIFIERS`), a spec its constructor
refuses (`jax_engine.SpecError`), rates that miss pyhf's per-sample rates
times the scaling by more than 1e-8 of a bin's gross rate (the sum of its
samples' |rates|: samples of either sign that cancel leave a net rate below
the rounding of their sum), a global fit that does not converge, a
verification that runs out of the task's cap (`Task.model_verify_cap`:
twice the reference's slowest dataset, as below).  Only the evaluator's own
failure (no jax, an exception in its code; the record marked
`evaluator_error`) leaves a dataset pending, under `Task.grader_strict`
too; which it is follows from where the refusal is raised, never from an
exception's text.  Every record is stamped `errors_marked`
(`verify.mark_errors`), and its post stage (`post.post_dataset`) writes
each dataset's placeholder before its first attempt
(`verify.unfinished_record`, "verify error: grading did not finish": a
grader process that dies leaves an evaluator fault, never a dataset
without a record), counts the grader's attempts on the dataset
(`grader_attempts`, carried from the record it replaces,
`post.previous_attempts`) and grades a dataset whose evaluator error has
not recurred once more, within the cap again (`verify.to_retry`), before
the score: a deterministic fault is reproduced at the run's first score,
as on the binned_likelihood post stage.  The first attempt's record is
written before the retry, so a job that ends during it leaves that
record.  The datasets one process is given (`post.batch` under `--phase
post` or `all` on several datasets; `post.post_datasets`) are all graded
once before any retry, so whether a fault is systemic (`verify.is_systemic`
over `verify.systemic_places`, `verify.seed_records` and
`verify.completed_seeds`) is decided over all of their first attempts; a
post stage of one dataset per job (the Slurm post array) decides from the
records on disk when its first attempt ends and again after its retry
(`retried`), and the score phase settles the marks over the run's final
records (`post.settle_systemic`) before it scores.  A systemic dataset is
not graded again (`verify.mark_systemic`).  The scorer decides the split
over the final records itself (`scoring.fault_split`), so only the post
stage's cost depends on the order: in a parallel array the earliest
faults of a systemic one may be graded once more before the majority
shows.  The self-reported claims are read through
`verify_fit.finite_number`, as above.  `sbh check` refits the model on the participant's side with scipy on
pyhf's per-sample rates times the scaling (`scaling.refit`, a diagnostic).
The grader runs once per completed dataset as the post stage between the
runs and the score (`--phase post`; `runs/<seed>/model_verify.json`), with
the task's POI ranges as the fit bounds: the global fit from starts spread
over the scanned POI's range; that POI's profile over the range (a coarse
scan holding the global fit, refined about every local minimum and q = 1
bracket, three warm-started sweeps; a point more than 1e-3 in NLL below the
global fit restarts it there and the scan is refined again); the best fit
is the scan's lowest point, so the best fit, the piece holding it and its
errors are one minimum's, and every q ≤ 1 piece is kept (a piece whose
q ≤ 1 run reaches a scan end off the range bound, where the fit at the
bound failed, ends at that point); every POI's q = 1 interval, Hessian
width and stat-only interval; the GOF and coverage toys with every POI that
has a truth pinned there (`freq_coverage(extra_pins=)`; coverage of the
task's second POI, the measurement's), the GOF dof on the twin's rates; a
truth-pinned fit that does not converge, or a generation point whose
rates no Poisson draw takes (`verify_fit.check_toy_rates`, as above), is
a failed GOF (`gof_failed`).
The record keeps the scan's failed points and disagreeing sweeps
(`n_failed`, `n_sweep_disagree`; the card shows them) and its rule
(`scan_version`; the scorer reads no other's).  No impacts, Asimov
significance, mass grid or combine leg.  Smoke: a model with two minima of
the scanned POI against a brute-force grid of multi-start fits, the
refusals, the cap (`checks/scan_likelihood_check.py --grader`).

## Per-dataset verification cap (opt-in: h4l_width and width_toy)

Definition and records: docs/METRICS.md, "## Model-level validation (h4l;
design in `MODEL_VALIDATION.md`)", "The model verification's cap".  In
short: the closure's reference run measures 2 × its slowest verification
wall (`sbh post-score --label reference` → closure.json
`model_verify_cap`), and every later run's verification fails a dataset not
finished within it.  Datasets graded by combine's battery are exempt (the
closure measured the MODEL.json path).
