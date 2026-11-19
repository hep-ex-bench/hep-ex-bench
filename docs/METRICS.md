# METRICS — every quantity the harness reports, defined once

Code: `sbh/metrics.py` (vendored, unchanged), `sbh/score.py` (per-case
loss, the score S_v3, compute), `sbh/hep/calibration.py` (Δ_cal),
`sbh/kinds/binned_likelihood/scoring.py` (score from the fit, Δ_gof,
Δ_syst, Δ_truth, Δ_claims, review flags), `sbh/kinds/scan_likelihood/`
(intervals from a declared scaled model), `sbh/kinds/corrections/`
(function outputs), `sbh/kinds/binned_likelihood/verify.py` (model
validation).

## Who gets scored on what

- **scored case** (`Task.poi_scored_v2`) — h4l: μ on every
  dataset; m_H on **every dataset with an injected signal** (μ_true > 0),
  weak ones included, never on a null.  A weak signal's accurate mass
  uncertainty is the grid's wide envelope interval, bounded by the task's
  mass range (Δ_phys below).  `gauss_peak`: μ on every dataset, peak
  position behind the applicability gate.
- **applicability gate** (`Task.poi_applicable`; the "scored seed" of the
  per-POI metrics) — a sealed seed enters a POI's pulls and slope (v1 stack,
  closure criteria, cluster bias test, leakage, discovery power) only where
  the task's gate holds, computed from the injected truth alone, so the
  gated set is the same for every submission (`gauss_peak`: **expected
  local significance Z ≥ 5**).  Null seeds (μ = 0) enter the FDR; weak
  seeds enter recovery and discovery power only.  Each v2 case records `gated`; `per_poi.n_ungated` counts cases
  outside the gate, `n_ungated_scalar_ref` those normalised by the scalar
  σ_ref (median over gated datasets) for lack of a per-seed σ — none in a
  passed closure, which records σ_ref,i on every dataset.
- **expected significance** — Asimov local Z of the injected signal under a
  reference-grade selection: Z = √(2[(s+b)ln(1+s/b) − s]), s and b the
  expected signal and background yields in the peak window.
- **pull** — (reported value − injected truth) / reported total uncertainty.

## Per-POI metrics (each POI: m_H, μ; over the gated seeds)

- **sensitivity (slope)** — OLS slope of reported vs injected values.  1 =
  faithful; 0 = data-independent output.  *Gate:* |slope − 1| ≤ max(0.15,
  3·slope_err); skipped when the truth does not vary.
- **bias** — mean pull.  *Gate:* |bias| ≤ max(3/√n, 0.05).
- **calibration (pull width)** — sample std of the pulls with the
  submission's own σ.  1 = accurate uncertainties; < 1 padded, > 1
  overclaimed.  *Gate:*
  |width − 1| ≤ max(0.25, 3/√(2(n−1))).
- **pull fit (artifact)** — unbinned Gaussian ML fit to the pulls: μ̂ (=
  bias), σ̂ (= width with 1/n), Fisher errors σ̂/√n and σ̂/√(2n).  Tested
  against N(0,1) at 2σ (μ̂ against 1/√n; σ̂ by n·σ̂² ~ χ²(n−1)).  Diagnostic.
- **precision** — median over scored seeds of min(1, σ_ref/σ_i).  Awarded
  only when bias and calibration both pass.
- **σ_ref** — the reference's median quoted uncertainty over the gated
  seeds, fixed by `sbh closure`; per-case baselines use the per-seed
  σ_ref,i (`closure.json` `reference_pois`, every dataset the reference
  completed).  The closure grades its own reference against exactly the
  per-dataset baselines it records (`sbh.score.own_reference_pois`), as
  every other run is graded, never against an earlier closure's or the
  scalar σ_ref.
- **POI score** — sensitivity_gate × calibration_gate × precision (gates
  0/1).
- **crosstalk** — slope of one reported POI on another injected POI;
  flagged above 0.10, not scored.

## Suite-level metrics

- **recovery** — fraction of sealed seeds finishing in time with exit 0 and
  a contract-valid RESULT.json.
- **false-discovery rate (FDR)** — fraction of μ = 0 seeds with a discovery
  claim.  Under score from the fit the claim is the model's: Z_obs ≥ 5 from
  the evaluator's fit (`discovery_source` "model"; `fdr_reported` keeps the
  RESULT.json claim, which Δ_claims checks); a task scoring μ over the mass
  counts Z_G ≥ 5, the grid's largest √q0 over the mass range (below).
- **discovery power** — fraction of gated seeds with a discovery claim (same
  source; `discovery_power_reported` beside).
- **S_phys** — recovery × mean over POIs of the POI score.  Diagnostic
  beside S_v3 (null for function outputs).

### Per-case loss (mean Δloss)

The basis of the score S_v3 (`sbh/score.py`).  Per scored (seed, POI) case the
Gaussian log-loss ℓ = ln σ + pull²/2 minus the **calibrated** baseline at
the reference's per-seed precision, ln σ_ref,i + ½ (σ_ref,i from
`closure.json`; never the reference's realized loss).  ℓ is strictly proper
for the forecast N(x, σ²): neither shaving nor padding σ helps in
expectation.  With σ_g the geometric mean of quoted σ and r the RMS pull,
mean ℓ = ln σ_eff + ½ + D(r), σ_eff = σ_g·r the error scale the pulls show, and
D(r) = (r² − 1)/2 − ln r ≥ 0 the calibration penalty (zero only at r = 1;
r = 2: 0.81, r = 0.5: 0.32).

**The physics score in one line.**  Per POI the charge is the mean of its
K per-dataset charges plus the bias term (below), Δ_p = (1/K) Σ Δ_i + β·B_p.
For an analysis quoting σ = r·σ_ref on every dataset, with pulls (in its
own σ) of mean b across the datasets and width s,

  E Δ_p = ln r + (b² + s² − 1)/2 + (K b² + s² − 1)/(2K)
        = ln r + (1 + 1/K)·(s² − 1)/2 + b²

— precision relative to the reference (ln r), accurate uncertainties (the
pull width s) and no bias shared across the pseudo-datasets (the pull mean
b: b²/2 from the cases, b²/2 from the bias term).  The reference, at its
own σ_ref with unit-width unbiased pulls, sits at 0.

The per-case rule (`sbh.score.case_charge`):

- *F* — charge of the task's "no answer" forecast, ℓ(fallback) − ℓ_ref
  (`Task.fallback_forecast`), through the case rule below.  A case with
  no measurement is charged max(F, 0) ("Failures"): F falls below 0 where
  the reference's width exceeds the forecast's, and "don't know" never
  scores better than the reference's own level.
- *the charge* — the raw charge ℓ − ℓ_ref is the precision piece
  ln(σ/σ_ref) plus the pull piece (pull² − 1)/2 (the split below), and the
  case is charged exactly that, the plain logarithmic score against the
  reference with no floor and no cap:

  Δ_i = ln(σ_i/σ_ref,i) + (z_i² − 1)/2.

  The rule is strictly proper at every precision: whatever an analysis's
  accurate uncertainty is, quoting it gives the lowest expected charge,
  and a narrower or a wider one costs D(r) on average.  A density case is
  charged −ln p(truth) against its baseline alike (its split by H, the
  density's mean −ln p, below), a function case the mean over its points.
  Per case: right, with accurate uncertainties, about 0; more precise
  than the reference with accurate uncertainties, ln(σ/σ_ref) < 0; "don't
  know" — a case the contract counts on a completed dataset without a
  measurement, max(F, 0), or the "no answer" forecast answered, F; a miss
  of z of its own σ, (z² − 1)/2 beyond its precision piece.  A charge
  that is not finite (a density that gives the injected value zero
  probability, a pull that overflows) is S = exp(−∞) = 0: the case fails
  the run (`UNBOUNDED_WHY`).
- *failures* — a failed dataset's cases are charged as cases with no
  measurement, max(F, 0) (`failed_charged`; the dataset is listed in
  `failed_datasets`), and the run stays scored.  The run fails (S_v3 = 0)
  only where a case cannot be charged: a POI without a no-answer
  forecast, a failed case without a reference scale, or a charge that is
  not finite.  A case with no measurement on a completed dataset is
  charged max(F, 0); a case left out as provisional is not charged
  ("Failures" below).
- *bias term* (`sbh.score.multiworld`) — each sealed dataset is one world.
  Under the case loss alone a bias small in every world but shared by all
  of them costs no more per world than random noise.  Per POI (per role
  for function outputs) with K charged cases (the answered ones and those
  charged max(F, 0) for no measurement), those with a signed pull z_i
  are combined as an analysis combines independent measurements, by
  inverse variance, and the combination's squared pull beyond its
  expectation is charged per case:
  - r_i = σ_i/σ_ref,i, the case's width over the reference's on its
    dataset: e^`delta_precision`, the precision piece below (σ for a
    Gaussian answer, the mean half-width of a two-piece interval, the
    equivalent width e^H/√(2πe) of a density, σ̄ of a function case), with
    σ_ref,i = e^(ℓ_ref,i − ½) from the case's baseline;
  - the width w_i = σ_i, as reported, and the miss u_i = z_i, the signed
    pull in units of that width ((x_i − truth)/σ_i for a Gaussian answer);
    neither is bounded;
  - a function case's z_i is the mean of its point pulls and s_i their sd
    over the points (`mw_spread`; 0 for a single value);
  - σ_star⁻² = Σ w_i⁻², z_star = σ_star Σ u_i/w_i,
    s_star² = σ_star² Σ s_i²/w_i², and
    B_p = (z_star² + s_star² − 1)/(2K);
  - the POI's charge Δ_p = (1/K) Σ_{i=1..K} Δ_i + β·B_p, β = 1
    (`V2_BIAS_BETA`).

  β = 1 reproduces, to O(1/K), the weighting of the combination scored as
  one more case among K + 1: a bias b shared by all K datasets at the
  reference's widths costs b²/2 in the cases and again b²/2 in B_p, twice
  its per-dataset charge, while independent errors cost nothing there on
  average (E(z_star² + s_star²) = 1).  An answered case without a signed pull
  counts in K, not in the combination, and so does a case with no
  measurement: charged max(F, 0), it adds nothing to z_star; with none to combine
  there is no term.  Under a task with `Task.multiworld` False, Δ_p is the mean of the
  cases.  For function cases z_star² + s_star² is the point-by-point
  combination's squared pull averaged over the points (the averaging the
  case loss uses), with the datasets' deviations from their own mean miss
  taken as independent: an accurate answer's mean pull over m points has a
  variance between 1/m and 1, and s_star² carries the rest, so
  E(z_star² + s_star²) = 1 whatever m and the points' correlation.
  B_p ≥ −1/(2K), with no upper bound.  On 15 datasets at the reference's
  widths: an accurate run's B_p is 0 on average (sd √2/(2K) = 0.047);
  every pull shifted by b gives a mean Δ_p of b², 0.25 at b = 0.5 and 1 at
  b = 1: b²/2 in the cases and b²/2 in B_p; pulls
  correlated within a profile (ρ = 0.3, 0.5) cost ρ·n_pairs/(2K²) =
  0.016 / 0.027 in B_p (n_pairs = 24 ordered pairs within profiles of 3,
  3, 3, 2, 2, 2); one dataset missed by 10σ adds 49.5/K in the cases and
  about 50/K² in B_p, 3.5 at K = 15 and 1.6 at K = 32; independent misses
  of any pull width pay B_p = (E u² − 1)/(2K),
  their case charge over K.  The pieces are standard: the case loss is the
  log score (Good 1952; Gneiting & Raftery 2007, JASA 102, 359) and the
  combination the inverse-variance weighted mean; charging its excess
  squared pull per case is this benchmark's.  Left out of the combination:
  a failed dataset's cases and cases with no measurement (charged
  max(F, 0), counted in K, no pull)
  and calibration cases (Δ_cal is the plain mean of its cases, with no
  bias term).  The closure
  records the reference's own z_star, B_p and its part of the charge per POI
  (closure.json `multiworld`: `rule`, `per_poi` with `zstar`, `bias`,
  `delta_multiworld`; `sbh closure` writes it, `sbh post-score --label
  reference` refreshes it), a diagnostic: nothing in the rule is set from
  it.
  - *signed pull* (each case's `mw_pull`, set by its kind): a pull is
    (reported − truth)/error, whatever the POI's parametrisation.  A point
    case (value − truth)/σ (the `pois` kind; binned_likelihood's fitted μ
    and grid m_H, a reported value while the grader is pending); a
    two-piece interval (value − truth) over the error on the truth's side;
    a density case (the mass-marginal μ, the scan_likelihood scan density) its
    likelihood pull sign(mode − truth)·√(2(ln p(mode) − ln p(truth)))
    (`sbh.score.likelihood_pull`): sign(x̂ − truth)·√q(truth) for a scan
    density, (mode − truth)/σ for a Gaussian one; a truth outside the
    density's support (p(truth) = 0) is an infinite pull with its sign, and
    a truth on an end of the support (a bound every density on that axis shares,
    such as μ = 0) has none: there a calibrated density's mode lies on the
    truth or on one side of it, so the pull is ≥ 0 by construction (about
    +0.4 on average) and would read as a bias; its case loss is unchanged.
    A function case the mean of its point pulls.  The case record keeps
    the miss u_i (`mw_miss`).
    Not the
    PIT pull of "Density pulls" (which the results page shows): the PIT of
    a flat-prior density shifts with the profile's skew, so on a skewed
    likelihood it would hide a bias.
  - *an error every dataset shares* (one simulation for all): an accurate
    run's misses are then correlated across the datasets, and the bias
    term charges the shared part to every run, the reference included:
    about (K − 1)c/(2K) per role at the reference's widths, c the
    covariance across datasets of the cases' mean misses (ρ for unit pulls
    correlated ρ; less for a function case, whose mean pull varies less).
    "About nothing for an accurate run" holds where the datasets' errors
    are independent.  The reference's own value is its `bias` and
    `delta_multiworld` in closure.json `multiworld.per_poi`.
- *Δ_phys* — the charge of the mean per-POI score,
  Δ_phys = −ln(mean_p exp(−γ Δ_p))/γ (`sbh.score.score_mean_charge`;
  `delta_cases` = `delta_mean`): the score of several quantities is the
  mean of their scores, so a run that measures one of n quantities keeps
  at least 1/n of that quantity's score, and one POI is its own Δ_p.
  `delta_multiworld` is the bias term's part, the difference to the same
  aggregate of the mean case charges.  Function outputs: a group of roles
  (one kind of correction, over flavours or regions) is one quantity,
  its Δ the mean of its roles' Δ_r, as the cases within a POI are
  averaged; the groups weigh equally in the score mean.  Within a
  quantity a charge is averaged, never a score: one role far more precise
  than the reference cannot stand in for the rest of its group.

Reported: `n_fallback` (cases without an answer: a failed dataset's, or
charged max(F, 0) for no measurement), `n_failed_charged` (the failed
datasets' cases charged max(F, 0)),
`n_no_measurement` (those charged max(F, 0) on a completed dataset, also per POI), per POI `n` (its K),
`delta` (= Δ_p), `delta_mean` (the mean case Δ),
`delta_multiworld` (β·B_p) and `multiworld` (`n` the answered cases
combined, `k` = K, `zstar`, `spread` = s_star², `bias` = B_p; the report card's
`score.v2.multiworld`; null where the term is not scored), `failed`,
`failed_datasets` and `unscored` ("Failures"), and `delta_se`, a cluster-robust SE of the cases' mean
(seeds sharing the task's `cluster_key` share a draw; report-only, without
the bias term: the score's band resamples the bias term with the cases,
"Error bands").
Constants echoed as `bias_beta`, `eps` (the effort share: 0, "Scores"), `physics_rule`
(`sbh.score.PHYS_RULE`), `multiworld_rule` (null without the bias
term) and, for grouped roles only, `group_rule` (`sbh.score.GROUP_RULE`);
`cap_floor`, `clip`, `n_capped` and `n_clipped` are kept for the record's
shape and are null or 0.  The reference is scored like any submission and
can sit below 1 if it overclaims.

Diagnostic split of the raw charge: per case Δ_raw = ln σ − ln σ_ref +
(pull²/2 − ½) is kept as `delta_precision` (the precision piece,
ln σ − ln σ_ref) and `delta_pull` (the pull piece, pull²/2 − ½; function
cases: mean over the points, with σ̄ as "Function outputs" defines it);
an unanswered case carries the fallback
forecast's pieces (a case with no measurement is charged on them, its
charge floored at 0).  A case scored on a density p (the marginal μ rule
below, the scan_likelihood kind's scan density) splits by H[p] =
−∫ p ln p, the mean −ln p a calibrated density expects at the truth:
H[p] − H[p_ref], the log ratio of the equivalent widths
σ_H = e^H/√(2πe) (a Gaussian's σ at the same H), and −ln p(truth) − H[p];
for a Gaussian p these are the two pieces above.
Per POI `delta_precision_mean` / `delta_pull_mean` average them over the
same cases as `delta_mean`, and `v2.delta_precision` / `v2.delta_pull`
aggregate the per-POI means as plain means: over the POIs, or over the
groups of their roles' means.  Their sum is the mean
case charge: it differs from `delta_cases` by the bias term and the
score mean over POIs, and says whether a run pays for width (precision) or for being off
or miscalibrated (pull).

### Failures

A frozen pipeline that fails on a sealed dataset has measured nothing
there: every quantity of that dataset is charged as "don't know", and
the run stays scored.  The participant contract, which the agents ran
under, defines a failed run on a dataset (EVAL_CONTRACT §1, and §3, the
kind's outputs); the score fails the dataset exactly there, and nowhere
else.  Where the contract counts a completed dataset as having no
measurement of a quantity, that case is charged as "don't know" alike;
the run itself fails only where a case cannot be charged; where
the evaluation itself loses a dataset or fails, nothing is charged until
it is fixed or rerun.  The rule is the same for every kind
(`sbh.score.record_ok`, `failure_reason`, `load_scored_records`,
`_poi_cases`; a kind's own measurement through `Kind.case_status`:
"failed", `NO_MEASUREMENT` or "pending").

- **A failed dataset: charged as no measurement.**  Each scored quantity
  of a failed dataset is charged max(F_i, 0), F_i the charge of its
  no-answer forecast, as a case with no measurement is (below): it
  counts in its POI's K and enters no signed pull (the case's
  `failed_charged`; `v2.n_failed_charged`).  The other terms count the
  dataset as they count one without a model ("S_v3 terms").  A dataset
  fails where the contract says the run on it fails:
  - "Non-zero exit, timeout, or a missing or malformed GATE file marks
    the run failed" (§1).  Non-zero exit: a crash, or the kill at the
    memory cap ("exceeding it kills the run").  Timeout: the kill at the
    stated wall clock ("the run is killed at the limit";
    `Task.run_timeout_s`), with no grace; running out of time is a failed
    dataset like any other, never a compute charge ("Compute").  The
    GATE file: no readable RESULT.json, or one the contract's checks
    reject (`sbh.contract.check_result`; "a symlinked output file counts
    as malformed"), whatever the run's other outputs;
  - "an environment that fails to build within 1500 s, or whose installed
    size exceeds 4 GB, fails every run" (§1);
  - "A missing or failing `MODEL.json` fails the run like a malformed
    `RESULT.json`" (§3, MODEL.json's GATE), and a run "whose
    `EVENTS.json` is missing or fails one counts as failed on that
    dataset" (§3);
  - an output over its size cap: "a larger file counts as a failed run on
    that dataset" (§3).  The grader reads every output as the contract
    does (never through a symlink) and within its cap: a MODEL.json over
    it fails unread ("invalid workspace"), and so does a MODEL_GRID over
    it where the contract makes that a failed run
    (`Task.grid_over_cap_fails`: the grader refuses the grid unread,
    `mass_grid.over_cap`, and every case of the dataset fails); the
    corrections kind's corrections.json, "larger than 20 MB fails the
    run";
  - with `Task.model_verify_cap`, "a model not checked within it counts
    as failed on that dataset" (§3).

  One more follows from what the contract says the evaluator measures:
  "The evaluator measures `mu`, `mH` and their uncertainties itself, from
  the model you declare" (§3), so a model the grader cannot fit leaves
  nothing to measure, and its dataset fails ("The grader's fit", below).
  The scan_likelihood kind's contract counts no case without a
  measurement: a scan that gives no density fails its dataset.

  `v2.failed_datasets` lists the failed datasets and why (`n_datasets`,
  `datasets[].seed`, `reasons`; the private scores also per dataset its
  `cases`, each POI the model cannot give and why).  The report card
  classes each dataset's failure (`failure_kind`, "Compute").
- **What fails the run.**  S_v3 = 0, the relative score 0 and its range
  [0, 0], final whatever else is pending, where a case cannot be
  charged:
  - a failed dataset's case on a POI for which the task states no
    no-answer forecast (`Task.fallback_forecast`), or for which there is
    no reference scale to charge it on;
  - a case whose charge is not finite: a density that gives the injected
    value zero probability, a pull that overflows (`UNBOUNDED_WHY`); at
    S = exp(−γΔ) such a charge is a score of 0 by the rule itself;
  - a failed dataset of a function-output task ("Function outputs"): its
    kind states no no-measurement charge for a dataset that did not
    complete.

  `v2` / `v3` `failed` list those datasets and why, in the form of
  `failed_datasets`.  Their uncharged cases carry `delta` null and `why`
  the reason; the Δ terms stay as diagnostics of what the run answered
  (null where it answered nothing).
- **No measurement: charged max(F, 0), never failed.**  The contract
  counts some cases of a completed dataset as having no measurement, and
  says the run does not fail:
  - the mass: "A dataset without a usable grid is evaluated as having no
    mass measurement", and "The grid is not a GATE: a run without one does
    not fail" (§3, the mass grid).  No usable grid is none declared; a
    grid `grid_profile` finds unusable (not one model, fewer than 3 fitted
    points, a minimum at a grid edge or a missing crossing on a side that
    stops short of the mass range, a profile whose whole region below 1
    lies outside it); a grid over its cap where the contract says so ("a
    grid above its cap counts as no grid"); a combine
    declaration's own MH scan in the place of a declared grid ("The mass
    measurement is evaluated only from a declared mass grid"); and a
    *grid not linked* to MODEL.json, by neither of the link's routes (the
    grid's workspace nearest the reported m_H is MODEL.json's, or its
    fitted point there reproduces MODEL.json's fit; "scored m_H (grid)").
    An unlinked grid is not the run's model ("`MODEL.json` is the grid's
    workspace at that mass", and a grid is usable "when it is
    `MODEL.json`'s model as above", §3), so it is never scored as the
    mass measurement; a task scoring μ over the mass takes μ from one of
    its points, as its contract states ("Fallbacks");
  - the mass-marginal μ of a dataset graded through a combine declaration
    without a usable scanned grid (none declared, one over its cap, one
    without the per-point scan, or one with no in-range point whose scan
    is usable): "without a grid its `mu` is not evaluated (a failed
    case)" (§3, μ over the mass), a case, not a failed run;
  - a correction: "a problem in one correction affects that correction
    only" (the corrections kind's §3), and each such problem "counts as
    its correction not reported" (missing, malformed, two entries under
    its name, not evaluating where its domain requires, or, where the
    task requires a declared measurement, one failing its checks); only
    the file-level problems above fail the run.

  Such a case is charged max(F_i, 0) (`sbh.score.no_measurement_charge`),
  F_i the charge of its no-answer forecast (`Task.fallback_forecast`; a
  correction's request `fallback`) through the case rule:
  the "don't know" charge.  The floor at 0 is the reference's own level,
  derived, not a constant: where the reference's width on a dataset
  exceeds the forecast's reach (a weak signal's mass, μ against a wide
  prior) F_i falls below 0, and without the floor withholding a
  measurement would score better than measuring as well as the
  reference.  It counts in its POI's K, enters no signed pull (it adds
  nothing to z_star), and its dataset is not failed (the case's
  `no_measurement` and `why`; `v2.n_no_measurement`; the fallback
  `no_measurement` with its reasons, "Fallbacks").  No measurement is the
  "don't know" outcome, and max(F_i, 0) puts it where the rule orders
  that outcome, right > don't know > confidently wrong ("Scores").  Every POI a kind can leave
  without a measurement needs a `Task.fallback_forecast`: a case with no
  measurement on a POI without one stops the score as a configuration
  error, never charged a default.  A case with no
  measurement on a POI without a reference scale to charge F on (no
  closure baseline and no scalar σ_ref) is provisional
  (`NO_REFERENCE_SCALE`); a case the model cannot give fails its dataset
  with or without one, and without one it cannot be charged (above).

  Answering the stated prior is not the same as no measurement.  The
  answered prior is an answer: charged its own F_i, below 0 where the
  reference is wider than the prior (the floor applies to the case
  without a measurement only), and its pulls (c − t_i)/σ_fb enter the
  bias term.  Where the truths centre on the prior's centre those pulls
  average out, and the two cost about the same; where the prior is off
  the truths' centre (a correction's nominal against one-sided truths)
  the bias term charges that shared offset to the answered prior and not
  to the case without a measurement, which then costs less (on a
  correction-like toy with one-sided truths, Δ_p 2.69 without a
  measurement against 3.49 answered; on h4l-like runs, "Scores").
  Neither is a cheap way out where F_i > 0: a wide measurement with an
  accurate uncertainty costs less wherever the task's mass range is
  narrower than its stated prior (h4l: 110–140 against 100–200 GeV).
- **Provisional, never charged.**  An infrastructure problem is never the
  run's.  A dataset without a record.json (the evaluation lost it: a node
  died, nothing was written; `load_scored_records`' placeholder) is left
  out of every term until it is rerun, and so is a case the grader has not
  finished that has no usable reported value (`v2.unscored`); a pending
  case with a reported value is scored on it (`v2.provisional`).  A
  strict grader (`Task.grader_strict`) never scores the reported value: a
  case it did not finish fails its dataset, except where the evaluation
  lost the dataset's grading (a completed dataset with no grader record,
  `NO_GRADER_RECORD`, "no grader record") or an evaluator fault left it
  out (below), which stay pending as on any task: strict fails only what
  the contract fails.  An
  evaluator fault is the evaluator's too: where the grader's own code
  raised (`evaluator_error` on the grader record, or on a block whose
  number a term reads, `verify.SCORED_BLOCKS`: `mu_fit`, `mass_grid` and
  its `link`, `toys`, `freq_toys`, `z_exp_asimov`, `quality_crosscheck`;
  a fit of the model that raises is the model's, "fit failed"), what
  reads it is left out as pending on an evaluator fault
  (`EVALUATOR_FAULT`), never scored on the reported value and never a
  failure, under `Task.grader_strict` too, until the evaluator is fixed
  and the dataset verified again: a physics case (its reason "evaluator
  fault: …"); a GOF case whose record or toy stage the fault hit ("GOF
  left out on an evaluator fault on n of N datasets"; strict charges only
  a missing p that is no fault on a dataset with a grader record); the
  claims checks read from
  `z_exp_asimov` or `quality_crosscheck` ("claims checks left out on an
  evaluator fault").  An error on a report-only block
  (`verify.REPORT_ONLY_BLOCKS`: `impacts`, `gof_seq`, `pyhf_crosscheck`,
  and the combine battery's own `combine_model`, which no term reads,
  since a combine-graded dataset is measured on MODEL.json and one the
  battery cannot grade is graded on MODEL.json whole; the grader's
  `evaluator_error_report_only`) leaves nothing out and counts in no
  attempt.

  The grader stamps every record it writes (`errors_marked`): it marks
  each error of its own code (`evaluator_error`), so an unmarked error on
  a stamped record is the model's.  A stored record without the stamp was
  graded before the grader marked its own errors, by a grader that wrote
  some blocks for any exception, its own code's included.  There each
  such error is read as an evaluator fault with the reason "graded before
  errors were marked: regrade" (`sbh.score.REGRADE`,
  `scoring.regrade_marked`): a "verify error" status of the record; an
  error block on a toy stage (`toys`, `freq_toys`) that is not a failed
  fit of the model ("fit failed: …") on a record whose GOF the model did
  not fail (`gof_failed`); a `mu_fit` error block that is not a failed fit
  of the model; a `mass_grid` error block the grader did not mark unusable
  (`usable` false is the model's grid with fewer than three usable points);
  and a grid whose link the grader could not check (its `why`
  "linkage check failed: …").  Each is left out, never charged and never
  a failure, strict or not, and named apart in the reasons, until the
  dataset is graded again.  Such a record without `grader_attempts`
  counts one attempt and no evaluator error (`verify.attempts_of`): the
  grader that wrote it may since have been fixed, so a fault in the
  dataset's first regrade is a first fault, verified once more before it
  can count as reproduced.

  What a run reports or declares is read at the model's boundary, so a
  run cannot cause an evaluator fault through what it reports or
  declares in any of the forms the grader's checks fuzz: each is a claim
  that disagrees, a hole in the grid or the model's failure, named
  before the grader's own code can raise.  The grader reads each
  optional claim of RESULT.json as a finite JSON number or as none (a
  string, a list, NaN: "not a number", a claim that disagrees, charged
  in Δ_claims); each of its stages after the measurement runs in its own
  guard, so an error there stays on its block and μ and the grid are
  kept; the grid checks' arithmetic is total, so no number a grid point
  holds makes the grader raise (a ratio of two of its numbers that is
  not a positive finite float is no ratio, an exponent is held within
  the float range), and each grid point runs within what is left of the
  grid's deadline, so a slow point is a hole in the grid, never the
  dataset's cap run out.  Under the standard rule what pyhf's own fit
  refuses, or no toy can draw, is the model's failure on both engines
  and both grading kinds:
  - a declared parameter, free or fixed, whose init lies outside its
    bounds (or whose bounds exclude every value): "fit failed: … init
    outside its bounds" (`jax_engine.init_outside_bounds`);
  - a constraint width that is not a positive finite number (a lumi
    parameter's or a staterror bin's Gaussian width; a shapesys
    Poisson constraint of width 0 or not a number): "fit failed:
    constraint width not a positive finite number: …"
    (`model.constraint_width_problem`), before any fit, on MODEL.json
    and on each grid point (a failed point);
  - a toy whose main-bin or Poisson-constraint rate at the toys'
    generation point is not a finite number in [0,
    `POISSON_RATE_MAX`], numpy's own Poisson limit of about 9.22e18
    events (`verify_fit.check_toy_rates`, before any draw): its toy
    stage's "fit failed", the GOF charged (`GOF_CHARGE_MAX`), strict or
    not; on the scan_likelihood kind its GOF failed (`gof_failed`).  On
    the pyhf engine pyhf's own fit already fails on the fuzzed models
    with such rates, so their datasets fail; the twin of the jax engine
    (the post stage's) fits them, so there only their GOF fails: both
    the model's;
  - a fit whose likelihood at its minimum is not a finite number: no
    fit (the twin's fit is valid only at a finite NLL), and a grid
    point whose fit ends at a −2 ln L that is not a finite number is a
    failed point; the grid's tie set ranks only finite ones.

  A task kept on the rule it shipped with keeps that grader's reading of
  each: its twin refused only a free parameter's init outside its
  bounds, checked no constraint width or toy rate, and counted a fit at
  a likelihood that is not a number as any other.  These are the forms
  the grader's checks establish, each on the values they build
  (docs/MODEL_VALIDATION.md, "What the grader reads, and whose failure an
  error is", lists each guarded place); an exception none of them
  foresees stays the evaluator's own: an evaluator fault as above,
  verified once more, then reproduced or systemic (below), never charged
  to the run.  A value the evaluator chooses itself (a toy's draw and
  refit) is never the model's: a toy whose refit fails costs only that
  toy (`n_failed`).

  The grader counts its verifications of each dataset and those that
  ended on its own error on a scored block (`grader_attempts`) and lists
  the datasets where one recurred (`evaluator_error_reproduced`).  Before
  the score, the post stage verifies once more every completed dataset
  whose record carries such an error that has not recurred (`verify
  --stage retry`; the scan_likelihood and corrections kinds' post stages
  alike): a one-off
  clears, and a deterministic fault recurs before the run is first
  scored, so a run is never first published with the faulted cases left
  out as if a rerun could clear them.  A rerun does not clear a
  recurring fault: the evaluator's code needs a fix, and the score names
  it apart ("reproduced evaluator fault on n datasets",
  `REPRODUCED_FAULT`; `v3.reproduced_faults`).  The results page shows
  such a run unranked, with that reason and no headline or range (the
  report card's `score.unranked`), and the scoreboard prints "unranked
  (its reason)" in place of its percentage and S, and "—" for its Δ_phys
  and S_phys: a score that leaves the faulted cases out would read better
  than the run is.  A configuration's mean over its trials on the results
  page averages its scored trials alone, shows "n=k of N" (k averaged, N
  its trials) and is marked provisional where any trial is unranked or
  provisional, so a trial left out of the mean is never dropped unseen.

  A grader process that dies before it writes a dataset's record
  (killed, out of memory) leaves an evaluator fault, not a missing
  record: before a dataset is graded its grader writes a stamped
  placeholder (`verify.unfinished_record`: status "verify error: grading
  did not finish", `verify.GRADING_UNFINISHED`, with `evaluator_error`
  and its attempt counted), which the dataset's own record replaces when
  its grading ends.  The binned grader writes every dataset's at the
  start of its verification; the scan_likelihood and corrections kinds'
  post stages each dataset's before its first attempt (a job that dies
  during the retry leaves the first attempt's record).  A death that
  recurs is then reproduced (the run unranked) and one on most datasets
  systemic, as any evaluator fault; "no grader record"
  (`NO_GRADER_RECORD`) is left for a completed dataset whose grading
  never started.  A task kept on the rule it shipped with writes no
  placeholder.

  A systemic fault is the exception: a fault whose places (the record, a
  scored block, the grid's link) all lie where the same place faults on
  more than half of the run's completed datasets (`verify.SYSTEMIC_FAULT`,
  `verify.systemic_places`, `verify.is_systemic`) is the evaluator's code
  or machine, not one dataset's.  Whether a fault is systemic is decided
  over the run's final records, whatever order the datasets were graded
  in: the scorer reads it from the records as they stand
  (`scoring.fault_split` of the binned_likelihood kind, which the
  scan_likelihood kind's scorer reads too; the corrections kind's
  `fault_split` alike), so a systemic dataset is never reproduced,
  whatever its attempts or marks, and any other faulted dataset whose
  fault the grader saw again (`n_evaluator_error` ≥ 2) is.  A dataset
  with a fault of its own beside the systemic one is not systemic: that
  fault is verified once more and can be reproduced.  The run stays
  provisional ("systemic evaluator fault on n of N datasets",
  `v3.systemic_faults`), never unranked by it, until the evaluator is
  fixed and the run verified again, whose first fault there is a first
  fault.  "More than half" is the plain majority: a fault on most
  datasets is the run's evaluation's, on fewer it may be the datasets'.

  The post stages keep the same rule for what they grade: a systemic
  fault is not verified again (verifying every faulted dataset again
  would only double the post stage's wall), and its attempts are counted
  apart (`grader_attempts`' `n_systemic`, the record's `systemic_fault`;
  two attempts where the record is the post stage's own retry,
  `retried`; the grader's file lists them, `evaluator_error_systemic`),
  so the fixed evaluator's next verification does not read them as
  reproducing it.  The binned grader's retry stage decides over the whole
  file, every dataset graded first; the scan_likelihood and corrections
  post stages grade the datasets one process is given (a batch: `--phase
  post` or `--phase all` on several datasets) all once before any retry,
  and a post stage of one dataset per job decides from the records on
  disk when its first attempt ends and again after its retry; the score
  phase then settles every mark over the run's final records
  (`settle_systemic`) before it scores.  The marks are bookkeeping; the
  score's split is the scorer's own.  The grading order changes only what
  the post stage costs: in a parallel array the earliest faults of a
  systemic one may be verified once more before the majority shows.

  A closure comparison the evaluator could not make (the corrections
  kind: the outputs it compares unreadable to it) is left out alike, and
  so is one not made within the evaluation's budget for it (also counted,
  as the fallback `peak_not_in_budget`) until it is made again at
  `PEAK_RERUN_X` = 2 or more times that budget (`measurement_verify
  --peak-budget-x [X]`, X at least `PEAK_RERUN_X` and defaulting to it; a
  smaller X is refused, since a miss there would reproduce nothing;
  "Constants of the score").  One that runs out again there is escalated
  as a reproduced fault is: the run is shown unranked ("reproduced budget
  miss", `REPRODUCED_BUDGET`).  A dataset the corrections kind's post
  stage itself fails on is recorded as its fault row (status "verify
  error: …", `evaluator_error`, `grader_attempts`; a placeholder row
  before it is checked, as above), checked once more and escalated as
  the grader's faults are (a systemic one alike); it is not checked, so
  its closure comparison and GOF are left out and, where the task gates
  on its checks, its gated corrections are pending, never scored as
  answered ("Function outputs").
  The score then carries its reasons
  (`v3.provisional_reasons`: named reasons and counts, never an
  evaluator's message, which can quote a path naming a dataset; the
  private scores keep each case's own reason), gets no speed bonus and
  is not final (the scoreboard marks it provisional and lists why); a
  page built with `--strict` refuses every provisional run, an unranked
  one among them.  A sealed-safe page keeps the card's
  per-case lists (`fit.failed`, `fit.no_measurement`, `fit.pending`)
  only for the POIs scored on every dataset, relabelled: a case of a POI
  scored on part of them (a pending m_H case) names a dataset of that
  truth-defined subset.  A failed run's S = 0 (a case that cannot be
  charged, above) is final whatever is pending.  A record.json that does not parse stops the score
  (`load_records`): the
  evaluator writes records atomically, so the fault is its own, neither
  dropped from the denominator nor charged.
- **The grader's fit.**  The grader fits the declared model as the
  contract states: cabinetry (pyhf's MLE) and, on the jax engine (the
  grader stage's) only after a failure, the JAX twin from its initial
  values (the twin first above 200 free parameters); where cabinetry is
  absent, the twin and then pyhf's own MLE.  A model it cannot fit is a
  bad model: its dataset fails (its cases charged as no measurement), the status naming the reason and the
  decisive attempt ("fit failed: …"), every attempt recorded
  (`fit_attempts`).  A model whose covariance is singular at a valid
  minimum (HESSE gives none) leaves its parameters' uncertainties
  undefined: the model's failure too.  Details: docs/MODEL_VALIDATION.md,
  "The nominal fit and its attempts".
- **A failed dataset is not scored from its model.**  A *malformed report* —
  RESULT.json present but rejected by the contract's checks
  (`sbh.contract.check_result`) — fails its dataset even beside a
  MODEL.json the grader could fit: the GATE file is the run's result ("a
  missing or malformed GATE file marks the run failed", §1).  A
  well-formed report that misstates what its model gives is a misreport:
  charged in Δ_truth and Δ_claims, never a failed dataset.
- **Why a failed dataset is charged as no measurement.**  A pipeline
  that fails on a dataset knows nothing about it, which is the outcome
  the no-answer forecast describes: its cases cost max(F, 0) each, and
  the datasets it measured keep their charges.  A measurement with an
  accurate uncertainty costs less than F wherever it is narrower than
  the forecast, and an answer that misses by many of its own σ costs
  more, without bound, so the outcomes order right > don't know (a
  failed dataset among it) > confidently wrong.  Nothing else in the
  score prices a failure: no failure premium, no completeness charge on
  the physics term, no reference side in the bias term ("Constants of
  the score").

### Scores

- **S_quality** — exp(−γ·mean Δloss), γ = 2 (`V2_GAMMA`): for accurate
  uncertainties (σ_ref/σ_eff)² · exp(−γD), the luminosity-equivalent
  precision relative to the reference, discounted by miscalibration.  A task may set
  `Task.score_gamma` (`sbh/score.py` `score_gamma`): γ = 1 scores the
  precision ratio σ_ref/σ; no task sets it (lep_calib: γ = 2,
  "Function outputs").  Recorded as `v2.gamma`.
- **No effort share.**  S is exp(−γΔ_total) alone: nothing in it rewards
  an output for moving with the data apart from the physics charge, so
  the outcomes order at the level of S as they do per case.
- The physics term is not a score of its own: S_v3 uses its Δ_phys.
  S_quality is not capped at 1: 1.0 is reference parity (for function
  outputs the ideal-case reference, "Function outputs"), and it has no
  upper bound: a run more precise than the reference with accurate
  uncertainties scores (σ_ref/σ)².
  Invariants, on S_v3:
  - *the outcomes in order* — right with accurate uncertainties > wide
    with accurate uncertainties > don't know (no measurement, a failed
    dataset among it: max(F, 0); the no-answer forecast answered: F) >
    confidently wrong.  On one synthetic h4l-like run each (μ and m_H,
    K = 16, σ_ref 0.3 and 0.5 GeV): right 1.45, right with one dataset crashed 1.02, accurate at
    4 σ_ref 0.091, no measurement on every case 0.084 (no pull, so no
    bias-term charge on the forecast's offset), the no-answer forecast
    answered 0.084, 10 σ_ref off with a random sign 10⁻⁴⁶, 10 σ_ref off
    tracking the truth 4·10⁻⁸⁷.  The bias term puts
    an answer off by the same amount and
    sign on every dataset below one off by as much with a random sign.  Where
    the reference is wider than the no-answer forecast (F < 0) a case with
    no measurement is charged 0, the reference's level: never better than
    measuring at the reference's precision;
  - *the basement* — a genuine poor fit (accurate uncertainties, 5× wider
    than the reference) > don't know (the no-answer forecast, a constant;
    abstention, every dataset failed, scores the same) > a nominal
    mover and a literature pin (a mover and a constant
    are not ranked: no physics goal orders them);
  - shaving, padding, selective failure, abstention and literature pins
    strictly lose; the reference is at the optimum; a crash is charged
    as "don't know" and so sits above a catastrophic miss; with
    continuous truths no snap, pin, shave, pad or
    failure attack beats a pipeline with accurate uncertainties;
  - *the exchange rate (rule A)* — every term is a charge on the physics
    term's per-case scale, so a weight is an exchange rate: a unit of
    calibration charge counts W_CAL = 0.10 of a unit of physics charge,
    and a run's S ratio to another is exp(−γ(ΔΔ_phys + W_CAL·ΔΔ_cal +
    W_GOF·ΔΔ_gof + …)) exactly.  Per case the rate differs from W_CAL:
    Δ_phys is the mean over the POIs of their mean case charges and Δ_cal
    the plain mean over the (dataset, nuisance) cases, so with each POI
    scored on every dataset one calibration case weighs
    W_CAL·n_POI/n_cal of one physics case, n_POI the POIs and n_cal the
    value-scored effects (0.04 with two POIs and five effects, 0.067 with
    two POIs and three, 0.2 with two POIs and one).  W_CAL is a stated valuation of the auxiliary
    measurements, not derived from the physics, and the per-case rate
    moves with the number of value-scored effects, which a declared
    working point can raise ("Known limits").  Good physics with
    calibrations left unlogged outranks exact calibrations with worse
    physics where the physics difference exceeds W_CAL times the
    calibration difference.  How far Δ_cal reaches is the task's: a
    nuisance measured to τ_j with an accurate uncertainty costs about 0,
    the exact value −½ (no credit for a width below τ_j), one unlogged max(F_j, 0)
    (about ln(prior_j/τ_j) on average), a confidently wrong one its cap
    ("Δ_cal").  The calibration outcomes order as the physics cases do:
    at τ_j with accurate uncertainties > wide with accurate uncertainties
    > unlogged (don't know) ≥ confidently wrong.  A measurement as wide
    as the effect's prior carries no more than the prior does and costs
    about what not logging it does, a wider one more.
  - *rule B, missing datasets cannot be bought back* — a failed dataset
    is charged max(F, 0) on each of its cases, never below the
    reference's level, so a run 2× or 2.5× more precise than the
    reference scores lower with three datasets missing than complete.
  - *known limit* — "don't know" has one price whatever its route:
    crashing on a dataset or completing it without a measurement costs
    max(F, 0) per case, writing the fallback forecast as the answer
    F and its share of the bias term (each loses to fitting).  RESULT.json cannot
    tell the forecast written as the answer from a genuine "measured
    nothing", and the answered forecast is charged as an answer: where
    the reference is wider than the forecast (F < 0) it scores below the
    reference's level, which the floor at 0 prevents only for a case with
    no measurement.
- **Relative score** — the headline (`summary.relative`, `sbh.score.relative_score`;
  the card's `score.headline` with rule "relative"): S_v3 over the reference's S_v3 on the
  same datasets, shown as a percentage (the reference 100 %; above 100 % a better result
  than the reference on these datasets).  The reference's S_v3 and per-case losses are the
  closure's `reference_score` (written by `sbh closure`, updated when `sbh post-score`
  grades the closure's run).  Its ±1σ range (`range`) is the paired band of S_v3/S_ref
  ("Error bands"), the reference's score recomputed from the cases its `reference_score`
  keeps (`band_inputs`) on every resample; a `reference_score` without them gives no range
  (null, `band_missing` saying why: re-score the closure's run).  A run scored 0 ("Failures") has ratio
  0, its range [0, 0].
  The
  `reference_score` keeps the physics rule it was scored under (`physics_rule`,
  `multiworld_rule`, `fail_premium`), and the relative score is refused for a run scored
  under another (re-score the closure's run: `sbh post-score --label reference`), as for
  `mu_rule`; the results page flags cards scored under different physics rules.  A suite
  whose closure recorded no reference score shows S_v3 (rule "v3").  The closure's
  reference, timed by the yardstick in its own evaluation, is the yardstick itself: its
  compute term is 0 (`sbh.score.reference_compute`, every kind); against a static
  reference wall it keeps the measured term.
- **S_v3** — the score.
  Δ_total = Δ_phys + 0.10·Δ_cal + 0.30·Δ_gof + 0.10·Δ_syst + 0.5·Δ_truth +
  0.05·Δ_claims + 0.1·Δ_compute (`V3_W_CAL`, `V3_W_GOF`, `V3_W_SYST`,
  `V4_W_TRUTH`, `V4_W_CLAIMS`, `V3_W_COMP`; a term with no cases adds
  nothing); S_v3 = exp(−γΔ_total), γ as S_quality; S_v3 = 0 for a run
  with a case that cannot be charged ("Failures").
  Each term is a charge on the per-case log-loss scale, 0 = the reference's
  calibrated level.
  **Score from the fit** (`v3.rule` "score-from-fit", `v3.score_source`
  "model"; opt-in via `Task.model_poi_sources`, h4l `{"mu": "fit", "mH":
  "grid"}`): Δ_phys and Δ_gof use the evaluator's fit of the declared
  model; of RESULT.json, Δ_truth and Δ_claims read the reported POIs and
  claims against that fit, and Δ_cal reads the nuisance log (each logged
  value against its injected truth), the model entering Δ_cal only through
  the calibration gate.  A task without it scores its reported POIs and
  has neither Δ_truth nor Δ_claims.

### Error bands

The ±1σ band of a score (`sbh/band.py`) is a paired bootstrap over the
pseudo-datasets: the suite's sealed datasets are resampled with
replacement, 1000 times (`BAND_REPLICAS`) from a fixed seed (`BAND_SEED`),
by cluster (`Task.cluster_key`: datasets that share a draw are drawn
together; without one each dataset is its own cluster), and on every
resample the whole score is recomputed from the per-dataset cases the
scorer kept, a dataset drawn twice counting twice:
- *Δ_phys*: per POI the mean charge of its charged cases (answered, or
  charged max(F, 0) for no measurement) plus β·B_p, the bias term of the drawn
  cases' misses (K the drawn cases), over POIs or groups of roles, as
  above.  A case drawn c_j times enters the combination c_j
  times, so for independent unit misses z_star² has the expectation
  Σ c_j² e_j² / Σ c_j e_j² (e_j its inverse-variance weight relative to
  the POI's narrowest case), not 1: each resample's B_p subtracts that in
  place of 1 (1 on the full sample), so a dataset drawn twice is not
  counted as two independent measurements;
- each weighted term from its cases, as the term is defined: Δ_cal the
  mean of the drawn cases' charges, each with its effect's gate forfeit;
  Δ_gof Fisher's ensemble test over the drawn usable
  p's, the other cases at 4.30; Δ_syst, Δ_claims and the closure
  comparison (Δ_peak) the mean of their cases; Δ_truth per POI the mean,
  then over POIs;
- Δ_compute the mean of the drawn completed datasets' charges, its speed
  bonus gated on the resample's own rest of Δ_total (none while the score
  is provisional).

Held at the full sample's value on every resample, and named in the band's
`held`: the calibration gate's forfeits (regressions over the whole run,
not per-dataset cases; the term is recomputed with them), a compute term
set by rule (0 for the closure's reference timed by the yardstick and for
a run under its own budget), and any term whose cases do not give back
its scored value.  The band is the 16 / 84 % quantiles (`BAND_Q`) over the
resamples, widened about the full-sample estimate by
√(n/(n − 1)) · t_{n−1}(0.84)/z(0.84), n the clusters drawn
(`sbh.band.widen_factor`, `v3.band.widen`; t_{n−1} the Student-t quantile
with n − 1 degrees of freedom, z the Gaussian one).  Two finite-sample
effects leave the plain quantiles short at few clusters: the
bootstrap's plug-in variance of a mean over n clusters is (n − 1)/n of
the unbiased one, and a mean over n clusters whose variance is estimated
from those same n clusters is Student-t distributed, not Gaussian.  The
factor is 1.10 × 1.11 = 1.22 at 6 clusters and 1.15 × 1.20 = 1.38 at 4;
it derives from n and is no free constant.  S and S/S_ref are widened in
their logarithm (S = exp(−γΔ_total): the same as widening Δ_total), the
Δ terms linearly.  With 1000 resamples each end is known to about 5 % of
the band's half-width.

- `v3.band`: `rule`, `replicas`, `seed`, `q`, `unit` (the cluster key),
  `n_units` (clusters), `n_datasets`, `widen` (the factor), the bands of
  `s_v3` and `delta_total`, `terms` (the band of each recomputed term,
  `phys` and `compute` among them) and `held`; null for a run scored 0
  (S = 0 is final).  Intervals null, with `why`, over fewer than two
  clusters or where the cases do not give back the run's own S.  The
  run's private scores keep what it resamples (`band_inputs`), and the
  closure's `reference_score` the reference's.
- `relative.range`: the band of S_v3/S_ref, the run's and the reference's
  scores recomputed on the same resamples, ratio by ratio (a dataset that
  fluctuates moves both and cancels), widened alike; `relative.band` its
  header and the band of the reference's own S (`reference_s_v3`).
- The report card's `score.range` is the relative range (rule "relative")
  or the band of S (rule "v3"); each breakdown term carries its `band` and
  the factor at its ends (`factor_range`).

Coverage, measured by the check on synthetic one-POI runs with accurate
uncertainties at the reference's widths (600 runs each, ±1.9 %; 68 %
nominal), of the band of Δ_phys around its expectation:

| datasets and clusters | pulls | widened | E Δ_phys above the band | below it | plain quantiles | resamples' mean − estimate |
|---|---|---|---|---|---|---|
| 15 in 6 profiles of 3, 3, 3, 2, 2, 2 | independent | 69.7 % | 23.2 % | 7.2 % | 64.2 % | −0.0003 |
| 15 in 6 profiles of 3, 3, 3, 2, 2, 2 | correlated ρ = 0.3 within a profile | 66.5 % | 25.7 % | 7.8 % | 60.3 % | +0.013 |
| 12 in 4 profiles of 3 | independent | 70.0 % | 24.0 % | 6.0 % | 59.2 % | −0.0003 |
| 12 in 4 profiles of 3 | correlated ρ = 0.3 within a profile | 70.0 % | 24.7 % | 5.3 % | 59.3 % | +0.018 |

On the 6 profiles at ρ = 0.3, a run biased by half its σ on every case
(E Δ_phys = 0.27) is covered 63.7 % of the time, slightly short (E Δ_phys
above the band 28.3 %, below 8.0 %).  The relative band of a run sharing
70 % of each pull with the reference's covers the true ratio
exp(−γ(E Δ_run − E Δ_ref)) 71.5 % of the time at ρ = 0 (the ratio above
the band 12.5 %, below 16.0 %) and 69.2 % at ρ = 0.3 (13.2 %, 17.5 %);
biased by half its σ, 67.0 % (11.5 %, 21.5 %).

On 40 datasets drawn in pairs (ρ = 0.9 within a pair, 20 clusters; 300
runs) the cluster band covers 70 %, a band resampling the datasets one
by one 59 %.  With independent misses
the resamples centre on the estimate; with misses correlated within a
cluster a cluster drawn twice still adds its cross terms to z_star² (the
+0.013 at ρ = 0.3), which the correction above does not remove.  In the
relative band the run's and the reference's offsets cancel.  The
analytic standard errors (`v2.delta_se`, `v3.delta_se`, each term's
`se`) are report-only.

The misses are uneven, and on both bands the more frequent miss reads
the run better than it is: at 4 to 10 clusters the expected charge lies
above the Δ band on 23–28 % of runs and below it on 6–11 %, and the true
ratio below the relative band more often than above it (1000 runs per
configuration, ±1.5 %; 3, 4, 6 and 10 clusters, ρ 0 and 0.3, accurate
runs and runs biased by half their σ).  No tested variant removes that
imbalance.  A bias-corrected and accelerated interval (BCa: its bias
correction from the replicas' share below the estimate, its acceleration
from the cluster jackknife, no new constant) leaves the sides about as
uneven (18.5–28 % above, 6.5–12.6 % below at 4 to 10 clusters, covering
64.9–71.4 % against the percentile band's 64.2–68.2 %) and covers less at
3 clusters (55–61 % against 64–68 % for Δ_phys, 65–66 % against 70–72 %
for the ratio); unwidened it covers 45–65 %, so it would need the same t
factor.  A studentized bootstrap evens the sides but covers 51–56 %
(57–65 % widened).  So the band keeps the replicas' percentiles with the
t factor; the remaining remedy is a variance-stabilising transform of
the charge's pull part.

### Fallbacks

A fallback is a secondary procedure the evaluation takes on a completed
dataset where its primary one cannot run or does not apply.  None fails a
dataset, and counting one charges nothing: each is part of the
procedure, and each is reported where it is taken, so that the procedure
can be revised.  One kind is a charge the rule makes anyway: a case with
no measurement (`no_measurement`) is charged max(F, 0) as "Failures"
states, and counting it adds nothing to that.  Another leaves the score
provisional anyway: a closure comparison not made within its budget
(`peak_not_in_budget`) is a rerun at `PEAK_RERUN_X` or more times its
budget the evaluation owes, and counting it says how often its budget
falls short.

**Counted** (`sbh.score.fallback_block`): {kind: {`n` datasets, `seeds`,
`what`, `pois` for a kind taken by physics cases, `reasons` (reason:
cases) where the kind says why per case, `detail` (dataset: number) where
it records what the fallback changed}} over the completed datasets, {}
when none was taken.  Where each is reported:
- the private scores' `fallbacks`: the full block, every kind with its
  datasets; the report card's `fallbacks` reads it;
- the public summary's `v3.fallbacks` (`sbh.score.public_fallbacks`): a
  kind taken by physics cases keeps only its count, what it is, its POIs
  and its reasons (each with any path or dataset id in it replaced,
  `sbh.score.public_reason`), never its datasets (which datasets carry a
  POI's case is truth-defined: a mass is scored only where a signal is
  injected); a kind taken by datasets keeps them;
- the results page's "fallbacks taken" tile and the scoreboard's last
  table, from those; a sealed-safe page keeps each kind's count and text,
  no datasets, and drops a kind taken by the cases of a POI scored on
  part of the datasets.

The kinds are binned_likelihood's (`scoring.FALLBACKS`, from the grader's
records and the scorer's measurement), the corrections kind's
(`corrections.scoring.FALLBACKS`, from its post stage) and the core's
(`sbh.score.CORE_FALLBACKS`, from the physics cases); the scan_likelihood
kind takes only the core's.  The last column quotes the participant
contract where it states the fallback.

| kind | the fallback taken | read from | the contract |
|---|---|---|---|
| `nominal_fit_fallback` | the nominal fit came from a later attempt: the first (cabinetry; the JAX twin first above 200 free parameters) failed | `fit_attempts` (the used one not the first); a record before it: `nominal_fit_twin.failed`, or `nominal_fit_fallback` where cabinetry is present | — |
| `twin_not_used` | the stages after the nominal fit ran on pyhf, not the twin | `engine_fallback` | — |
| `combine_not_graded` | a combine declaration could not be graded: MODEL.json graded instead | `engine_fallback` "combine: …" | "is recorded as such and the checks run on `MODEL.json` as without it" (§3) |
| `stage_on_pyhf` | a stage of a twin-graded dataset ran on pyhf: the truth-pinned toys or the Asimov Z_exp | `freq_toys.engine`, `z_exp_info.engine_fallback` | — |
| `mu_width_fallback` | MODEL.json's μ width not the two-sided profile: a side's scan failed, or neither side crossed (the Hessian width) | `mu_fit.sigma_source`, `mu_fit.profile` | — |
| `grid_points_on_pyhf` | grid points the twin could not build, fitted on pyhf | `mass_grid.engine_fallback` | — |
| `grid_link_by_fit` | the grid linked to MODEL.json by refitting, not as the identical workspace | `mass_grid.link.by` (a record without it: the refit, then the only route) | the contract's test: "the grid point nearest your reported mass must reproduce `MODEL.json`'s own best fit" (§3) |
| `grid_link_cross_engine` | the link's 2NLL compared across engines | `mass_grid.link.cross_engine` | — |
| `model_off_grid_minimum` | MODEL.json's point above the grid's 2NLL minimum by more than the link tolerance | `mass_grid.content.model_off_minimum.flag` | "flagged on the report card, not charged" (§3, μ over the mass) |
| `grid_point_from_model_fit` | the grid point that is MODEL.json's identical workspace scored at MODEL.json's own fit, lower than the point's warm-started one (a stalled grid fit); its neighbours keep their own fits, so a stall they share narrows the profile ("Known limits"); `detail` the drop in 2NLL substituted per dataset | `mass_grid.link.model_nll.used`, `twice_nll_grid` − `twice_nll_model` | — |
| `yield_band_clip` | grid points above the yield band scored at its edge (the mass-marginal μ) | `mass_grid.content.n_yield_clipped` | "the yield band clips, as above, and is flagged" (§3, μ over the mass) |
| `mu_single_mass` | the mass-marginal μ from one mass point: the grid unusable for the marginal (its best point) or absent (MODEL.json's own fit) | the μ measurement's `regime` | "evaluated for `mu` on one mass alone" (§3, μ over the mass, "Fallback") |
| `prior_curve_task_average`, `prior_flat` | the mass-marginal μ's prior on the suite-average reference curve, or flat | the μ density's `prior_source` | — |
| `gof_at_fit_stage` | the GOF p from the toys at the fit: the truth-pinned stage gave none | the GOF case's `stage` | — |
| `gof_global_only` | the GOF p the global saturated p alone: no combined p in the record | the GOF case's `rule` "legacy_global" | — |
| `ref_scalar_sigma` | a physics case's baseline the scalar σ_ref: the closure records no per-dataset baseline for it | the case's `ref_source` (core; `pois` set) | — |
| `no_measurement` | a scored case on a completed dataset has no measurement (no usable or no linked mass grid, a quantity the contract leaves unevaluated, a correction not reported): charged max(F, 0), its dataset not failed ("Failures") | the case's `no_measurement` and `why` (core; `pois`, `reasons`) | "A dataset without a usable grid is evaluated as having no mass measurement" (§3); "a problem in one correction affects that correction only" (the corrections kind's §3) |
| `peak_not_in_budget` | the closure comparison was not made within the evaluation's budget for it (`peak_budget_y`): the dataset has no peak case, and the score is provisional until the comparison is made again at `PEAK_RERUN_X` or more times that budget (`measurement_verify --peak-budget-x X`); missed again there, the run is shown unranked (`REPRODUCED_BUDGET`) | the comparison's `why`; `peak_attempts`; `v3.peak.n_not_in_budget`, `v3.peak.n_budget_reproduced` (corrections) | — (no contract states that budget) |

**Recorded, not counted.**  Each is in the record named, and none is a
failure of the procedure it belongs to:

| fallback | what it does | recorded in | why not counted |
|---|---|---|---|
| the grader runtime's nominal fit | where cabinetry is absent (the grader stage), the JAX twin from its initial values first, then pyhf's own MLE | `fit_attempts`; `nominal_fit_fallback` "no cabinetry here …" | that runtime's fit sequence ("The grader's fit"), not a failed first attempt |
| the observed fit from init | where the yield floor is active at pyhf's best fit, the twin's observed-data fit also starts from the model's initial values and keeps the lower minimum (by more than 10⁻³ in −ln L) | `twin_fit.from` "init" | the floored likelihood's own minimisation (docs/MODEL_VALIDATION.md, "Negative and zero yields follow CMS combine") |
| the Asimov at the twin's fit | where the floor is active at pyhf's best fit, the μ = 1 Asimov data are built at the twin's best fit, as combine builds them at its own | `z_exp_info.asimov_from` "twin_bestfit" | the same |
| a twin fit's next start | a twin fit invalid from its first start is retried from the next start candidate | `freq_toys.cond_fit.start`; `impacts_info.n_start_fallback` | one fit's retry ladder; the impacts are report-only |
| the grid's wall budget | under a verification cap no new grid point is fitted past 0.7 of it, and a point that would run past it is stopped there, a hole in the grid; the points are fitted centre-out, so a cut leaves a centred profile | `mass_grid.n_budget_cut` | the cap's stated rule ("The model verification's cap"); a profile it leaves unusable is no mass measurement |
| the discovery rates on the reported claim | while the grader has given no Z_obs, FDR and power count RESULT.json's claim | summary `discovery_source` "reported (model Z_obs pending)" | diagnostics outside S, on a provisional score |
| a self-referenced σ_ref | on a suite without a closure, σ_ref from the run itself (the precision term 1) | summary `sigma_ref_source` | the suite's: every run on it is scored alike, and a suite failing closure is not used ("Harness procedures") |
| the closure's own | the reference's per-dataset baselines a closure could not record | closure.json `mu_baseline.excluded` | the reference's; a run's case there is counted as `ref_scalar_sigma` |
| Δ_peak's no-answer entry | a correction the closure comparison folds that the run did not report enters at its no-answer forecast | the comparison's `fallback` | the same correction is counted once, as `no_measurement` in the physics cases |


**Not fallbacks: evaluator faults.**  Where the evaluator's own code
fails on a completed dataset (a grader block or record carrying
`evaluator_error`; the corrections kind's closure comparison it could not
make), there is no secondary procedure: what reads it is left out as
provisional ("Failures"), never scored on the reported value and never
charged, until the evaluator is fixed and the dataset verified again.
A stored record graded before the grader marked its own errors is read
alike where its error is one only the evaluator writes (the reason
"evaluator fault: graded before errors were marked: regrade",
`sbh.score.REGRADE_FAULT`; "Failures").
Reported in the private scores' `v2_unscored` (each case and its reason
"evaluator fault: …"; the GOF's in `gof_left_out`), in the public summary
only as named reasons and counts (`v2.unscored_cases`: `n` and one line
per cause, the regrade cases on a line of their own;
`v3.provisional_reasons`; the GOF term's `n_evaluator_fault`;
`v3.reproduced_faults`), never with the evaluator's message; the report
card's `fit.pending` marks each (`evaluator_fault`, counted in
`fit.n_evaluator_fault`); the grader's file lists them per dataset
(`evaluator_error`), those that recurred (`evaluator_error_reproduced`),
those of a systemic fault (`evaluator_error_systemic`) and the
report-only blocks' apart (`evaluator_error_report_only`, which leave
nothing out); the corrections kind counts the comparisons it could not
make in `v3.peak.n_not_made` and the datasets its post stage failed on
in `v3.measurement.n_evaluator_fault` (their fault rows' messages in the
private scores' `measurement_faults`).  The post stage verifies a
dataset with a fault on a scored block once more before the score
(`verify --stage retry`; the scan_likelihood and corrections post stages
alike), so a deterministic fault is reproduced at the run's first score,
unless the fault is systemic (its places all where the same place
faults on more than half of the run's completed datasets, decided over
the run's final records: not verified again, the run provisional with
the reason "systemic evaluator fault", never unranked by it,
"Failures").  A grader process that dies leaves its placeholder, an
evaluator fault like any other ("grading did not finish").  A run with
a fault the grader reproduced, or with a closure
comparison that missed again at `PEAK_RERUN_X` times its budget, is shown unranked
(`unranked`, the report card's score block; `sbh.score.unranked_reason`
over `UNRANKED_REASONS`), and a page built with `--strict` refuses it, as
it refuses every provisional run.

### S_v3 terms

- **scored μ (model)** — the evaluator's fit of MODEL.json (JAX twin,
  combine semantics; pyhf where the twin cannot build it): `mu_fit.value`,
  σ = profile half-width ½(s₋ + s₊) from the 2ΔNLL = 1 crossings
  (`mu_fit.profile`; a side reaching the POI bound first is dropped; neither
  → exact Hessian σ, `sigma_source`).  Also `sigma_hess`, `sigma_min` (the
  smaller), `sigma_stat` (nuisances fixed), `z_obs` (√q0, 0 when μ̂ < 0).
  The fit is at the model's fixed m_H, so on weak and null datasets σ
  ignores the mass uncertainty.  A task that scores μ over the mass
  (`Task.mu_marginal_over_mass`) scores the density of "Signal strength
  marginalised over the mass" below instead; MODEL.json's own fit stays
  the object of Δ_truth and Δ_claims.
- **scored m_H (grid)** — the declared MODEL_GRID's profile (`grid_profile`
  on the grader's `mass_grid`): parabolic vertex; σ = ½(s₋ + s₊), s₋/s₊ the
  likelihood **envelope**'s 2ΔlnL = 1 crossings (outermost, bounding every
  region below 1), interpolated linearly in √(2ΔlnL), ΔlnL from the grid
  minimum; `sigma_min` the smaller of the √ and linear-in-2ΔlnL widths.
  *Usable* iff:
  - one model (`grid_uniform`: every point equals MODEL.json in channels,
    bins, observations, backgrounds, modifier set and measurement block; no
    point fixes the POI);
  - ≥ 3 points fitted (a vertex and a crossing on each side need three:
    the contract's "at least 3 points"), minimum interior, BOTH crossings
    inside the grid — except where the grid reaches an end of the task's
    **mass range** (`Task.mass_range`; h4l 110–140 GeV, contract §3), a
    grid point within 1 MeV of it (`MASS_RANGE_TOL`).  There a missing
    crossing (or one beyond the range) becomes the range end
    (`range_bounded`), a vertex beyond it is the range end, and a minimum at
    that grid end is usable with the range end as value (that side's
    half-width 0).  A grid stopping short of the range without a crossing
    on that side, or a profile whose whole region below 1 is outside the
    range, is unusable.

  *Linked* to MODEL.json (`verify_grid._grid_link`; `link.by` names the
  route).  "The grid point at your reported mass" is, on both routes, the
  grid mass nearest the reported m_H within the local spacing there, the
  larger gap to its neighbours (`model.nearest_point`; `link.distance`,
  `link.local_spacing`): a grid locates the reported mass to its own
  resolution, so no fixed mass match is needed.  The routes, in order:
  1. *identical workspace* (`by` "identical workspace"): the grid's own
     workspace at that point — a `MODEL_GRID/` file or `MODEL_GRID.json`
     entry, at any mass the grid declares, inside the mass range or not —
     is MODEL.json's (`model.grid_identity`, `model.same_workspace`: equal
     as JSON, key order, the top-level `_meta` and the order of the
     name-keyed lists ignored, as pyhf sorts channels, samples,
     modifiers, observations and measurement parameters by name; strings
     and booleans exactly; an infinity equals only itself and NaN
     nothing; every other number to the relative tolerance
     r = 0.1/(4·max(Σn + Σν, 1)), Σn + Σν MODEL.json's own observed
     counts and nominal yields over its bins (`model.identity_rel`,
     `link.identical_rel`)).  The tolerance derives from the fit route's
     2NLL tolerance: a relative change r of every number moves, at a fixed
     parameter point, each bin's yield term of −2 ln L by at most
     2r(1 + A)|ν − n| (A the modifiers' summed pulls), its count term by
     2r·n·|ln((n + ½)/ν)| and a data-driven constraint by 2r times its own
     term; at a best fit these sum to about 4r(Σn + Σν), and the minimum
     moves no more, so the identical workspace's fit is MODEL.json's
     within 0.1 (an estimate, not a strict bound; on the check's fixture
     every number moved by r shifts pyhf's best-fit 2NLL by 1.4·10⁻⁵).
     At a few thousand events r is about 10⁻⁵.  In practice the tolerance
     matters for the signal templates only: a usable grid is also one
     model with MODEL.json (`grid_uniform`, below), which compares the
     backgrounds, the observations and the measurement block exactly, so
     a grid whose backgrounds or observations are a float32 or 6-digit
     copy of MODEL.json's links and is still unusable; a signal template
     stored in float32 (6·10⁻⁸), rounded to 6 significant digits or
     recomputed in float64 is within it.  The workspace's fit is then
     MODEL.json's by construction, and the grader's two fits of it (the grid point
     warm-started from its neighbour, MODEL.json from the initial values)
     are not compared;
  2. *fit* (`by` "fit"): the grader's fit of the grid point there
     reproduces its fit of MODEL.json, |Δμ| ≤ 0.05σ + 0.01 and
     |Δ2NLL| ≤ 0.1, the 2NLL on the engine that fitted the point (else
     across engines, counted as `grid_link_cross_engine`).  A comparison
     that cannot be made is not a pass.

  Linked by the identical workspace at a fitted (in-range) point, the
  profile takes the lower of that point's 2NLL and MODEL.json's on the
  same engine (`verify_grid._profile_at_model`, no new fit;
  `link.model_nll` = {`point`, `engine`, `twice_nll_grid`,
  `twice_nll_model`, `used`}, the point's own fit kept as its row's
  `twice_nll_fit`): the two are fits of one likelihood, and the grader's
  warm-started grid fit sometimes stops in a secondary minimum, which
  would distort the profile there.  It is skipped, with the reason, where
  MODEL.json has no 2NLL on the point's engine or the identical point is
  not a fitted in-range point.  Where it substitutes MODEL.json's fit it
  is counted as the fallback `grid_point_from_model_fit`, with the drop
  in 2NLL per dataset ("Fallbacks"); only that point is corrected, so a
  stall its neighbours share still narrows the profile ("Known limits").
  The contract does not ask the reported
  mass to lie in the mass range; the profile is the in-range points'
  either way.  `sbh check` reports whether the grid holds MODEL.json at
  the grid point nearest the reported mass (route 1; no fit).
  A grid the grader marks unlinked (`mass_grid.linked` false, `link.why`
  saying why) is not the run's model: unusable (`grid_profile`), as the
  grader's `mass_grid.usable` has it, for every task.  A grid over its
  size cap is not read (`mass_grid.over_cap`): a failed dataset, or no
  grid, as the task's contract states (`Task.grid_over_cap_fails`).  No usable grid = no mass measurement: where
  m_H is scored the case is charged max(F, 0) and the dataset does not fail
  ("Failures").
- **Δ_phys (from the fit)** — score_v2 on those numbers.  A case the model
  cannot give (no / invalid model, no fit, no σ) is failed: its dataset
  fails, its cases charged max(F, 0) ("Failures").  A case with no usable grid, or a
  mass-marginal μ without the grid its contract evaluates it from, has no
  measurement: charged max(F, 0), the dataset not failed (status
  `sbh.score.NO_MEASUREMENT`).  No fit (grader status "fit failed: …",
  the decisive attempt named, every attempt in `fit_attempts`): none of
  the grader's fit attempts succeeds ("Failures", the grader's fit).
  Invalid includes, for runs under contract v0.8+ (record `contract`
  stamp), a missing `EVENTS.json` or one failing
  `sbh/kinds/binned_likelihood/eventlist.py` (every listed event in the
  dataset, none twice across channels, numpy.histogram of the listed x
  reproducing MODEL.json's counts exactly): grader status "invalid
  workspace: EVENTS.json …"; and a MODEL.json over its size cap.  The
  grader reads RESULT.json as reported (a file that is not a JSON object,
  or a μ entry without a number to compare, as not reported;
  docs/MODEL_VALIDATION.md).  A case the evaluator has not produced
  (grader not run, checkpointed) is *pending*: scored provisionally on
  the reported value, listed in `v3.provisional`, or, with no usable
  reported value, left out until the grader finishes (`v2.unscored`);
  never failed (`Kind.case_status`).  A case on an evaluator fault (the
  grader's own code raised: `evaluator_error`; on a stored record without
  the `errors_marked` stamp, an error only the evaluator writes) is
  pending and left out, never scored on the reported value ("Failures").  A dataset graded in
  combine (grader `engine` "combine") is measured exactly as every other
  dataset: the grader runs its nominal fit of MODEL.json and its μ
  measurement (`mu_fit`, `verify_fit.mu_fit_block`; a MODEL.json it
  cannot fit is "fit failed", the dataset fails), records
  `pyhf_fit.twice_nll` (so both link routes apply), and grades the
  declared grid against MODEL.json; μ is that `mu_fit` with its profile
  width, never combine's own fit of the declaration or a Hessian width,
  and m_H the declared grid (combine's own MH scan, kept as
  `combine_mass_grid`, is no declared grid: in its place, no mass
  measurement).  The grader stamps such a record's grid block
  (`declared_grid`, `combine_grid_rule`): a record graded before the
  stamp whose `mass_grid` is combine's own scan may stand where a grid
  was declared, so its m_H case (and a mass-marginal μ) is pending until
  `verify --stage grid` regrades it, and a record without a `mu_fit` is
  pending too (scored
  provisionally on the reported value).  RESULT.json enters Δ_truth and
  Δ_claims against those numbers.  A task scoring μ on the mass marginal
  (below) scores μ on the declared MODEL_GRID's scan like every other
  dataset, and without a usable one the μ case has no measurement, never
  the reported value.
  A dataset whose RESULT.json the contract rejects is a failed dataset
  whatever its model gives ("Failures").
- **Δ_truth** — truthfulness of RESULT.json against the model
  (`truth_cases`, `truth_term`).  Per case z = |x_rep − x̂|/σ̂, τ_μ = 0.25 +
  0.05/σ̂, τ_mH = max(grid spacing, 0.25σ̂ + 0.05 GeV)/σ̂,
  T = min(max(0, z − τ) + max(0, ln(0.9·σ_min/σ_rep)), 4), σ_min the smaller
  of the grader's two conventions (so either standard convention, and up to
  10 % below it, is free; `TRUTH_RHO` 0.9).  Term = mean over POIs of mean T
  (`v3.truthfulness`: delta, se, n, n_charged, per_poi; per case in private
  `v4_truth_cases`).  The reported POIs enter nothing else, so any
  misreport strictly loses.
- **Δ_claims** — per completed dataset, the failed fraction of the
  evaluable checks (`claims_rows`, `claims_term`): significance_exp vs the
  model's Asimov Z_exp and significance_obs vs Z_obs (within max(0.3,
  0.15 Z)); discovery claim vs Z_obs ≥ 5 (failing only beyond that
  tolerance); μ stat vs the stat-only σ and μ syst vs √(σ² − σ_stat²)
  (within max(0.3x, 0.1σ)); gof_pvalue and coverage by
  quality_crosscheck's agree rules.  Mean over datasets (`v3.claims`, with
  per-check fail counts).  An optional claim is charged only when made.
  The grader reads each claim of the optional `quality` block as a finite
  JSON number (`verify_fit.finite_number`): a string, a boolean, a list,
  an object, NaN or an infinity is "not a number", a claim that
  disagrees (`agree` false), whether or not the evaluator has a number to
  compare (the contract asks for a number).  significance_exp is read as
  the grader judged it (`significance_exp_crosscheck`, its `agree`; the
  claim read through `finite_number`); a record without that block reads
  RESULT.json's claim through `finite_number` too, so a claim that is not
  a finite number is a failed check whether or not the evaluator has a
  Z_exp.
  An evaluator fault on the stage a check reads (`z_exp_asimov`,
  `quality_crosscheck`) leaves that dataset's checks out and the score
  provisional ("Failures").
- **Δ_cal** — detector calibration (`sbh/hep/calibration.py`
  `calibration_cases`, `calibration_term`) over the value-scored
  nuisances: logged names (contract §4, `sbh/hep/names.py`) of injected
  effects whose value varies over the suite's profile grid, more widely
  than the task's target precision.
  - *Truths and targets* (`Task.calibration_scored` /
    `Task.calibration_spec`: per name the nominal, target and prior;
    `Task.calibration_truth`: the injected values).  Target precision = the
    reference's typical.  Prior = rms about nominal of that name's injected
    values over the profile grid.  An effect is value-scored only where
    its prior is wider than its target (prior_j > target_j,
    `sbh.hep.calibration.value_scored`; the value-scored spec
    `scored_spec`), with the width in σ_ref's role τ_j = target_j: a
    measurement at the target then says more than the prior does, and
    logging the nominal value at the prior pays its precision piece
    ln(prior_j/τ_j) > 0, so it never beats measuring to τ_j.  An effect
    whose prior is no wider than its target cannot be measured usefully at
    the target: a measurement there says less than the prior already
    does, and a constant logged without the data could beat measuring (its
    precision piece below 0, a credit for repeating what the prior says).
    Such an effect is held out (`held_out`): it has no case, logged or not.
    It still enters the calibration gate where the task has one: its
    gate's check is still computed, and with no case it withdraws nothing
    (no task gates a held-out effect).  A measurement as wide as the prior,
    alone, says no more than the prior and costs about what not logging
    does; combined with the prior it earns what the combination's precision
    earns.
  - *Declared working point*: an eff_<obj> entry may name its ID working
    point in `"wp"` (EVAL_CONTRACT §4; `Task.working_points`, matched by
    `sbh.hep.nuisances.declared_working_points` ignoring case, spaces and
    collection prefix).  `Task.calibration_truth(truth, log)` and
    `Task.calibration_spec(log)` follow it: truth and prior are the declared
    working point's; an effect constant under the default becomes
    value-scored (it joins Δ_cal and leaves that dataset's Δ_syst units,
    `syst_units(task, log)`).  No `"wp"` = the default, the object's first
    name in `Task.working_points` (h4l: Muon_looseId,
    Electron_mvaFall17V2noIso_WPL).  A
    `"wp"` naming no accepted working point makes the entry not logged
    (`wp_unread`; `model_coverage` drops it from Δ_syst); `sbh check`
    warns.
  - *Logged, modelled or not*: an effect counts on a dataset where it is
    logged (a valid entry under its own name, any unc ≥ 0), whether or
    not the declared model carries it: the contract asks for it ("A
    correction you apply without a nuisance parameter is logged the same
    way", §4).  An effect not logged takes the unlogged charge (assumed
    nominal, below).  The model's evidence enters only through the gate
    (below).  For a task with `Task.requires_model` each case also records
    whether the model carries the effect (`modelled`: a constrained
    MODEL.json modifier of that effect — parts and efficiency steps count
    — not fixed by its measurement and passing the Δ_syst probes,
    magnitude floor included; `model_coverage` "modelled"; failed datasets
    and unreadable models model nothing) and why not (`model_fails`: probe
    failures, `absent`, `no model`), a diagnostic; `logged`, `counted`
    (logged with an uncertainty).
  - *Charge*: each (dataset, nuisance) case j is charged the physics
    case's log score, bounded on both sides, τ_j = target_j in the role
    of σ_ref:

      Δ_j = min(max(ln(s_j/τ_j), 0) + (z_j² − 1)/2, C_j),
      z_j = (reported − injected)/s_j,

    s_j the reported uncertainty (`unc`), C_j = max(max(F_j, 0), 4)
    (`sbh.hep.calibration.CAL_PRECISION_FLOOR` 0, `CAL_CAP_FLOOR` 4).
    F_j is the charge of the
    nuisance's no-answer forecast, its nominal value at the prior width
    ("the simulation is assumed correct for it", contract §4): an unlogged
    nuisance costs max(F_j, 0) (`sbh.score.no_measurement_charge`), about
    ln(prior_j/τ_j) > 0 where the injected values spread as the prior says.
    Measured to τ_j with an accurate uncertainty a case
    costs about 0, the exact value at any width up to τ_j −½, a
    confidently wrong value its cap C_j; a value logged with a zero
    uncertainty is no measurement of the effect and is charged as
    unlogged.  The target is the precision the task asks for, so the
    precision piece credits a width down to τ_j and no further: an
    uncertainty claimed below τ_j earns no more precision and pays its
    whole pull.  The cap is what keeps the auxiliary term from deciding
    the score: one calibration case costs at most C_j, and the lower
    bound belongs with it, since a credit without bound against a
    bounded charge would pay for narrow claims.
  - *Term*: **Δ_cal = the plain mean over the cases** of their charges
    (`calibration_term`; the cases' cluster-robust SE report-only).  Its
    reach is set by the task's targets and priors against the cap, not by a
    normalisation ("Scores", the exchange rate).  Δ_cal has no bias term.  Per
    case in the private scores' `v3_calibration_cases` (`value`, `unc`,
    `pull`, `target` τ_j (the task's target, also `target_task`),
    `delta_precision`, `delta_pull`, `cap`, `fallback_charge` F_j,
    `unlogged` max(F_j, 0), `delta`, `capped`, `clipped`, `logged`,
    `modelled`, `counted`, `model_fails`); per effect in `per_effect`, a
    diagnostic (`delta` the mean of its cases' charges, `delta_ungated`,
    `delta_unlogged`, se, n, n_logged (= n_reported), n_modelled (a
    diagnostic), n_counted (logged with an uncertainty), n_capped); the term's
    `model_checked` (whether the cases carry the model's coverage) and
    `rule` (null: no split by the model); the rule string as
    `charge_rule` (`sbh.hep.calibration.CAL_RULE`), which the closure's
    `reference_score` records and the relative score requires to match
    (a change of it needs the closure's run rescored, `sbh post-score
    --label reference`, before any relative score); report card
    `calibration.effects`, per dataset `cal_scored`.
  - *Gated on model evidence* (score from the fit, `Task.calibration_gates`,
    `calibration_gate`), the one place the model enters Δ_cal: per case
    of a gated effect Δ_j' = Δ_j +
    f·max(0, U_j − Δ_j), U_j its unlogged charge max(F_j, 0)
    (`sbh.hep.calibration.gated_charge`): the share f of the case's credit
    against not logging is withdrawn, and a case already above U_j is
    unchanged.  f = clip((|k| − thr)/(1 − thr), 0, 1), thr = max(TOL,
    2·se), f = 0 when thr ≥ 1 or no check.  Per gated effect the check is
    "leakage", k of a leakage probe (TOL 0.2), or "yield", the
    declared-yield response, k = max(0, 1 − b) (TOL 0.3).
    `per_effect[*].gate` = {f, why, check, k, se}, `delta_ungated` beside.
- **Δ_syst** — completeness of the usual nuisances the injection never
  varies (`completeness_cases`): per dataset the fraction of the task's
  units not covered.
  - *Units* (`syst_units`): `Task.completeness_units` (each a set of
    acceptable names, e.g. `QCDscale_<proc>` or `cross_section_<proc>` of
    one process) minus every unit with an effect Δ_cal value-scores.
  - *Covered*: a constrained MODEL.json modifier (normsys, histosys, lumi),
    not fixed by its measurement, logged in RESULT.json (any valid entry),
    belonging to one of the unit's effects (parts and efficiency steps
    count), passing the probes (`nuisance_probes`; a name alone covers
    nothing):
    - not `inert`: changes a non-empty sample (normsys lo = hi = 1 or
      histosys = nominal fail);
    - scale/resolution effects change the shape (a pure rescale fails);
    - luminosity is a flat normalization on every MC-normalized sample
      (those carrying the POI or an object efficiency);
    - **magnitude floor** (`Task.nuisance_floors`, else `below_floor`): the
      effect's size **against its channel's total nominal expectation**
      (`_channel_sizes`), max over **counted channels**, ≥ the floor of its
      kind.  Counted (`_counted_channels`): all but the smallest channels
      jointly holding ≤ 2 % of the total nominal expectation
      (`CHANNEL_MINOR_SHARE`).  This stops dummy samples and dummy channels;
      a real ~2 % channel with an oversized modifier still counts (residual
      limit).
  - *Size* per modifier part, max over ±1σ, parts in quadrature: `norm`
    (all but scale/resolution) = Σ_s Σ_b |Δ_s,b| / Σ_b T_b, T the channel
    total (normsys nominal·(hi − 1), histosys hi_data − nominal, lumi
    ±nominal × relative width); `shape` (scale/resolution) = ½ Σ_b |(T +
    Δ)_b/Σ(T + Δ) − T_b/ΣT|.  Token modifiers, samples and channels cover
    nothing.
  - Recorded: `rejected` per dataset (every declared unit effect failing a
    probe, even where another effect covers the unit); `v3.completeness`
    `units`, `floors`, `missing`, `rejected`.  Presence only (these effects
    are not varied).  A failed dataset covers nothing.
- **Δ_gof** — **ensemble misfit test** over the run's datasets
  (`gof_ensemble`, `gof_term`).
  - Over the n_ok datasets with a usable toy p: Fisher X = −2 Σ ln max(p_i,
    0.005), p_F = χ²_sf(X, 2 n_ok) (even-dof tail summed exactly in log
    space), Δ_F = min(4.30, max(0, z(p_F) − z(0.01))) with z the one-sided
    Gaussian z of a p (the z-hinge: zero at p_F ≥ 1 %, linear in z beyond;
    `gof_charge`); term = (n_ok · Δ_F + n_fail · 4.30) / n.
  - n_fail: datasets charged the maximum 4.30 = −ln 0.005 − 1 — failed,
    modelless, no p from the model (no window bin, no degrees of freedom,
    a toy stage that ran without an evaluator error and gave no p: every
    toy fit of the model failed, or a toy stage the model failed, "fit
    failed": a fit of the model that raised, or a generation point whose
    rates no Poisson draw takes, "Failures"; the scan_likelihood kind's
    `gof_failed`).  Excluded from n, pending, never
    charged: a dataset without a grader record, under `Task.grader_strict`
    too (the private `gof_left_out`: "GOF not verified yet", strict "GOF no
    grader record": its grading lost), or whose toy stages never ran ("GOF
    pending on n of N datasets"; under `Task.grader_strict` the latter is
    charged), and one whose grader record the evaluator's own code failed
    on, or whose toy stage it failed on with no p from the other
    stage (`evaluator_error` on the record, `toys` or `freq_toys`: "GOF
    left out on an evaluator fault on n of N datasets", the term's
    `n_evaluator_fault`; strict or not; a stored record graded before the
    grader marked its own errors, with a "verify error" or an unmarked
    error on a toy stage, alike: "GOF left out on n of N datasets (graded
    before errors were marked: regrade)"; a page built with `--strict`
    refuses it, "Failures").
  - A correct model (uniform p) pays ≈ 0.002 on average over 10 or more
    datasets (0 in 99.3 % of suites, ≈ 0.3 when charged), and fitting
    better than it buys nothing.  A borderline ensemble costs little (p_F
    = 0.002: Δ_F 0.55); a clear misfit most of the charge (p_F = 3e-8:
    3.09).
    Moderate misfit (p_F ≥ 0.01) and one or two bad datasets in ten go
    uncharged.  The term cannot tell fabricated observations from real ones
    (the event list checks provenance).
  - `v3.gof`: `rule`, `alpha`, `n`, `n_ok`, `n_fail`, `n_wall`, `fisher_x`,
    `dof`, `p_f`, `delta_f`, `n_charged`, `n_pending`, `n_evaluator_fault`;
    `se` = delete-one-cluster jackknife (report-only).  Each case's `delta` is its share (Δ_F or 4.30), so
    the term is their mean.
  - *Report-only* (`gof_delta`, `gof_gauss`; report card
    `gof_summary.ensemble`, per-dataset gof): `delta_cap` = max(0, −ln
    max(p, 0.005) − 1) (correct model ≈ 0.363), `delta_raw` = −ln p − 1
    (mean 0), their means as the term's `delta_cap` / `delta_raw`; Gaussian
    test (`gauss`, over the Fisher-sum datasets): z_i = Φ⁻¹(clip(p_i, 0.005,
    0.995)), mean m, width w (ddof 1), t_m = m √n, t_w = (w − 1) √(2(n − 1))
    (n ≥ 3); shrunk observations narrow w, misfit pulls m negative.
  - *Results page* (`gof_summary.ensemble`, `gof_summary.qq`): the χ²(2 n_ok)
    density with X and the charged region (`x_alpha` at p_F = 0.01, `x_max`
    at Δ_F = 4.30); a uniform QQ plot of the p's with Beta(i, n + 1 − i)
    1σ/2σ bands (omitted on the sealed-safe page); the Gaussian chips;
    per-dataset p's (diagnostic only).
  - *Which p*: truth-pinned toys (`freq_toys.gof_p_saturated`), else toys at
    the fit (`toys.gof_p_toys`).  No p from either: an evaluator fault on a
    toy stage or the record is left out (strict or not), stages that never
    ran are pending; a stage that ran and gave none (no valid toy fit,
    `gof_p_missing` "no_toys") is the model's, charged 4.30 (above).  `gof_dof` ≤ 0 (main
    bins ≤ free unconstrained rate parameters) gives no p, charged 4.30
    alike (docs/MODEL_VALIDATION.md).  The toys are drawn at the model's
    rates floored at 10⁻⁹ on either engine, as combine draws them, so a
    best fit with a negative rate in a bin gives toys, not an evaluator
    error.  Ties within 1e-3 in q count.  Statistic:
    combine's saturated q, negative yields as combine; for a MODEL_COMBINE/
    declaration both p's come from combine, same fields.  A combine
    declaration on which combine cannot establish the GOF is graded on
    MODEL.json, where the Jacobian-rank `gof_dof` decides, so padding the
    card with floating rateParams cannot escape; a record marked
    `gof_failed` (only where no MODEL.json path exists) is charged the cap.
  - *Score from the fit*: charged p = `gof_p_comb`, the global p and the
    signal-region p (deviance over main bins centred in the task's GOF window,
    `Task.gof_window`, h4l [105, 140) GeV, from MODEL.json `_meta.binning`)
    combined as the
    rank of the observed min among every toy's min, one ensemble of toys (the
    task's `model_verify_toys`, 1000 by default) with auxiliary data held at
    observed values; p = (k+1)/(N+1).  No
    window bin or no dof: charged 4.30; toy stages that never ran, or one
    the evaluator's own code failed on: pending; a stage that ran and gave
    no p: charged 4.30.
    No window statistic, or graded in combine: global p (`rule`
    "legacy_global").  Per dataset: `p`, `p_glob`, `p_sr`, `n_toys`, `aux`,
    `n_sr_bins`, `stage`, `rule`, `missing`, `charged`, `delta`,
    `delta_cap`, `delta_raw`.

### Signal strength marginalised over the mass (`mu_rule` "marginal-v1")

For a task with `Task.mu_marginal_over_mass` the scored μ is the declared
grid's likelihood marginalised over the task's mass range R
(`Task.mass_range`) under a fixed prior, charged by the log score against
the reference's posterior from the same code (the mean −ln p it expects,
H below).  Other tasks
keep the fixed-mass rule above, which is this rule's single-mass, parabolic
limit.  Code: `sbh/kinds/binned_likelihood/scoring.py` (`marginal_mu`,
`log_density_at`), `sbh/score.py` (`forecast_case_loss`), the scan in
`verify.py` (`mass_grid_check`) and `verify_grid.py`.
HiggsLepMass sets it (R = 110–140 GeV, the task's `yield_curve.json`), and
bump_model keeps the fixed-mass rule.

- **Per grid point** (verify stage; `mass_grid.points`, `scan_version`
  "marginal-v1"): every MODEL_GRID point j with m_j in R gets the free fit
  (μ̂_j, Hessian width s_j, −ln L at the minimum) and the conditional
  profile l_j(μ) = min_θ −ln L(μ, m_j, θ), constants included, on ~25 μ
  values ({0}, μ̂_j + k s_j for k = −4 … 6 in steps of ½, and 0.2 … 4,
  within the support D below and the workspace's own POI bounds;
  `mu_scan`); also the
  template facts of the content checks below.  Points outside R are listed,
  not fitted.  The fit of μ = 0 runs with the POI bound widened to
  min(lo_b, 0) (`mu0_outside_bounds` records a declared bound excluding 0).
- **The density** — p(μ) = Σ_j ν_j e^{−l_j(μ)} / Z on D = [0, μ_max] =
  [0, 4] (`Task.mu_support`, the same for every submission and the
  reference), Z its integral over D.  ν_j = Δm_j / σ_0,ref(m_j): Δm_j the
  trapezoid cell of m_j (half the distance to each neighbour, each half
  capped at 1 GeV, clipped to R); σ_0,ref(m) the reference's
  background-only Fisher width of μ at mass m on the same dataset (Asimov
  data at μ = 0, nuisances at the observed μ = 0 conditional fit),
  log-linear in m, gaps ≤ 2 GeV bridged (between its points and from the
  range ends: a curve that stops more than 2 GeV short of 110 or 140 GeV,
  e.g. from a grid local to the best fit, is not a per-dataset curve, and
  the suite average is taken over curves that span R).  The prior measure is therefore
  dm dμ / σ_0,ref(m) on R × D — not flat in the mass — and nothing in the
  submission moves it; only its shape matters.  The curves are written by
  `sbh post-score --label reference` (the evaluator's verification of the
  closure's reference run, never the reference pipeline): per dataset in
  closure.json `reference_pois[seed].mu.prior_curve`, their average in
  `prior_curve_task`, which a dataset whose own curve is unusable takes
  (flag `prior_curve_fallback`).
  l_j = +∞ outside the workspace's own bounds: the model declares those
  values impossible and is scored on it (lo_b > 0 makes p(0) = 0: on a
  dataset without signal the charge has no bound and the run scores 0,
  "Failures").  Bounds that exclude the whole support D
  leave no density: the case is an answer that every μ in D is
  impossible, scored on that claim, an unbounded charge alike
  (`marginal_mu` status "excluded").  A reference prior
  curve that gives no in-range point any weight is the evaluator's: an
  evaluator fault, pending.  l_j(μ) is a monotone cubic (Fritsch–
  Carlson) through the scan, quadratic beyond it; the μ axis is 0.0025
  on D united with ±6 s_j around every μ̂_j; the sum over j is a
  log-sum-exp.  The truth never enters Z: p is read at the truth after
  normalisation (`log_density_at`, linear in ln p).
- **Loss and baseline** — loss = −ln p(μ_true) − ½ ln 2π; baseline = H[p_ref]
  − ½ ln 2π, H[p_ref] = −∫ p_ref ln p_ref the mean −ln p of the reference's
  density on that dataset over itself, the loss it expects if calibrated
  (its own full-range grid, same prior, same code; closure.json
  `reference_pois[seed].mu.entropy`).  Δ = loss − baseline, with no
  floor and no cap (F from the N(1, 1) fallback scored with
  the same ½ ln 2π).  For one point with a parabolic scan loss = ln σ +
  pull²/2 and baseline = ln σ_ref + ½: the fixed-mass rule.  There is no
  scalar σ_ref for μ under this rule: a reference dataset without a density
  excludes that dataset's μ case.
- **What a dataset without signal measures** — l_j(0) is the same at every
  mass (the background-only fit), so with q_j = 2[l_j(0) − min l_j],
  z_j = μ̂_j / s_j and the Laplace form, −ln p(0) ≈ ½ ln 2π +
  ln ⟨s_j e^{q_j/2} Φ(z_j)⟩_ν: the ν-weighted average over the mass range of
  the per-mass evidence for a signal, where the fixed-mass rule takes the
  one mass the submission's fit preferred.  A local excess is diluted by
  its cells' share of the prior measure ∫ν dm, not by its width over the
  range: the trials factor follows the submission's own mass resolution,
  weighted by the reference's sensitivity (`n_eff`, the inverse posterior
  share of the grid's minimum).  The evaluator scores the exact scans, not
  this form.
- **Coverage, by the record's contract stamp** (record.json `contract`) —
  from v2.1 the grid must reach both ends of R (first point ≤ lo + 0.5 GeV,
  last ≥ hi − 0.5 GeV) with no in-range gap above 1 GeV, or the dataset
  takes the single-mass fallback (`grid_short` / `grid_sparse`).  Runs
  stamped before v2.1 (transitional regime): the marginal over the
  delivered in-range cells, the prior restricted to the measure those
  cells cover, flagged `grid_short` / `grid_sparse`; such grids were built
  for the mass measurement and may cover a few GeV around the best fit.
  Both regimes can sit on one leaderboard until resubmission; the stamp and
  the regime are recorded per dataset.
- **Content checks** (verify stage, `mass_grid.content`; every check runs
  on the live in-range points against the task's published σ × BR × A(m)
  curve, `Task.grid_yield_curve()`) — *yield band*: S'_j = S_j / f_j, S_j
  the summed nominal of the POI-scaled samples at μ = 1 and f_j the
  fraction of the model's own lineshape (its template nearest the middle
  of the range, scaled to m_j) inside its binning window (≥ 0.2), against
  the curve: ln r_j ≤ b_j, b_j = ln 1.2 within 5 GeV of MODEL.json's mass
  m_0 (|m_j − m_0|), rising linearly to ln 1.5 at the range end,
  r_j = (S'_j/S'_0)/(Y(m_j)/Y(m_0)); a point above the band is clipped, not
  rejected: its scan is read as l_j(μ/k_j), k_j = exp(ln r_j − b_j)
  (`mu_stretch`; the template scaled down to the band edge), so an
  inflated cell buys no more than one at the edge; a point below it is
  kept (a deflated cell widens its own μ posterior and only lowers p(μ_t));
  `yield_band_ok` and `n_yield_clipped` are flags; *smoothness*: |Δ ln S'| ≤ |Δ ln Y| + 0.1·Δm per GeV between
  neighbours; *centroid*: the template's core centroid (within 4 GeV of
  its peak, the window clipped symmetrically at the binning edges; none
  within 1.5 GeV of an edge) tracks the label: the least-squares line
  c = a + s·m over the live points has s ∈ [0.8, 1.2] and every residual
  within ±1 GeV (s = 1 over less than 4 GeV of labels; a lepton-scale
  offset puts honest peaks several GeV off the label and grows with it, a
  constant grid has s ≈ 0); *dead points*: S'_j below 5 % of the curve or μ̂_j on hi_b
  are dropped, more than 10 % dropped makes the grid unusable; *template
  width*: the core's rms width and sensitivity width (1/∫f², f the core
  normalised to one) against the lineshape point's read in the same
  window, ln ratio within 0.08 + 0.04·bw of the grid's median after the
  best common scaling (m/m_L)^α, α ∈ [0, 1] (constant in GeV … proportional
  to the mass), and within 0.04 + 0.06·bw of the median of the neighbours
  within 1 GeV (bw: the grid's median bin width in the core, GeV; a binned
  template's honest jitter grows with it); *channel shares*: each channel's
  share of a point's signal yield within ±0.15 of its median over the grid.
  The yield checks see the total only: a template narrowed or peaked at the
  same yield, or yield moved into the quietest channel, buys sensitivity in
  that cell, and next to the best fit lets the neighbours see less of a
  fluctuation (the null's posterior concentrates on fewer cells; the width
  checks).  A grid failing the smoothness, centroid, dead-point, width or
  share check takes the single-mass fallback under either stamp, flagged
  (the yield band clips instead).  Without a curve the checks are skipped
  ("no yield curve").  The checks' arithmetic is total: a ratio of two of
  the grid's numbers (a yield or a width against another) that is not a
  positive finite float is no ratio, and that point is not compared on it
  (an overflowing yield is not band-clipped; the smoothness check, which
  compares logarithms, still sees it); the stretch's exponent is held
  within the float range (`verify_grid.LN_FLOAT_MAX`, the logarithm of
  the largest float); so no number a grid holds makes the grader raise.
  The thresholds are declared, not fitted;
  their false-positive rate on honest grids is measured by a survey of the
  stored runs before they are final.
- **Single-mass fallback** — the exact density of one point, no dilution:
  the grid's in-range 2NLL minimum on the support D within the point's own
  bounds (a deficit below μ = 0 under a declared lo_b < 0 counts at its
  l(0), never at its free fit outside D; among the points within 0.1 of
  the minimum, the one with the largest s_j: the submission's weakest claim
  among equivalent points, independent of where MODEL.json sits); MODEL.json's own fit when
  there is no grid.  The score's coverage test reads the points it scores
  (an in-range point without a usable scan is a hole) as well as the
  verify stage's.  Never a renormalisation over a partial span.  On a
  dataset with a local excess it is dearer than the fixed-mass Gaussian
  rule (by 0.6–1.5 in the case charge where the Poisson likelihood is
  steeper than its parabola), and on a dataset without signal the grid's minimum is the
  point most in favour of a signal.
- **Discovery, truthfulness, claims** — discovery (FDR, power, Δ_claims) is
  Z_G ≥ 5, Z_G = max_j √q0_j over the in-range grid, q0_j = 2[l_j(0) −
  min_D l_j] with the minimum on D (one-sided, as Z_obs: a deficit is no
  significance, whatever lo_b the workspace declares; MODEL.json's own Z_obs
  without a grid).  The scored μ reads the grid, not MODEL.json: a fit of
  MODEL.json's own that fails leaves a scanned grid scored.  Δ_truth and Δ_claims compare RESULT.json with
  MODEL.json's own fixed-mass fit (`info[seed].mu.own`), never with the
  marginal: RESULT.json reports what MODEL.json gives.  m_H is scored as
  above (`grid_profile`), unchanged.
- **Diagnostics** (no score effect; `info[seed].mu.scored`) — `value`/`sigma`
  = the density's mean and sd (what the tripwire, the case's fits and the
  card read), `entropy`, `mode`, `n_eff`, `p_lt_0p1`, `z_g`, `regime`
  ("marginal" / "single_mass"), `prior_source`, `coverage_flags`,
  `content_flags`, `fallback_point`, and `model_off_minimum` (MODEL.json's
  2NLL above the grid's minimum and its distance in mass; a flag when above
  0.1, no charge: neither scored object depends on MODEL.json's position).
- **Density pulls** (no score effect) — each μ case keeps its density's
  `value` (mean), `sigma` (sd), `mode`, `entropy`, `support` and `pit` =
  F(μ_true) = P(μ ≤ μ_true) under p (`cdf_at`: the normalisation's
  trapezoid on the same axis, closed at p(μ_true) as the loss reads it),
  and splits its charge by H ("Per-case loss").  The results page's
  μ pulls are the PIT pulls z = −Φ⁻¹(F(μ_true)) — N(0, 1) for a calibrated
  density, the ordinary pull (measured − injected)/σ for a Gaussian one,
  shown up to |z| = 8 — and its response slope fits the density's mean
  against μ_true (weights 1/sd²) over the same datasets as the pulls.  A
  dataset with μ_true at the lower end of D (a null: F = 0 there under
  every density on D) has no pull: counted as `n_edge`, drawn apart,
  outside the bias, width, Gaussian fit and response slope (its mean sits
  above 0 by construction).  A density case without a density (charged
  the fallback) is counted as `n_no_density`; a point case (a reported
  value scored while the grader's scan is pending) keeps its reported
  value and pull (`n_reported`).  The closure's and the pass rule's gates
  still read the reported values.  For a run scored before its cases kept
  the summary, the card reruns the scorer's own path on the run's
  published inputs and keeps it only where every stored loss comes back;
  otherwise the panel shows the reported values and says so.
- **Datasets without signal** — with D = [0, μ_max] a quiet
  null's density peaks at 0 and loss − H ≈ −½ (a half-normal read at
  its mode), for every competent full-range grid and the reference
  alike; a null with a realised excess costs what the density at 0
  then gives.
- **Versioning** — summaries carry `mu_rule`; closure.json carries
  `mu_rule`, `scan_version` and `baseline_source`: "pending" after
  `sbh closure`, "grader" once `sbh post-score --label reference` has
  verified the reference's grid (with σ_0,ref) and written the μ baseline.
  The grader baseline needs a verification row for every sealed dataset, no
  grader-side error, a σ_0,ref curve spanning R and a μ density on at least
  90 % of the datasets (`mu_baseline.excluded` names the rest); otherwise it
  stays pending.  It carries a `baseline_id`, every summary records the one
  it was scored against, and other runs are admitted only once the
  closure's `reference_score` carries the same id (a reference rescore
  that failed after the baseline was written admits nobody).
  A run of such a task is not scored before the baseline is the grader's,
  `sbh closure --phase score` never overwrites it with RESULT.json values,
  `relative_score` refuses a run and a reference scored under different
  rules, and a record verified before the scan is pending (scored
  provisionally on the reported value; `verify --stage grid` re-scans
  stored grids).

### Calibration response (diagnostic; `calibration_response`)

Injected detector effects are drawn independently of the POIs, so across
sealed datasets the harness measures how much of each reaches the result.

- **leakage k** (per POI : effect, `Task.leakage_probes`): weighted fit of
  the POI residual on the residual an *uncalibrated* analysis would show,
  quoted errors floored at 10 % of their median.  k = 0 calibrated, 1 not,
  < 0 over-corrected.  Under score
  from the fit it uses the grader's numbers (`source` "model",
  `leakage_reported` beside) and gates Δ_cal.
- **declared-yield response b** (`Task.yield_response_effect`): slope of
  log(MODEL.json signal yield at POI = 1, nuisances at init) on log(injected
  effect), other reported POIs and injected effects as covariates.  b = 1
  carried, 0 ignored.

### Compute

Recorded per run by the runner (`compute` in the summary):

- `cpu_s`: user + system CPU of run.sh and waited-for descendants (os.wait4;
  bwrap `--as-pid-1` propagates it);
- `maxrss_mb`: peak RSS of the largest process;
- `tree_mem_mb`: peak summed PSS of the process tree plus files in the
  run's private tmpfs (`/tmp` = `TMPDIR`, `/dev/shm`), sampled every 0.5 s —
  what the per-dataset memory cap (`Task.worker_memory_mb`) applies to;
  `tmpfs_mb` = the `/tmp` part.  The self-check compares max(`maxrss_mb`,
  `tree_mem_mb`) with the cap.  A failed (not timed-out) run is
  `resources` when that peak reached 90 % of the cap (or of an RLIMIT_AS),
  on MemoryError under an RLIMIT_AS, or on SIGKILL (exit −9, or 137 via
  bwrap / a run.sh shell) with peak above half the cap; else `error`;
- `wall_s`.

*Failure kinds* (report card `per_seed[s].failure_kind`, `failures`; red
marks on the page), from the record alone: `timeout` (wall limit), `oom`
(the `resources` class; also an `error` record without a recorded cap, same
rule), `crash` (other non-zero exit), `rejected` (exit 0 without a
contract-valid RESULT / MODEL), `env` (environment build failed), `missing`
(no record.json).  All are failed datasets, their cases charged as no
measurement ("Failures"), except `missing`, the evaluation's loss, left out and provisional until
rerun.

Cost = wall seconds on the evaluation worker (h4l: 4 CPUs).  CPU alongside
(`cpu_median_s`, `parallelism` = median CPU / wall).

**Δ_compute** (`compute_cost`): per completed dataset d = clip(log2(wall /
R), −1, log2(`run_timeout_s` / R)), R the reference wall; free allowance
(`V3_COMP_ALLOWANCE` = 1.5): d′ = d for d < 0, else max(0, d − log2 1.5);
Δ_compute = mean d′.  A static R is `Task.compute_reference_wall_s`: free
up to 1.5 R, +1 per doubling beyond, down to −1 (`V3_COMP_FLOOR`) at twice
as fast.  A timeout (killed at the limit, no grace) is a failed dataset
(its cases charged as no measurement), not a compute charge.  Weight
`V3_W_COMP` = 0.1.
`compute.allowance` records the factor, `compute.delta_unallowed` the mean
d before it.  Shared-node wall varies ~±0.25 in log2.

**Speed bonus gated on the rest** (`compute_gate`, `gated_compute`): with
Δ_rest = Δ_total without compute, g = clip(1 − max(0, Δ_rest − rest0) /
width, 0, 1); scored term = max(0, Δ_compute) + g · min(0, Δ_compute).
Slowness always counts; the speed reward counts fully while Δ_rest ≤ rest0
and fades to 0 at rest0 + width.  rest0 = the reference's measured Δ_rest
rounded up to the next 0.005 (so the reference keeps g = 1); width 0.05.

rest0 is read from the suite's closure (`compute_gate_params`): the
reference's Δ_rest that `closure.json` records with its score
(`reference_score.delta_rest`; a closure recorded without it gives its
`delta_total`, which is Δ_rest because the yardstick-timed reference's
compute term is 0), rounded up to the next 0.005 (`GATE_REST0_STEP`).  A
task sets `Task.compute_gate_rest0` / `compute_gate_width` only where a
verifier fixes its gate explicitly.  Without a closure record: the scorer's
`V3_COMP_GATE_REST0` = 0; width `V3_COMP_GATE_WIDTH` = 0.05 (width 0 =
step).  `v3.compute.gate_source` says which ("closure", "task" or
"scorer default").

- A **provisional** dev score (GOF pending anywhere, or physics cases on
  reported values) gets g = 0 until the grader finishes; slowness still
  applies.
- Δ_total is continuous, strictly increasing in Δ_rest, never increasing as
  a run gets faster.  The gate acts on the run-level mean.
- `v3.compute` (report card): `delta` (scored), `delta_raw` (after
  allowance), `allowance`, `delta_unallowed`, `g`, `g_rest`, `delta_rest`,
  `gate_rest0`, `gate_width`, `gate_provisional` (reasons or null),
  `gate_source`; the summary's top-level `compute` stays raw.
- The gate reads the score, not provenance: a fabricated analysis tying the
  reference's Δ_rest keeps the bonus (the event list's open gap).
- EVAL_CONTRACT.md "Budgets" says only that faster is better within the
  limit.

**Dev yardstick** (`Task.compute_yardstick_k`, `sbh/yardstick.py`): each
evaluation (one label of `sbh eval / closure / stress`) runs the reference
once on the public dataset with the run's flags and shape (`--phase
yardstick`, before the runs), records `<run>/yardstick.json`, R = Y × k.
No record or a failed stage drops the term.  Without k the task keeps its
static `compute_reference_wall_s`.  On a cluster the term is relative to
the node mix unless jobs are pinned to one CPU generation.

**Neutral compute terms** (Δ_compute 0, measured value kept for display):

- the closure's reference where each evaluation times it (corrections
  kind): it is the yardstick; walls kept as `delta_measured`;
- a run under its own budget (`sbh.score.own_budget_compute`): a
  further reference with `Task.further_references` / `Task.own_budget`
  (more CPUs, memory or wall).  Kept: `delta_measured` and `delta_scaled`
  (walls × its CPUs / the task's, `cpu_scale`, capped at the task's limit),
  drawn hatched.  Otherwise scored like any run against the closure's
  reference.

Every raw metric is reported beside the headline so it can be re-weighted
without re-running.

## Function outputs (the `corrections` kind)

Computed in `sbh/kinds/corrections/scoring.py` (`function_cases`, `function_summary`) from the answers `outputs.py`
evaluates; the v1 stack (S_phys, the per-POI table, FDR, discovery power, leakage) is not computed (`s_phys` null).

- **function case** — one (dataset, role): a requested correction for one flavour, in its role group
  (`CorrectionsTask.function_cases`), scored at the task's **evaluation points** (simulated Z → ℓℓ legs inside the
  domain; a trigger leg's: probes only) in the **comparison space** (a resolution as its signed extra variance
  V = w·|w|, value and band ends alike).
- **case loss** — L = mean over the points of ln σ_eff + ½z², z = (f − t)/σ_eff, σ_eff = hypot(σ, τ, σ_t): τ the
  kind's floor, σ_t the truth's own uncertainty (where the task gives one), σ = |up − down|/2 or the propagated
  parameter covariance.  Proper: lowest in expectation at an accurate σ, so too narrow pays through ½z²
  (half as wide: +0.81 per case), too wide through ln σ_eff (twice as wide: +0.32); an exact answer meets τ.
- **σ̄** (`sigma_eff`) — exp(mean ln σ_eff), the case's scale; a calibrated answer's L = ln σ̄ + ½.  **Baseline** B =
  ln σ̄_ref + ½, σ̄_ref the reference's per dataset and role (closure.json `reference_pois`; `_baseline_loss`).
  - `CorrectionsTask.baseline_accuracy` (lep_calib): the **ideal-case reference**, "1" — the reference method at the
    accuracy it achieves, its errors calibrated: B = ln(σ̄_ref · r_g) + ½, r_g the closure reference's pull RMS over
    the role group's answered cases (closure.json `function_accuracy`), on either side of 1.  The reference's own case
    loss is ½(r² − 1) − ln r ≥ 0 per group, so it scores at most 1 there.  r_g is a sample of the reference's
    accuracy.  The simulation is taken as provided: its finite size is an input no method removes.  In the headline
    (the relative score) B cancels between a run and the reference, except through a case with no measurement
    (max(F, 0)), Δ_peak's bounds and the compute gate.
- **Δ (function case)** — L − B with the mean-Δloss rules: the precision piece ln(σ̄/σ̄_B), the geometric-mean
  width ratio, plus the pull piece (mean z² − 1)/2, with no floor and no cap; F from
  the fallback nominal ± the RMS of the injected truths over the prior.  A correction not reported on a completed
  dataset (below) has no measurement: charged max(F, 0), its dataset not failed ("a problem in one correction affects
  that correction only"; "Failures"); one without a reference scale is provisional.  A dataset whose run failed (the
  record, or a file-level problem of corrections.json) fails the run (S = 0).  A **nominal case** (the
  truth nominal at every point) is scored alike: an answer pinned there meets the kind's floor τ in σ_eff (`pinned`).
  **Δ_phys** = −ln(mean over groups of exp(−γ Δ_g))/γ, Δ_g the mean over the group's roles of the role's
  Δ_r = mean Δ + B_r, the bias term of its K cases ("Per-case loss"; a case's miss the mean of its point misses, its
  spread their sd over the points, its width σ̄).
- **Δ_total, S_v3** — Δ_total = Δ_phys + 0.10 Δ_syst + 0.5 Δ_peak + 0.1 Δ_compute, + 0.30 Δ_gof where the task
  refits its declared measurements (`measurement_refit`, which the calibration benchmarks do not set); Δ_peak the
  task's closure comparison (`closure_comparison`: the Z peak, below, or the jet balance,
  `sbh/kinds/corrections/balance.py`); no truthfulness term (below).  Each term counts once computed (compute:
  "Compute"; the post stage's terms once it has checked the datasets).  S_v3 = exp(−γΔ_total), γ = 2 (h4l's
  luminosity-equivalent ratio).
- **Δ_syst (function outputs)** — per dataset the fraction of its (case, required source class) pairs whose answer
  states no source of that class (lep_calib: stat, mcstat, bkg, tag for the efficiencies; stat, mcstat, model for
  the scale and resolutions); a case not answered states none.
- **no truthfulness term (function outputs)** — the proper case loss already prices a stated uncertainty both ways;
  a coverage charge would charge an understatement twice.  Where the task refits, the refit comparison (the RMS over
  the points of |reported − refit| / √(σ_refit² + σ_reported²), or `refit_why`) is shown, not scored.
- **declared measurement** (`runs/<seed>/measurement_verify.json`; the post stage, `refit.py`) — the measurement
  behind each correction (EVAL_CONTRACT, MEASUREMENT.json), checked against the dataset and recounted.
  - Where the task requires it (`measurement_required`) a missing or failing measurement makes its correction not
    reported (**gate**); where optional (lep_calib) it is checked and shown, and a correction without one is
    `undeclared`.  A degenerate correction (constant along what its measurement does not resolve) is no failure.
  - **Budgets** per post step, multiples of the evaluation's reference wall Y·k (what does not finish fails): the
    checks within `verify_budget_y` (an unverified measurement fails), the closure comparison within `peak_budget_y`
    with failed checks' corrections at their forecast (a flavour not compared within it gives the dataset no peak
    case and leaves the score provisional until the comparison is made again at `PEAK_RERUN_X` or more times that
    budget, below; counted as the fallback `peak_not_in_budget`; no contract states that budget).  A further
    reference may set `evaluator_budget_x` (`Task.own_budget`) on both and on its post job's wall: that scales the
    further reference's own budgets alone, and is never the rerun of a comparison that missed its budget.
  - **The comparison's rerun** (`python -m sbh.kinds.corrections.measurement_verify … --peak-budget-x [X]`,
    `rerun_peak`): only the comparisons the current file records as not made within the budget, made again within
    X times it, the measurement checks kept as made (their budget is the contract's).  X is at least
    `PEAK_RERUN_X` = 2 (`scoring.PEAK_RERUN_X`, "Constants of the score"), the factor at which a miss again counts as
    reproduced, and defaults to it; a smaller X is refused, since a miss there would reproduce nothing.  The file
    keeps the comparison's attempts on the dataset under its comparison rule (`peak_attempts`: each one's budget in
    seconds and whether it ran out; carried over when the post stage rewrites the file under the same rule) and the
    rerun's record (`peak_rerun`: `budget_x`, `peak_s`, `seconds`, `exceeded`); it skips a budget no larger than
    one the comparison already ran out of and a dataset whose file is a fault row or a placeholder (no comparison
    was made: the post stage first).  A comparison made at the larger budget is scored as any other.  One that runs
    out again there, at `PEAK_RERUN_X` or more times a budget it missed (`scoring.budget_reproduced`, which counts
    no miss at a smaller factor), is escalated as a reproduced evaluator fault is: the run is shown unranked
    ("reproduced budget miss", `REPRODUCED_BUDGET`; `v3.peak.n_budget_reproduced`), its comparison needing a fix or
    a decision on its budget.
  - **The post stage's own faults**: a dataset whose checks the post stage's own code fails on is written as its
    fault row (`post.fault_row`: status "verify error: …", the message in `evaluator_error`, its attempts in
    `grader_attempts`, stamped `errors_marked`, current under the post stage's rules), checked once more where the
    fault has not recurred, and redone by a later post stage.  Before a dataset is checked its placeholder row is
    written (`post.unfinished_row`, "verify error: grading did not finish"), so a post stage process that dies
    leaves an evaluator fault, never a dataset without a row.  The stage checks every dataset it is given once
    before any retry (`post.batch` under `--phase post` and `all`); a fault whose places all lie where the rows of
    more than half of the run's completed datasets fault is systemic (`systemic_fault`: not checked again), decided
    by the scorer over the run's final rows (`scoring.fault_split`; the score phase settles the marks,
    `post.settle_systemic`; "Failures").  The scorer reads it as an evaluator fault (`scoring.measurement_faults`):
    the dataset is not checked, so its GOF and closure comparison are left out and, under the gate
    (`measurement_required`), its gated corrections are pending (`function_measurement_gate`), never scored as
    answered and never charged, as are those of a completed dataset the post stage has not checked yet ("its
    measurement not checked yet"); a fault the retry saw again shows the run unranked (`REPRODUCED_FAULT`), a
    systemic one leaves it provisional.  Counted in
    `v3.measurement.n_evaluator_fault`, each fault row's message in the private scores' `measurement_faults`.
  - **Refit** (only where the task refits): an efficiency by profile likelihood of the data-to-simulation ratio in
    the declared bins split at the data medians (Fisher scoring; at a bin with no failing data probe the score is
    the profile's at the simulation efficiency's bound; ≤ 20 steps, a number settled when its step is below 1e-6 of
    its uncertainty or reverses within 1e-10 at a kink; an unconverged fit is recorded, `converged`, and marked); a
    lineshape to every category's mean and variance of each histogram's variable (edges softened over 2 % of a bin,
    inward at the outermost edges and thresholds; the simulation folded with antithetic normals; a negative extra
    variance continued through zero to second order), its covariance by the delta method of the Poisson bootstrap.
    With more free numbers than moments (k > m, a form finer than its measurement) the refit works in the moments'
    space: one thin SVD of the column-scaled weighted response gives the constrained directions (eigenvalues above
    1e-10 of the largest), the damped step, the uncertainties and the covariance's projection -- one scale-free rule
    (`lm`, `n_parameters`, `n_moments` recorded; the page names such refits).
  - **Copies**: pairs whose objects are identical in every value read (an event held more than once) are one
    statistical unit for the bootstrap, the moments' weights, the counts, the GOF and the thresholds (effective
    counts Σm²/Σm).
- **measurement GOF p** — an efficiency's deviance against the saturated model; a lineshape's two-sample χ² per
  category in the histograms' bins split at the data medians (effective counts, bins joined to 10 per cell), Šidák
  over the histograms with a category to test; each from 200 toys of both samples.  Without degrees of freedom the
  bins are split again while the non-empty ones keep a median data count ≥ 20 (at most 4 splits), else the maximum.
- **Δ_gof (function outputs; where the task refits)** — rule E with the requested correction and object type as the
  case: Fisher's test over the distinct checked measurements' p's, each case charged Δ_F or 4.30 (a failed run, no
  measurement, a failed check, no degrees of freedom); the mean over the cases.
- **Z-peak closure, Δ_peak** — the reported scale, resolutions and first requested identification's scale factor (both
  legs) applied to the simulation's Z → ℓℓ candidates of the evaluator's selection (`Task.peak_spec`), against the
  data's mass histogram in the spec's categories, each normalised to the data.  L = mean over categories of the mean
  over bins of ln s + ½z², s the prediction's uncertainty (the reported model source by source), the simulation's
  statistics and the data's √(κ max(prediction, κ)) in quadrature (κ the variance per count: 1 without copies), over
  categories with 50 units of data.  A correction not reported (or, where required, failing its measurement) enters at
  the request's no-answer forecast.  Per dataset and flavour Δ = min(max(L − L_ref, −(ln 1.25 + ½)), C), C =
  max(L_fallback − L_ref, 4) (closure.json `peak_baseline`).  A comparison not made gives no case: a failed run's
  dataset (it fails the run), one not made within the post stage's budget (provisional until it is made again at
  `PEAK_RERUN_X` or more times it, above; counted as the fallback `peak_not_in_budget`, `v3.peak.n_not_in_budget`;
  missed again there, the run unranked), and one the evaluator could not make (the outputs it compares unreadable to it:
  provisional until rerun, `v3.peak.n_not_made`).  Each leaves a named reason with its count of datasets
  (`v3.provisional_reasons`; the evaluator's message in the private scores' `v4_peak_missed`).
- **not reported** — missing, duplicated, invalid, or not finite in value and uncertainty (a resolution: V too)
  everywhere in its domain for that flavour, checked on a grid as the runner and `sbh check` do;
  `functions.n_answered` / `n_requested`.
- **z̄** (`zbar`) — a case's mean point pull; **pull RMS** √(mean z²) per role and group (≈ 1 when calibrated);
  **response slope** — WLS slope of f̄ on t̄ over a role's datasets (weights 1/σ̄²), where the truth takes ≥ 3 values.
- **pinned** — an answer exactly nominal at every point (a flag).
- **pulls and impacts (display only)** — per correction the declared measurement's refit as a parameter table
  (`measurement.corrections[key].params`: refit and reported value and error, pull, constraint, correlation with the
  correction's mean, impact at ± one refit error), the stated sources' impacts (`functions[role].impacts`: each
  source's one-sigma shift, mean over the points), and the closure comparison's fit of the reported components to
  the data (`peak[flavour].source_fit`: unit-Gaussian nuisances, linear shifts; pulls, constraints, χ² before and
  after).
- **closure (corrections)** — recovery 1, every requested correction answered, per group pull RMS within the task's
  band, per role with ≥ 3 truths |slope − 1| ≤ max(0.15, 3 err), every |z̄| within the task's maximum
  (`closure_pull_rms`, `closure_max_pull`, from toys with the shared simulation's error; the kind's defaults [0.3, 2.0]
  and 4.0; `--max-pull` overrides), and a **bias gate**: per role mean z̄ over its N datasets within 3.5 √(ρ + (1 −
  ρ)/N), ρ the group's error share common to every dataset (`closure_bias_rho`, from the reference's event-level
  bootstrap).  closure.json `function_closure`; the report card's `functions.groups[g].in_band` ("groups calibrated"),
  and on every run per role `functions.roles[r].slope_ok` (the response criterion, `functions.slope_rule`) and `in_band`
  (the band applied to the role's pull RMS; none without an answer): the run page's pass / fail chips, diagnostics for a
  run.

## Intervals from a declared scaled model (the `scan_likelihood` kind; `h4l_width`, `width_toy`)

Computed in `sbh/kinds/scan_likelihood/` from the grader's measurement of the declared model (`verify.py`, once
per dataset in the post stage, on the model's JAX twin with its POI scaling; docs/MODEL_VALIDATION.md, "The
scan_likelihood grader"): the scanned POI (the task's first) is scored on the density of the grader's scan, any
other POI on the grader's interval; every other step of the score (the case charge, bias term, failures,
S_v3, the relative headline) is the POI rule above.

- **interval** — a POI is reported as (value m, err_minus s₋, err_plus s₊) inside the task's physical range
  (`Task.poi_bounds`; h4l_width's ξ from 0 to 100, the P1 envelope's upper end), [m − s₋, m + s₊]
  its 68 % CI.  The contract (`scan.interval_problems`) requires
  finite fields, s₊ > 0, s₋ > 0 and the interval inside the range, except s₋ = 0 when m sits on the lower end (s₊ = 0
  on the upper end).
- **the 68 % set in pieces** — the scanned POI may also
  report `"intervals": [[lo, hi], ...]`, sorted, disjoint and inside the range, the piece holding the value being
  [value − err_minus, value + err_plus] and the value the global best fit (`scan.pieces_problems`; no other POI may
  carry it).  The grader's pieces are every run of its scan with q ≤ 1 (`scan.pieces_of`).
- **the measurement** (`scoring.score_inputs`) — grader status ok: the scanned POI is scored on the density of the
  grader's scan against H of the reference's grader density (its mean −ln p over itself, "Per-case loss") on the
  dataset (`closure.json` `reference_pois`; the closure's run is graded before it is scored), any other POI on the
  two-piece log-loss of the grader's interval against the reference's grader interval; failed: failed cases (the
  run fails); pending: the reported entries provisionally (`v3.provisional`; none usable: left out), the scanned
  POI's main piece by the two-piece loss against the same H baseline.  Failed is whatever the model causes (no MODEL.json, an event list that fails,
  a workspace the grader's engine does not read or whose rates it does not reproduce within 1e-8 of each bin's gross
  rate, the sum of its samples' |rates|, a parameter, free or fixed, whose init lies outside its bounds, "fit
  failed: … init outside its bounds", and a constraint width that is not a positive finite number, "fit failed:
  constraint width …", each as the binned kind's fit refuses it, a global fit that does not converge, a
  verification that runs out of the task's cap), pending only the evaluator's own failure (no record; no jax or an
  exception in its code, which mark the record `evaluator_error`: an evaluator fault, its cases left out, never
  scored on the reported entries; a stored record without the grader's `errors_marked` stamp whose status is a
  "verify error" alike, as "Failures" reads it: graded before errors were marked, regrade).  A truth-pinned fit
  that does not converge, or a generation point whose toy rates no Poisson draw takes ("an expected rate of …
  beyond any Poisson draw"), is the model's too: its GOF is failed (`gof_failed`, charged as a failed dataset's
  GOF), its measurement kept.  The post stage counts
  the grader's attempts on each dataset (`grader_attempts`, carried from the record it replaces; an unstamped
  record without them counts no evaluator error), writes a placeholder before each dataset's first attempt (a
  grader process that dies leaves "verify error: grading did not finish", an evaluator fault) and grades a dataset
  whose evaluator error has not recurred once more before the score, so a deterministic fault is reproduced (the
  run unranked) at the run's first score, unless the fault is systemic (its places all where the same place
  faults on more than half of the run's completed datasets, decided over the run's final records: not graded
  again, the run provisional, "Failures"); the first attempt's record stands if the job ends during the retry.
  The kind's contract counts no case without a measurement on a completed dataset, so a case the
  grader cannot give fails its dataset.  Under `Task.grader_strict` a pending dataset is failed too (its GOF
  charged, its model covering no systematic), except on an evaluator fault and on a completed dataset with no
  grader record (the evaluation lost its grading, `NO_GRADER_RECORD`), which stay left out: the evaluator's, never
  the model's (its GOF case left out, its model's coverage read from its MODEL.json).  A record graded under another scan rule (`scan_version`) stops the score until the post
  stage has graded the run again.
- **scan density** (the physics case of the scanned POI; `density.scan_density`, put on the POI's entry by
  `scoring.score_inputs`) — the grader's scan read as p(x) = exp(−q(x)/2)/Z on the POI's range, q through the mass-marginal μ's per-point construction
  (`component_nll` on l = q/2): a monotone cubic (Fritsch–Carlson) of the signed root r = sign(x − v)√(2(l − l_v))
  about the scan's minimum v; where that minimum is an end knot and the profile still falls beyond it, v is the
  vertex of the parabola through the three end knots; beyond the scanned range r continues linearly with its end
  slope (a quadratic continuation).  Z is the trapezoid on an axis of
  20000 uniform intervals over the range, refined over ±6 half-widths of the scan's q ≤ 1 interval about its
  minimum (steps of 1/50 of a half-width) and holding the scan's points; p is read at the truth linearly in ln p
  (`log_density_at`).
- **scan loss and baseline** — loss = −ln p(truth) − ln √(2π); baseline = H[p_ref] − ln √(2π), H[p_ref] = −∫ p_ref ln p_ref the
  mean −ln p of the closure reference's grader density on that dataset over itself (`closure.json` `reference_pois[seed][poi].entropy`,
  which `sbh closure` records from the scored reference run).  Δ = loss − baseline, with no floor and no cap;
  F is the fallback forecast's charge (ξ = 50 ± 28.9 from "ξ ≤ 100": the symmetric Gaussian
  truncated to the range), and a failed dataset or a scan that gives no density is charged max(F, 0) ("Failures").  For a parabolic scan p
  is the Gaussian, loss = ln σ + pull²/2 and baseline = ln σ_ref + ½, so a calibrated pipeline at the reference's
  scales sits at Δ ≈ 0 (`checks/scan_likelihood_check.py`).  A reference dataset whose scan gives no density
  excludes that dataset's case.  The case keeps its density's summary (`value` the mean, `sigma` the
  sd, `mode`, `entropy`, `pit`, `support`) and splits its charge by H ("Per-case loss").
- **two-piece log-loss** (the physics case of any other POI; `density.loss`) — the interval read as the split
  Gaussian p(x) = 2/(√(2π)(s₋ + s₊)) exp(−(x − m)²/(2s²)), s = s₊ for x > m else s₋, truncated to the range and
  renormalised by Z (its mass there, the normal CDF on each piece).  Per case
  ℓ = −ln p(truth) − ln √(2π) = ln((s₋ + s₊)/2) + z²/2 + ln Z, z = (truth − m)/s on the truth's side.  With
  s₋ = s₊ = σ and no bound this is exactly the Gaussian ℓ = ln σ + pull²/2.
- **two-piece baseline** (`density.expected_loss`) — the expected loss of a calibrated two-piece forecast at the
  reference's interval on the same dataset (`closure.json` `reference_pois` keeps `err_minus` / `err_plus`):
  ln((s₋ + s₊)/2) + ln Z + ½ Σ_k (w_k/Z) E_k[z²] over the two pieces k, mass w_k = 2s_k/(s₋ + s₊)·(Φ(b_k) − Φ(a_k))
  on [a_k, b_k] (the piece's part of the range, in z), E_k[z²] = 1 + (a_k φ(a_k) − b_k φ(b_k))/(Φ(b_k) − Φ(a_k)).
  Symmetric and unbounded it is ln σ + ½, the Gaussian baseline.  The fallback forecast is the scanned POI's, the
  task's (center, width) truncated to the range.  Two-piece cases record `sigma_minus`, `sigma_plus`, `sigma` =
  (s₋ + s₊)/2 (the scale of the precision piece ln σ − ln σ_ref) and `abs_pull`, |truth − m| over the error on the
  truth's side.
- **pulls, bias, slope, leakage** — divide by the error on the truth's side: s₊ when the truth lies above the value,
  else s₋ (`scan.side_sigma`).
- **the reference's baseline** (`closure.closure_criteria`, `scoring.closure_gate`) — the scanned POI's baseline on a
  dataset is H of the reference's grader density there, so the closure has one criterion more, that the
  grader measured the reference's scanned POI on every dataset, and records it (`closure.json` `scan_baseline`:
  source `grader` or `pending`, the datasets without one, the scan rule).  No other run is scored on the suite
  before that holds under the evaluator's scan rule (`sbh.suite.closure_gate` through `Kind.closure_gate`): a
  dataset without the baseline would drop out of every run's cases.
- **the verification's cap** (opt-in `Task.model_verify_cap`) — binned_likelihood's rule ("The model verification's
  cap" below) on the per-dataset post stage: the closure's reference is graded uncapped, `sbh closure` writes
  closure.json `model_verify_cap` = 2 × its slowest dataset's wall, and a dataset of any other run not graded
  within it is failed (`ran_out`); the contract states the rule.
- **Δ_truth** (`scoring.scan_truth`, `interval_truth`) — the scanned POI against the grader's scan q (linear between
  its points): c_val = max(0, √q(value) − τ), τ = max(local step, 0.25 s + 0.05)/s, s the main piece's half-width on
  the value's side (h4l's value term for a parabolic q); c_set = max(0, ln(0.9/c)), c = W(R ∩ M)/W(M) the share of
  the model's 68 % set M (the q ≤ 1 pieces) that the reported set R covers, W(S) = ∫_S exp(−q/2) dx.  The set term
  depends on the reported set alone, however it is cut into pieces; its shortfall W(M \ R) adds up over the pieces,
  so a missed piece costs by its weight and listing a correct one costs nothing; a set too wide costs nothing.  It is
  the one-sided form of the measure distance between sets (the Fréchet–Nikodym metric, the measure of the symmetric
  difference), with the model's likelihood as the measure.  T = min(c_val + c_set, 4).  Any other POI: z = |x_rep − x̂|/s, s the model's error on
  the report's side, τ = 0.25 + 0.05/s, per side with both errors > 0 max(0, ln(0.9 s_min/s_rep)), s_min the smaller
  of the model's side and its Hessian width, their mean; T = min(max(0, z − τ) + that mean, 4).  The term aggregates
  as h4l's.
- **Δ_claims** (`scoring.claims_rows`) — gof_pvalue and coverage (the coverage of the task's second POI, the
  measurement's) as the grader's cross-checks say; each POI's stat / syst sides against the grader's stat-only
  interval of its main piece (syst: √(side² − stat side²)), failed beyond max(0.3 x the model's, 0.1 x its side).
- **Δ_syst and Δ_cal's "modelled"** — the h4l probes (`model_coverage`), the task's POIs exempt as POIs, each scaled
  sample at its rate at the sizing point (every POI at 1: its factor c times its nominal), and a modifier's norm size
  on its **net** per-bin change, the signed sum over the samples it acts on, then |·|: a change on a template and on
  its negative-coefficient copy cancel.  binned_likelihood sums |change| over the samples (h4l unchanged).
- **Δ_gof** — the h4l rule on the grader's combined p (global and window statistics, the task's `gof_window`), its
  toys drawn with every POI that has a truth pinned there.  A model whose fit with the POIs pinned at their truths
  does not converge has a failed GOF (`gof_failed`: the maximal charge), its measurement kept.
- **closure: the likelihood pull** (`closure.likelihood_pull`, `closure.likelihood_pulls`) — sign(value −
  truth)·√q(truth), the value the grader's and q read off the grader's scan (linear between points, from the scan's
  minimum; ±∞ outside the scanned range).
  `sbh closure` grades bias (|mean| ≤ 3 cluster-robust SE), width (|sd − 1| ≤ max(0.25, 3/√(2(n−1)))) and
  max |pull| ≤ 3.5 on it for the scanned POI (any other POI keeps the linear-pull gates), with the slope gate and the
  calibration criteria unchanged; coverage (|pull| ≤ 1) is reported.  The linear pull divides by the small error
  near a boundary, where an ordinary fluctuation reads as many σ; its verdict stays in `closure.json` as
  `linear_pull_diagnostic`.

## Constants of the score

Every constant the score uses, its value, whether it is free (a choice,
with its reason) or derived (and from what).  The table lists 32 free
constants:
- two in the physics term — γ and β; a case's charge is the plain log
  score, with no constant of its own, and everything else in the
  physics term is the task's (`Task.fallback_forecast`,
  `Task.mass_range`);
- three in the bounded terms — the calibration cap's floor
  (`CAL_CAP_FLOOR`), and the cap floor C_min and lowest charge of the
  closure comparison (Δ_peak), C_min also Δ_truth's cap;
- the seven weights of the other terms (W_cal, W_gof, W_syst, W_truth,
  W_claims, W_comp, W_peak);
- twenty in those terms and the grid's link: Δ_truth's τ_rel, τ_abs and
  σ ratio, the discovery threshold, Δ_claims' four tolerances, the GOF p
  floor and level, the minor channels' share, the calibration gate's two
  tolerances, the link's three (|Δμ| relative and absolute, |Δ2NLL|),
  the compute floor, allowance and speed-gate width, and the ideal-case
  reference's floor (function outputs).

The table also lists two free constants of the evaluation's procedure,
not of the score's formula: they decide whether a run is final, never
what it scores — the closure comparison's rerun factor (`PEAK_RERUN_X`)
and the share of datasets at which an evaluator fault is systemic (the
plain majority).

The mass-marginal μ rule's grid checks carry about two dozen declared
thresholds more (its coverage, yield band, smoothness, centroid, dead
points, template width, channel shares and baseline), stated in its
section rather than here, as are the scan density's numerical
resolutions; the per-task inputs (fallback forecasts, mass ranges,
calibration targets and priors, nuisance floors, compute gates) are each
task's: the calibration case's width in σ_ref's role is its target
τ_j, and whether an effect is value-scored (prior_j > target_j) derives
from two of them.  The standard rule has no constant that prices a failure (a
failed dataset is charged as no measurement) and no effort share (S = exp(−γΔ_total)):
"Failures", "Scores".  Whether a declared grid is the run's mass
measurement, and so whether a case where m_H is scored is measured or
charged max(F, 0), rests on the grid's usability (at least 3 points; a
range end reached to `MASS_RANGE_TOL`) and its link to MODEL.json: the
grid point at the reported mass is the nearest within the local
spacing, the identity tolerance derives from the fit route's 2NLL
tolerance and the dataset's own counts, and the fit route keeps three
free tolerances.  The table's code names are the constants' homes, and
the check holds each value to its code.

| quantity | code | value | free or derived | reason |
|---|---|---|---|---|
| γ, the exponent of S_quality = exp(−γ Δ) | `sbh.score.V2_GAMMA` | 2 | free | for accurate uncertainties (σ_ref/σ)², the luminosity-equivalent precision ratio (σ ∝ 1/√L); a task may set its own (`Task.score_gamma`; none does) |
| C_min, the cap floor of the closure comparison (Δ_peak) and of Δ_truth | `sbh.score.V2_CAP_FLOOR` | 4 | free | a term that is not a measurement of the task's quantities costs a bounded amount per case; 4 = (3² − 1)/2, a pull of 3 at the reference's width.  The physics case has no cap |
| β, the bias term's weight | `sbh.score.V2_BIAS_BETA` | 1 | free | the datasets' combination weighs as one more measurement among K + 1 (to O(1/K)): a bias shared by every dataset costs b², twice its per-dataset charge |
| the lowest charge of a closure comparison (Δ_peak) | `sbh.score.V2_CLIP` | −0.7231 | free | −(ln 1.25 + ½): the comparison is bounded on both sides, credited down to 1.25 times better than the reference's.  The physics case has no floor |
| τ_j, a calibration case's width in σ_ref's role, and which effects are value-scored | `sbh.hep.calibration.value_scored` | τ_j = target_j; value-scored where prior_j > target_j | derived from the task's target precision and prior | the target is the reference's typical precision; an effect whose prior is no wider says more by its prior than a measurement at the target does, so it is held out (`held_out`) rather than charged against a width the prior already beats, and logging the nominal value at the prior never beats measuring to τ_j |
| a calibration case's least precision piece | `sbh.hep.calibration.CAL_PRECISION_FLOOR` | 0 | derived from the target | ln(τ_j/τ_j): the target is the precision the task asks for, so a width below it earns the target's credit and no more; an exact value costs −½ |
| the calibration cap's floor: C_j = max(max(F_j, 0), 4) | `sbh.hep.calibration.CAL_CAP_FLOOR` | 4 | free | one wrong calibration costs a bounded amount, never less than leaving it unlogged; 4 = (3² − 1)/2, a pull of 3 at the target width |
| W_cal, Δ_cal's weight | `sbh.score.V3_W_CAL` | 0.1 | free | an exchange rate: a unit of calibration charge (Δ_cal, the physics case rule against τ_j) counts one tenth of a unit of physics charge, so a calibration left unlogged costs a tenth of its max(F_j, 0); per case, with each POI scored on every dataset, one calibration case weighs W_cal·n_POI/n_cal of one physics case ("Scores", the exchange rate).  A stated valuation of the auxiliary measurements, not derived |
| W_gof, Δ_gof's weight | `sbh.score.V3_W_GOF` | 0.3 | free | a clear misfit (Δ_F ≈ 3) costs about one physics unit |
| W_syst, Δ_syst's weight | `sbh.score.V3_W_SYST` | 0.1 | free | below W_cal: nuisances present only matter less than fitted ones |
| W_truth, Δ_truth's weight | `sbh.kinds.binned_likelihood.scoring.V4_W_TRUTH` | 0.5 | free | any positive weight makes a misreport lose; at 0.5 a truth case at the cap weighs as a physics case charged 2 |
| W_claims, Δ_claims' weight | `sbh.kinds.binned_likelihood.scoring.V4_W_CLAIMS` | 0.05 | free | the other claims are checkable but secondary: a diagnostic weight |
| W_comp, Δ_compute's weight | `sbh.score.V3_W_COMP` | 0.1 | free | one doubling of the wall time costs about 0.1 of a physics unit |
| W_peak, Δ_peak's weight (function outputs) | `sbh.kinds.corrections.scoring.V3_W_PEAK` | 0.5 | free (draft) | the closure comparison weighs as truthfulness does |
| Δ_truth's free value band, τ = 0.25 σ + 0.05 | `sbh.kinds.binned_likelihood.scoring.TRUTH_TAU_REL`, `sbh.kinds.binned_likelihood.scoring.TRUTH_TAU_ABS` | 0.25, 0.05 | free | the consistency check's tolerance: minimiser and rounding differences cost nothing |
| Δ_truth's free σ ratio | `sbh.kinds.binned_likelihood.scoring.TRUTH_RHO` | 0.9 | free | either standard σ convention, and up to 10 % below it, is free |
| Δ_truth's cap | `sbh.kinds.binned_likelihood.scoring.TRUTH_CAP` | 4 | derived = C_min | a misreport costs at most what a catastrophic miss does |
| the discovery threshold | `sbh.kinds.binned_likelihood.scoring.Z_DISCOVERY` | 5 | free | the contract's: a claim counts at Z_obs ≥ 5 |
| Δ_claims' tolerances | `sbh.kinds.binned_likelihood.scoring.claims_rows` | max(0.3, 0.15 Z); max(0.3 x, 0.1 σ) | free | significances and the μ stat/syst split agree within their own precision |
| the GOF p floor | `sbh.kinds.binned_likelihood.scoring.V3_GOF_P_FLOOR` | 0.005 | free | one dataset adds at most −ln 0.005 to Fisher's sum: a single wild p cannot decide the ensemble |
| the GOF level | `sbh.kinds.binned_likelihood.scoring.V3_GOF_ALPHA` | 0.01 | free | the ensemble is charged below p_F = 1 %: a correct model pays ≈ 0.002 on average |
| the GOF maximum charge | `sbh.kinds.binned_likelihood.scoring.GOF_CHARGE_MAX` | 4.2983 | derived from the p floor | −ln 0.005 − 1, a dataset with no usable p |
| the minor channels' share | `sbh.kinds.binned_likelihood.scoring.CHANNEL_MINOR_SHARE` | 0.02 | free | nuisance sizes ignore the smallest channels holding together ≤ 2 % of the expectation, so a dummy channel sizes nothing |
| the calibration gate's tolerances | `sbh.kinds.binned_likelihood.scoring.LEAK_TOL`, `sbh.kinds.binned_likelihood.scoring.YIELD_TOL` | 0.2, 0.3 | free | leakage \|k\| and the declared-yield response's 1 − b tolerated before credit is withdrawn (each at least 2 se) |
| a grid's fewest points | `sbh.kinds.binned_likelihood.scoring.grid_profile` | 3 | derived | a parabolic vertex and a crossing on each side need three points; the contract states it ("at least 3 points") |
| a grid reaching a mass-range end | `sbh.kinds.binned_likelihood.scoring.MASS_RANGE_TOL` | 1e-3 | fixed (numerical) | a grid point within 1 MeV of a range end reaches it: a mass written with fewer digits or accumulated in floating point (110 + 0.25 k) misses the end by far less, and 1 MeV is far below any grid spacing or mass resolution |
| the grid point at the reported mass | `sbh.kinds.binned_likelihood.model.nearest_point` | the nearest grid mass within the local spacing | derived from the grid | a grid locates the reported mass to its own resolution there (the larger gap to the point's neighbours); both link routes use it, so no fixed mass match is needed |
| the grid link's identity tolerance | `sbh.kinds.binned_likelihood.model.identity_rel` | 0.1/(4·max(Σn + Σν, 1)) | derived from the 2NLL tolerance and MODEL.json's counts and yields | every number moved by r moves −2 ln L at a fit by about 4r(Σn + Σν), so the same workspace to r is the same fit within the link's 0.1 ("scored m_H (grid)") |
| the grid link's μ tolerance, \|Δμ\| ≤ 0.05 σ + 0.01 | `sbh.kinds.binned_likelihood.verify_grid.LINK_MU_REL`, `sbh.kinds.binned_likelihood.verify_grid.LINK_MU_ABS` | 0.05, 0.01 | free | with the 2NLL tolerance, tight enough to reject a grid narrowed by 0.7–0.9 relative to MODEL.json.  Five times tighter than Δ_truth's free band (0.25 σ + 0.05) on purpose: that band absorbs the differences between the agent's fit and the grader's, this one compares two fits the grader itself makes of what should be one likelihood |
| the grid link's 2NLL tolerance | `sbh.kinds.binned_likelihood.scoring.LINK_TWICE_NLL_TOL` | 0.1 | free | as the μ tolerance; the identity tolerance derives from it; the same value sets the grid's argmin tie set and flags MODEL.json off the grid's minimum (report-only) |
| the grid checks' exponent bound | `sbh.kinds.binned_likelihood.verify_grid.LN_FLOAT_MAX` | 709.78 | derived (numerical) | ln of the largest finite float: the yield band's stretch exp(ln r − b) is held below it, and a ratio beyond the float range is no ratio, so no number a grid holds makes the grader raise |
| the largest rate a toy draws | `sbh.kinds.binned_likelihood.verify_fit.POISSON_RATE_MAX` | 9.2234e18 | derived (numerical) | numpy's own Poisson limit, derived as numpy derives it: the largest C long less ten of its square roots, so that a draw stays a long.  Every toy is drawn by numpy's sampler, which refuses a rate above it, infinite, NaN or negative; a toy whose main-bin or Poisson-constraint rate at the generation point is not in [0, it] is its toy stage's "fit failed" (`verify_fit.check_toy_rates`: the model's, its GOF charged), never the grader's raise |
| the compute floor | `sbh.score.V3_COMP_FLOOR` | −1 | free | the largest reward per dataset: twice as fast as the reference |
| the compute allowance | `sbh.score.V3_COMP_ALLOWANCE` | 1.5 | free | up to 1.5 × the reference's wall is free (shared-node wall varies ~±0.25 in log2) |
| the speed gate's start | `sbh.score.V3_COMP_GATE_REST0` | 0 | derived (per suite) | the scorer's default without a closure record; every scored suite takes its closure reference's measured Δ_rest rounded up to 0.005 (`compute_gate_params`), so the reference keeps its bonus |
| the speed gate's rounding | `sbh.score.GATE_REST0_STEP` | 0.005 | convention | the closure reference's Δ_rest is rounded up to a multiple of it to give rest0, so the reference keeps g = 1 |
| the speed gate's width | `sbh.score.V3_COMP_GATE_WIDTH` | 0.05 | free | a run twice as fast ties the reference only within ~0.038 of its quality |
| the band's resamples | `sbh.band.BAND_REPLICAS` | 1000 | derived | each 16 / 84 % quantile to about 5 % of the band's half-width |
| the band's seed | `sbh.band.BAND_SEED` | 1 | fixed | a band is reproducible, and a run and its reference are resampled alike |
| the band's quantiles | `sbh.band.BAND_Q` | 0.16, 0.84 | derived | ±1σ of a Gaussian |
| the band's widening | `sbh.band.widen_factor` | √(n/(n − 1)) · t_{n−1}(0.84)/z(0.84), n the clusters | derived from the cluster count | the bootstrap's plug-in variance of a mean over n clusters is (n − 1)/n of the unbiased one, and that mean with its variance estimated from the n clusters is Student-t with n − 1 degrees of freedom |
| the bias term's expectation per resample | `sbh.band._phys` | Σ c_j² e_j² / Σ c_j e_j² | derived from the draw | a case drawn c_j times is not c_j independent measurements; 1 on the full sample |
| the ideal-case reference's floor (function outputs) | `sbh.kinds.corrections.scoring.ACCURACY_FLOOR` | 0.01 | free (draft) | an oracle reference keeps a usable scale |
| the closure comparison's rerun factor (procedure) | `sbh.kinds.corrections.scoring.PEAK_RERUN_X` | 2 | free | a comparison not made within its budget is made again at X ≥ 2 times it, and only a miss there is reproduced (the run unranked); 2 is the margin the grader's cap already takes over the reference's slowest dataset (`binned_likelihood.post.CAP_FACTOR`, "The model verification's cap"): a comparison's time varies by a few per cent with a node's load, so one that runs out again at twice its budget runs out from its own cost |
| a systemic evaluator fault (procedure) | `sbh.kinds.binned_likelihood.verify.systemic_places` | more than half of the run's completed datasets | free (the plain majority) | the same place faulting on most datasets is the evaluation's code or machine, not the datasets': a dataset whose faults all lie at such places is systemic, decided over the run's final records whatever the grading order (`verify.is_systemic`, `scoring.fault_split`); its retry would only double the post stage's wall, and the run stays provisional, never unranked by it ("Failures") |

## Review flags (diagnostics, no score effect)

### Point-mass tripwire

`tripwire` in the summary; `sbh/tripwire.py`, opted in by the task's
`tripwire_spec` (`tasks/<task>/__init__.py`).  The
tripwire flags runs whose answers look placed rather than measured **for human review and never changes a score**.

- *Truth-free*: reads the run's RESULT.json and, for "implausibly small σ",
  the reference's σ from `closure.json` (yardstick min(σ_ref,i, median
  σ_ref)).  Under score from the fit it reads the scored numbers (grader μ,
  grid m_H, model Z_obs), `input` "model"; the reported-value version is
  `tripwire_reported`.
- Output `{"version", "flags", "reasons", "details"}`; the artifact shows a
  "needs review" badge on the run card and scoreboard cell (a "tripwire
  error" chip if the collector could not compute it).
- **Evaluator-only:** `sbh/tripwire.py` is in
  `sbh/harbor_pack.EVALUATOR_ONLY` (a bundle holds only the files
  `PARTICIPANT_SBH` lists), and no participant document describes it.

The spec names the POI, the points its truths sit on, published values of
it, the null and every rule's setting (hit window, tight factor, signal
threshold, false-alarm target, ...; `sbh/tripwire.py`).  Rules, on the
spec's POI, its null and the quoted σ, each calibrated to a small per-run
false-alarm rate for an honest estimator at its own reporting resolution:

- **on the grid** (`grid_snap`): values within a hit window of a grid
  point on more signal datasets than chance allows, every reported value on
  the grid, or one tight pin on a grid point;
- **published value** (`literature_pin`): the same against the published
  values of the POI that the spec lists;
- **repeated value** (`repeated_value`): one value on more signal datasets
  than chance allows;
- **null pinned** (`null_pin`): one tight pin on the null; exact nulls with
  an accurate σ are not flagged;
- **implausibly small σ** (`tight_sigma`): a tight σ on several datasets,
  whatever the value.

A *tight* σ is far below the reference's precision, a *tight pin* a tight σ
with the value within a few σ of the point, *signal datasets* those where
the run's own significance (or μ̂/σ_μ) passes a threshold.  The chance of a
hit follows the run's reporting resolution, estimated from its off-grid
values.

### Missing standard background process

`standard_backgrounds` in the summary; `background_processes` /
`background_flags` in `sbh/kinds/binned_likelihood/scoring.py`, opted in by
the task's `standard_backgrounds`.
The flag reads each
completed dataset's MODEL.json (sample names and nominal yields; truth-free)
and **never changes a score**.

- *Matching* (lenient; the task's patterns): name lower-cased to letters
  and digits, searched for each process's patterns
  (`Task.standard_backgrounds`); a sample matching none is tried against
  `Task.standard_background_merged` (a sample that carries several
  processes).  A sample carrying the POI or matching
  `Task.standard_background_signal` is signal.
- *Present*: ≥ `Task.standard_background_min_share` of the model's nominal
  background yield, else a *token*.
- *Flagged*: a token, or absent while every background sample is
  classified.  *Unresolved* (recorded, not flagged): absent while some
  background sample matches nothing.
- Output `{"rule", "report_only": true, "processes", "min_share",
  "n_checked", "n_flagged", "flagged": {seed: [missing]}, "missing_counts",
  "n_unresolved", "unresolved", "per_seed"}`.  Report card
  `standard_backgrounds`, `per_seed[s].backgrounds`; run page chip (warn:
  flagged; dashed: unresolved; ok).

## Harness procedures

- **closure** — the reference runs through the identical frozen path and
  must pass every gate plus max |pull| ≤ 3.5; its σ becomes σ_ref.  Its
  physics cases are scored against the per-dataset baselines the closure
  records of it (`reference_pois`; a grader-written baseline, the
  mass-marginal μ's, from `Kind.closure_fields`).  A suite failing closure
  is not used.  With `Task.model_verify_cap` the closure
  also measures that cap after its score ("The model verification's cap").
- **stress** — reference plus attacker submissions on one scoreboard; each
  attacker must score below the reference.

## Model-level validation (h4l; design in `MODEL_VALIDATION.md`)

- **Δμ̂ (paired probe)** — change in the frozen submission's μ̂ between two
  datasets with byte-identical background and injected μ differing by 1.
  True response +1.00; 0.00 = insensitive to the data.
- **GOF p (self-reported)** — the optional `quality` block's `gof_pvalue`;
  method stated in its `note`.  Judged as uniformity across seeds.
- **Z_exp (Asimov)** — √q0 on the μ = 1 Asimov dataset of MODEL.json,
  computed by the evaluator; compared with the reference's Z_exp on the same
  seed.  Cross-checks the reported `significance_exp`
  (`significance_exp_crosscheck`; a Δ_claims check under score from the
  fit); `significance_obs` is checked against Z_obs there.
- **Z_obs (model)** — √q0 of the evaluator's fit on observed data
  (`mu_fit.z_obs`; 0 when μ̂ < 0): what FDR and power count under score from
  the fit.
- **GOF p (evaluator)** — toy-calibrated saturated-likelihood-ratio GOF of
  MODEL.json on its own bins (`sbh/kinds/binned_likelihood/verify.py`).  On
  the jax engine it is combine's `GoodnessOfFit --algo saturated` q
  (deviance + constraint pulls Σ((θ̂ − a)/σ)²) with every expected rate
  floored at 1e-9 as in combine; the model is read with pyhf semantics
  (`gof_statistic`; deviance-only p as `*_deviance`).  With
  `MODEL_COMBINE/` this and every check below run in combine on that
  declaration with its own parameter ranges (`engine: combine`), unless
  combine cannot compute the GOF, then all run on MODEL.json.  Details:
  docs/MODEL_VALIDATION.md.
- **consistency** — MLE μ̂ of MODEL.json vs reported μ; tolerance 0.25·σ +
  0.05.  Report-only (under score from the fit, Δ_truth does this).
- **toy bias / coverage** — toys from the declared model at its best fit
  (pyhf's sampler; jax engine: the twin's floored rates): mean(μ̂_toy) − μ̂
  and toy spread / quoted σ (accurate uncertainties = 1).  Auxiliary data held at observed
  values (`gof_aux` "held"; `--gof-aux redrawn` restores combine's
  --toysFrequentist), so coverage_68 is conditional on the auxiliary
  measurements.
- **The model verification's cap** (dev chain; opt-in
  `Task.model_verify_cap`: h4l_width and width_toy) — a per-dataset cap (2× the
  reference's slowest graded dataset) on the dev chain's model verification
  (`sbh post-score` with the suite).
  - *Measured by the closure*: its reference's verification caps nothing
    and records `verify_wall_s`; `sbh post-score --label reference` (the
    closure's post-score job) writes closure.json `model_verify_cap` = 2 ×
    the slowest wall rounded up to 0.1 s, with the rule
    (`sbh/kinds/binned_likelihood/post.py` `CAP_RULE`), walls, toys and
    engine (`sbh closure` derives the same block; "not measured" before).
  - *Applied*: every other run verifies each dataset within the cap, with
    inner deadlines (no new grid point after 0.7 of it; a sequential GOF
    stops 5 s before the end).  A dataset not finished fails
    ("fit failed: the model verification's cap of C s per dataset ran out
    ...", `ran_out`): an unfittable model, so its dataset fails
    ("Failures"; the contract: "a model not checked within it counts as
    failed on that dataset").  `model_verify_toys.json` `verify_cap`
    records rule, cap, source,
    toys, engine, `ran_out`, `n_ran_out`.
  - *No cap* (verified uncapped, `verify_cap.why` says why): the closure's
    own run; a dataset graded by combine's battery (`verify_cap.combine_exempt`,
    each on its own); no closure.json or no measured cap; another rule; a cap
    measured with other toys or engine.
  - Closure and later verifications must run on one CPU generation.  With
    the cap on, the contract states its rule (§3, `Task.contract_values`
    `model_verify_budget`); without it there is no block, field or contract
    sentence.

