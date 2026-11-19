<!-- The scan_likelihood kind's parts: RESULT.json with POIs as asymmetric intervals (result, then its optional
fields: result_optional), the scanned POI's 68 % set in several pieces, MODEL.json with its POI scaling, its gate,
the evaluator's own measurement and the size caps (with binned_likelihood's binning and event list).
A template (the unified tasks/EVAL_CONTRACT.in.md, or a task's own EVAL_CONTRACT.in.md) includes them as
{{core:<part>}}, {{hep:<part>}}, {{kind:<part>}} (the task's own kind's) or {{kind.<name>:<part>}} (a named kind's);
{{value:<name>}} is filled from the task (Task.contract_values; the unified contract's own names win there).  Each
part is the exact text between its opening marker line and its closing marker (`python -m sbh.contract render`
writes tasks/<task>/EVAL_CONTRACT.md). -->

<!-- part: result -->
A JSON object (GATE) with these fields:

    {
      "pois": [
        {"name": "<poi>", "value": f, "err_minus": f, "err_plus": f}, ...
      ],
      "nuisances": { "<name>": {"value": f, "unc": f, "note": "..."}, ... },   # {{ref:nuisances}}
      "quality": { "<name>": {"value": f, "note": "..."}, ... }
    }

`pois` — the physics quantities you measure, one entry each:

| name | quantity |
|---|---|
{{value:interval_pois}}

`[value − err_minus, value + err_plus]` is your 68 % confidence interval.
All values finite; the interval lies inside the quantity's physical
range; `err_minus` and `err_plus` > 0, except that `err_minus` = 0 is
allowed when `value` sits on the lower end of the range (the interval then
starts at the boundary), and `err_plus` = 0 when it sits on the upper end.
Report every field on every dataset.  Extra keys are allowed and ignored.
<!-- /part -->

<!-- part: result_optional -->
Optional, recorded as given:

- per POI, `"stat"` and `"syst"`, each `{"err_minus": f, "err_plus": f}`
  (≥ 0): the statistical component of the interval and the systematic
  one, which combine in quadrature to the total;
- `"quality"`: quality or validation measurements you performed on your
  analysis.  Use these names when you measured the corresponding quantity,
  with the method stated in `note` (e.g. "saturated model", "toys", "KS"):
  - `gof_pvalue` — goodness-of-fit p-value of your final fit to the
    dataset;
  - `coverage` — expected coverage of your quoted 68 % intervals, from
    your own toy studies.

  Other measurements may be reported under names of your own choosing.

`nuisances` — the log of the systematic effects and corrections of your
analysis, {{ref:nuisances}}.
<!-- /part -->

<!-- part: pieces -->
The 68 % set of `{{value:scan_poi}}` may be several disjoint pieces (a
likelihood with more than one minimum).  Then report them all:

    {"name": "{{value:scan_poi}}", "value": f, "err_minus": f, "err_plus": f,
     "intervals": [[lo, hi], ...]}

`intervals` — every piece, sorted, disjoint, each with `lo` < `hi`, inside
the quantity's range; the piece holding `value` is
`[value − err_minus, value + err_plus]`, `value` is the global best fit,
and `stat` and `syst` describe that piece.  List every piece where your
q = −2 Δln L is below 1; without `intervals` the set is the one piece.
Only `{{value:scan_poi}}` may carry `intervals`.
<!-- /part -->

<!-- part: model -->
Next to `RESULT.json`, write `MODEL.json`: your fitted statistical model
exported as a pyhf/HistFactory workspace — binned channels of the
observable you fit, the samples with their modifiers, the observed counts
of your own selection on that dataset as `observations` (integer event
counts: non-negative, and no channel's total above the dataset's number
of events — the evaluator checks both), and every quantity of
{{ref:result}}'s table as a `normfactor` under its own name (the
measurement's `poi` one of them).  A top-level `_meta` object must contain
`binning`, one entry per channel, and may contain `poi_scaling` (below):

    "_meta": {"binning": {"<channel>": {"observable": "<<!-- /part -->

<!-- part: scaling -->
POI scaling.  A sample's expected rate may depend on the quantities of
{{ref:result}} through a monomial, declared in `_meta.poi_scaling`:

    "poi_scaling": {"<channel>": {"<sample>": {"factor": c,
                                                "powers": {"<quantity>": p, ...}}}}

or, for the least of several monomials, `{"factor": c, "min_of":
[{"<quantity>": p, ...}, ...]}` (exactly one of `powers` and `min_of`).
`c` is finite and non-zero, of either sign; each `p` is one of −1, −½, 0,
½, 1; only the quantities of {{ref:result}} appear.  The evaluator's
expected rate of such a sample in a bin is max(nominal + `histosys`
shifts, 1e-9) × c × ∏ quantity^p (or the least of the monomials) × its
other modifiers, and each bin's total expected rate is floored at 1e-9.  A
negative contribution (an interference term, say) is therefore a positive
template with a negative `c`: the nominal of a scaled sample must be ≥ 0.
The sample carries a `normfactor` named after exactly each quantity with a
non-zero power in its entry; these stand for the scaling, which pyhf alone
would read as a plain product.  A monomial with a negative power on a
quantity whose range reaches 0 must sit in a `min_of` beside a monomial
without one (its rate is infinite at 0).  A sample without an entry scales
with its normfactors as pyhf reads them.
<!-- /part -->

<!-- part: gate -->
GATE, all mechanical:

- the file loads as a pyhf workspace and builds a model
  (`pyhf.Workspace(spec).model()`, with `_meta` removed);
- every modifier is of a type the evaluator reads: `normfactor`,
  `normsys`, `histosys`, `staterror`, `shapesys` or `lumi`;
- `_meta.binning` is complete as above and `_meta.poi_scaling`, when
  present, follows the rules above;
- every quantity of {{ref:result}}'s table is a `normfactor` of the model;
- no sample carries the same modifier (name and type) twice;
- every modifier except those quantities and the bin-wise MC-statistics
  types (`staterror`, `shapesys`) is named by the rules of
  {{ref:nuisances}} and logged in `RESULT.json["nuisances"]`.

A missing or failing `MODEL.json` fails the run like a malformed
`RESULT.json`.  You do NOT need pyhf at run time: the workspace is plain
JSON, writable from numpy alone.  `sbh check` validates the export (the
gate) and, where pyhf is importable and the model has at most 20 free
parameters, refits it with its POI scaling and prints the best fits and
the 68 % pieces next to those of `RESULT.json` — a diagnostic; a fit by
pyhf alone ignores the scaling.
<!-- /part -->

<!-- part: measured -->
The evaluator measures the quantities and their uncertainties itself, from
the model you declare, with the POI scaling: it scans the profile
likelihood of `{{value:scan_poi}}` over its whole range (q = −2 Δln L,
every other parameter profiled) and fits every other quantity (its best
fit and q = 1 interval).  The quantities' ranges in {{ref:result}} bound
these fits, whatever `MODEL.json` declares.  `RESULT.json` must report
what that model gives: the best fits, the q = 1 intervals and, for
`{{value:scan_poi}}`, every piece below q = 1.  The evaluator also tests
the model against your data with toys drawn from it (goodness of fit) and
probes its systematics.

Fit conventions, so that you can reproduce the evaluator's fit of a model
`sbh check` does not refit.  The evaluator reads `MODEL.json` with
HistFactory semantics, as pyhf 0.7 builds it with `_meta` removed: its
parameters, their constraint terms and auxiliary data, and the initial
values, bounds and `fixed` flags your measurement sets (pyhf's defaults
where it sets none); `normsys` interpolated with code 4 and `histosys`
with code 4p; a `staterror` as one factor per bin on every sample that
carries it, scaled or not, Gaussian-constrained around 1 with the relative
width √(Σ δ²) / Σ nominal over those samples as declared (a bin where that
width is 0 keeps its factor at 1).  Three things differ from pyhf.  A
sample with a `poi_scaling` entry has the rate given above.  Every
sample's nominal plus `histosys` shifts is floored at 1e-9 before its
multiplicative modifiers, and each bin's total expected rate at 1e-9, as
CMS Combine does.  Every quantity of {{ref:result}} floats inside its
range there, starting from its declared initial value moved into that
range; a `fixed` flag or other bounds on it are not read.  The fit
minimises −ln L = Σ over bins (ν − n ln ν) + the constraint terms (ν the
bin's expected rate, n its observed count); q is twice the rise of that
minimum, at a fixed value of the quantity, above its lowest value in the
range, and the interval of a fitted quantity ends at the end of its range
on a side where q stays below 1.
{{value:model_verify_budget}}<!-- /part -->

<!-- part: size_caps -->
Size caps.  The evaluator reads at most 256 MiB of `MODEL.json` and
64 MiB of `EVENTS.json`; a larger file counts as a failed run on that
dataset.
<!-- /part -->
