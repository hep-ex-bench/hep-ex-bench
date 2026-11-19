<!-- The binned_likelihood kind's parts: the quality block, MODEL.json, its gate, the event list, the evaluator's own measurement, the declared mass grid and the size caps (each in two variants: `measured` + `mass_grid` + `size_caps` for a task that scores the signal strength on the fit at the reported mass, `measured_marginal` + `mass_grid_marginal` + `size_caps_marginal` for one that marginalises the grid's likelihood over the mass range -- the kind's variant_pieces picks `mu_point` or `mu_marginal`) and the optional combine declaration.
A template (the unified tasks/EVAL_CONTRACT.in.md, or a task's own EVAL_CONTRACT.in.md) includes them as
{{core:<part>}}, {{hep:<part>}}, {{kind:<part>}} (the task's own kind's) or {{kind.<name>:<part>}} (a named kind's);
{{value:<name>}} is filled from the task (Task.contract_values; the unified contract's own names win there).  Each
part is the exact text between its opening marker line and its closing marker (`python -m sbh.contract render`
writes tasks/<task>/EVAL_CONTRACT.md). -->

<!-- part: quality -->
Optionally, a `"quality"` object may report quality or validation
measurements you performed on your analysis:

    "quality": { "<name>": {"value": f, "note": "..."}, ... }

Use these names when you measured the corresponding quantity, with the method
stated in `note` (e.g. "saturated model", "toys", "KS"):

- `gof_pvalue` — goodness-of-fit p-value of your final fit to the dataset.
- `coverage` — expected coverage of your quoted 68% intervals, from your own
  toy studies.

If reported, they are compared with the evaluator's own saturated-likelihood
goodness-of-fit test of the declared model and with the toy coverage of
`mu`'s 68% interval at the best-fit mass.  Other measurements may be
reported under names of your own choosing; everything is recorded as given.
<!-- /part -->

<!-- part: model -->
Next to `RESULT.json`, write `MODEL.json`: your fitted statistical model
exported as a pyhf/HistFactory workspace — a binned channel (or channels) of
the observable you fit, signal and background samples with modifiers, the
observed counts of your own selection on that dataset as `observations`
(integer event counts: non-negative, and no channel's total above the
dataset's number of events — the evaluator checks both), and `{{value:signal_poi}}` as the
POI.  A top-level `_meta` object is allowed and must contain `binning`, one
entry per channel:

    "_meta": {"binning": {"<channel>": {"observable": "<<!-- /part -->

<!-- part: binning -->
>",
                                        "edges": [e0, e1, ..., eN],
                                        "data_derived": bool}}}

`edges` has one more entry than the channel has bins; `data_derived` says
whether the binning was chosen by looking at the data being fitted.
<!-- /part -->

<!-- part: gate -->
GATE, all mechanical:

- the file loads as a pyhf workspace and builds a model
  (`pyhf.Workspace(spec).model()`, with `_meta` removed);
- `_meta.binning` is complete as above;
- every modifier except the POI `{{value:signal_poi}}` and the bin-wise MC-statistics types
  (`staterror`, `shapesys`) is named by the rules of {{ref:nuisances}} and logged in
  `RESULT.json["nuisances"]`.

A missing or failing `MODEL.json` fails the run like a malformed
`RESULT.json`.  You do NOT need pyhf (or to fit with it) at run time: the
workspace is plain JSON, writable from numpy alone — export whatever you
fitted with into the schema.  pyhf is importable in the development
environment, and `sbh check` validates the export for you (the gate; it also
refits and reports whether the workspace's MLE `{{value:signal_poi}}` matches your reported
value, as a diagnostic).
<!-- /part -->

<!-- part: event_list -->
Event list.  Next to `MODEL.json`, write `EVENTS.json`, naming the data
events behind every observed count of `MODEL.json`:

    {"channels": {"<channel>": {"file":  ["part_0000.root", ...],  # its data file (a name in data/)
                                "entry": [i, ...],     # its 0-based entry index in that file's Events tree
                                "x":     [f, ...]}}}   # the value you binned it by

one entry per channel of `MODEL.json`'s `observations` (empty lists for a
channel with no events), and in each the three lists of equal length, one
element per counted event (plain JSON, writable from numpy alone).
Requirements: every listed event is an event of the dataset; no event is
listed more than once, across all channels; and in every channel, each
`x` lies within the channel's `_meta.binning` edges, and histogramming the
`x` values with those edges (as `numpy.histogram` does: each bin includes
its lower edge, the last bin also its upper one) reproduces the channel's
observations exactly.  The evaluator checks all three on every dataset; a
run whose `EVENTS.json` is missing or fails one counts as failed on that
dataset.  `sbh check` runs the same checks against the development
dataset.
<!-- /part -->

<!-- part: measured -->
The evaluator measures `{{value:signal_poi}}`, `mH` and their uncertainties itself, from the
model you declare: `{{value:signal_poi}}` from its own fit of `MODEL.json`, `mH` from the
mass grid below.  `RESULT.json` must report what that model gives.  The
evaluator evaluates the workspace with HistFactory semantics (as pyhf
builds it), flooring each sample's expected rate in each bin, before its
multiplicative modifiers (its nominal plus `histosys` shifts), and each
bin's total expected rate at 1e-9, as CMS Combine does.
{{value:model_verify_budget}}<!-- /part -->

<!-- part: mass_grid -->
The mass measurement is evaluated only from a declared mass grid: the same
model at fixed mass hypotheses, in a directory `MODEL_GRID/` next to
`MODEL.json` containing `<poi>_<value>.json` files (the mass POI's name of
{{ref:result}} and the value in GeV, e.g. `mH_124.50.json`), each a pyhf workspace; a
single file `MODEL_GRID.json` mapping `"<value>"` to the workspace is
accepted equivalently.  Every grid
workspace MUST be the same model as `MODEL.json` on the same observations
— identical channels and bin counts, `MODEL.json`'s observations, identical
background samples and modifier data, identical nuisance and modifier set,
an identical measurement block (POI and every parameter setting) — with
only what depends on the mass (the signal templates) re-evaluated at that
point's fixed mass: that is what makes the fitted grid a profile
likelihood in the mass, and the evaluator checks it structurally.  Include
your reported mass as a grid point: `MODEL.json` is the grid's workspace
at that mass, and the grid point nearest your reported mass must
reproduce `MODEL.json`'s own best fit (the same `mu` and likelihood, to
numerical precision).  Recommended: 15–60 points covering the reported
mass and a few times its uncertainty on both sides, spaced ≤ 0.25 GeV
near the best fit.  The evaluator fits each point (`mu` free — a point
whose measurement fixes it is not a profile — and the nuisances as the
measurement block sets them) and reconstructs the mass profile from the
grid.  A grid is usable when it is `MODEL.json`'s model as above, has at
least 3 points, and covers your mass interval: on each side of the
profile minimum, a ΔlnL = 0.5 crossing lies inside the grid, or the grid
extends to that end of the evaluation's mass range — `mH` 110–140 GeV,
`mZ` 81–101 GeV, `mW` 70–90 GeV, `mt` 160–185 GeV (that end
then bounds the interval, and the mass when the minimum lies at or beyond
it); the mass interval is the envelope of every grid region with
ΔlnL < 0.5, within that range.  A dataset without a usable grid is evaluated as
having no mass measurement.  The grid is not a GATE: a run without one
does not fail.
<!-- /part -->

<!-- part: measured_marginal -->
The evaluator measures `{{value:signal_poi}}`, `mH` and their uncertainties itself, from the
model you declare: `mH` from the mass grid below, and `{{value:signal_poi}}` from the same
grid — your model's likelihood in `{{value:signal_poi}}` at every grid mass, combined over
the evaluation's mass range as the grid section describes — not from the
fit at your reported mass alone.  `RESULT.json` must report what
`MODEL.json` gives: its own best-fit `{{value:signal_poi}}` at your reported mass, with
that fit's uncertainty.  The evaluator evaluates the workspace with
HistFactory semantics (as pyhf builds it), flooring each sample's expected
rate in each bin, before its multiplicative modifiers (its nominal plus
`histosys` shifts), and each bin's total expected rate at 1e-9, as CMS
Combine does.
{{value:model_verify_budget}}<!-- /part -->

<!-- part: mass_grid_marginal -->
The mass measurement and the signal strength are evaluated from a declared
mass grid: the same model at fixed mass hypotheses, in a directory
`MODEL_GRID/` next to `MODEL.json` containing `<poi>_<value>.json` files
(the mass POI's name of {{ref:result}} and the value in GeV, e.g. `mH_124.50.json`),
each a pyhf workspace; a single file `MODEL_GRID.json` mapping `"<value>"`
to the workspace is accepted equivalently.  Every grid workspace MUST be
the same model as `MODEL.json` on the same observations — identical
channels and bin counts, `MODEL.json`'s observations, identical background
samples and modifier data, identical nuisance and modifier set, an
identical measurement block (POI and every parameter setting) — with only
what depends on the mass (the signal templates) re-evaluated at that
point's fixed mass: that is what makes the fitted grid a profile
likelihood in the mass, and the evaluator checks it structurally.

Extent (checked).  The grid MUST reach both ends of the evaluation's mass
range — `mH` 110–140 GeV: first point at or below 110.5 GeV, last point at
or above 139.5 GeV — with no gap wider than 1 GeV between neighbouring
points inside the range.  Recommended: 0.5 GeV steps across the range,
refined to ≤ 0.25 GeV within a few times your mass uncertainty of the best
fit (60–90 points in all).  `MODEL.json` SHOULD be the grid's workspace at
the grid's in-range minimum of the fitted −2 ln L; the grid point at your
reported mass must reproduce `MODEL.json`'s own best fit (the same `{{value:signal_poi}}`
and likelihood, to numerical precision).  A `MODEL.json` that sits off the
grid's minimum is flagged on the report card, not charged.  A grid that
fails the extent is evaluated as described under "Fallback".

Content (checked).  Each grid workspace carries your model's own signal
template at that mass: the yield it predicts at `{{value:signal_poi}}` = 1, its shape and
its position are what your pipeline gives for a Standard-Model signal of
that mass, as they are in `MODEL.json` for your reported mass.  The
evaluator checks, point by point over the range:

- yield: the template's summed yield at `{{value:signal_poi}}` = 1 (every channel), relative
  to `MODEL.json`'s, follows the task's published σ × BR × acceptance
  curve (`yield_curve.json`, shipped with the task; relative to its value
  at your reported mass) within a factor 1.2 up to 5 GeV from your
  reported mass, the factor growing linearly in its logarithm to 1.5 at
  the ends of the range.  The part of the lineshape your binning leaves
  out near its edges is allowed for: each yield is divided by the
  fraction of your own template (the grid point nearest the middle of the
  range, scaled to that mass) inside your binning.  A point above this
  band is not rejected: it is evaluated as if its template were scaled
  down to the band's edge (its likelihood read at `{{value:signal_poi}}` divided by the
  excess factor), so a larger yield there earns nothing; a point below
  the band is kept as it is;
- smoothness: from one grid point to the next the logarithm of the yield
  changes by no more than the curve's own change plus 0.1 per GeV;
- position: the template's core centroid (within 4 GeV of its peak, from
  your `_meta.binning` edges) moves with the file's mass label: the
  straight line through the centroids against the labels has a slope
  between 0.8 and 1.2, and every centroid lies within ±1 GeV of it (an
  offset from the label, constant or growing with the mass, is allowed;
  over less than 4 GeV of labels the slope is taken as 1);
- no dead point: a template below 5 % of the curve's yield, or one whose
  fit drives `{{value:signal_poi}}` to its upper bound, is left out; more than one point in
  ten left out makes the grid unusable;
- shape: the template's core (within 4 GeV of its peak) has the width of
  your own lineshape — the grid point nearest the middle of the range, read
  in the same window — both its rms width and its peakedness (1/∫f², f the
  core normalised to one): the logarithm of their ratio stays within
  0.08 + 0.04 b of its median over the grid, allowing one common scaling
  of the width with the mass between constant and proportional to it, and
  within 0.04 + 0.06 b of its neighbours' within 1 GeV, b your median bin
  width in GeV within the core; and each channel's share of the
  template's yield stays within 0.15 of its median over the grid.  A
  template narrower or more peaked than your model's resolution at that
  mass, at the same yield, would claim a sensitivity the model does not
  have there.

A grid failing the smoothness, position, dead-point or shape checks is
unusable for `{{value:signal_poi}}` and takes the fallback, flagged; the yield band clips,
as above, and is flagged.  The evaluator runs the extent and content checks; the curve ships
with the task, so you can apply them to your grid yourself.

What the evaluator does with the grid.  It fits every in-range point with
`{{value:signal_poi}}` free (a point whose measurement fixes it is not a profile) and the
nuisances as the measurement block sets them; `mH` is scored from the mass
profile reconstructed from the grid, as before.  For `{{value:signal_poi}}` it also scans
each point's profile likelihood in `{{value:signal_poi}}`, at `{{value:signal_poi}}` ≥ 0 and within your
declared POI bounds (outside them your model calls the value impossible
and is taken at its word), and scores the signal strength on the density
in `{{value:signal_poi}}` obtained by combining these per-mass likelihoods over the mass
range with a fixed weight per unit mass — the same for every submission,
set by the reference analysis's expected sensitivity at each mass and by
nothing in your submission — that is, on your declared model's likelihood
marginalised over the mass range, rather than on the fit at your reported
mass alone.  A grid that states your model's sensitivity honestly at every
mass loses nothing by this; a grid that is insensitive where the signal may
lie, or whose bounds exclude `{{value:signal_poi}}` = 0 on a dataset without a signal, pays
for it.  For the mass, the grid is usable when it is `MODEL.json`'s model
as above, has at least 3 points, and covers your mass interval: on each side of the profile minimum, a ΔlnL =
0.5 crossing lies inside the grid, or the grid extends to that end of the
mass range (that end then bounds the interval, and the mass when the
minimum lies at or beyond it); the mass interval is the envelope of every
grid region with ΔlnL < 0.5, within that range.

Fallback.  A dataset whose grid fails the extent or the content checks, or
is not `MODEL.json`'s model, is evaluated for `{{value:signal_poi}}` on one mass alone — the
grid point where −2 ln L is lowest at `{{value:signal_poi}}` ≥ 0 within your bounds (among
points within 0.1 of it, the one with the widest `{{value:signal_poi}}` uncertainty) — with that point's exact
profile likelihood in `{{value:signal_poi}}` and no credit for the mass range left
uncovered (on a dataset without a signal that point is the one most in
favour of a signal); its mass measurement follows the rule above.  A
dataset without a grid is evaluated for `{{value:signal_poi}}` on `MODEL.json` alone, the
same way, and as having no mass measurement.  A dataset whose model checks
run on a `MODEL_COMBINE/` declaration (below) is evaluated for
`{{value:signal_poi}}` from this grid in the same way; without a grid its
`{{value:signal_poi}}` is not evaluated (a failed case).  The grid is not a GATE: a
run without one does not fail.
<!-- /part -->

<!-- part: size_caps -->
Size caps.  The evaluator reads at most 256 MiB of `MODEL.json`, 256 MiB of
`MODEL_GRID/` in all and 64 MiB of `EVENTS.json`; a larger file counts as a
failed run on that dataset.  The reference's are about 0.4 MiB, 25 MiB and
10 KiB.
<!-- /part -->

<!-- part: size_caps_marginal -->
Size caps.  The evaluator reads at most 256 MiB of `MODEL.json`, 512 MiB of
`MODEL_GRID/` in all and 64 MiB of `EVENTS.json`; a larger file counts as a
failed run on that dataset (a grid above its cap counts as no grid).  The
cap applies to the grid directory as a whole: a 0.5 GeV grid over the
range (61 workspaces) stays under it up to about 8 MiB per workspace.  The
reference's are about 0.3 MiB, 25 MiB (86 grid workspaces) and 10 KiB.
<!-- /part -->

<!-- part: combine -->
Optionally, a submission MAY also declare the same model in CMS combine
form: a directory `MODEL_COMBINE/` next to `MODEL.json`, containing either
`datacard.txt` plus every input it references by relative path (shape
files, workspace inputs), or a ready `workspace.root` in the layout
`text2workspace.py` produces (a `RooWorkspace` named `w` holding the
dataset `data_obs` and a `ModelConfig`).  Requirements:

- it describes the SAME observed data as `MODEL.json`'s `observations`
  (per channel, summed counts equal within rounding — checked
  structurally; a mismatch voids the whole check);
- the signal strength is combine's default POI `r`, with the same meaning
  as `mu` in `RESULT.json`;
- optionally, the mass is a parameter named `MH` — floating, via whatever
  combine-native construction (parametric or morphing shapes) the
  declaration builds into its shapes or workspace.  This is the one thing
  the pyhf form cannot express.

The evaluator builds the model (`text2workspace.py`) and then runs its
model checks on this declaration, with combine's own machinery, instead
of on `MODEL.json`: a `MultiDimFit --algo singles` best fit of `r` — of
(`r`, `MH`) simultaneously when the model depends on `MH` — whose fitted
`r` is compared with the reported `mu` under the same tolerance as the
`MODEL.json` refit, and whose fitted `MH` ± uncertainty is compared with
the reported mass `value`/`total`; `FitDiagnostics` (prefit/postfit
distributions and the nuisance pulls); a saturated `GoodnessOfFit`
(observed, and toy-calibrated when the evaluator runs toys); per-nuisance
impacts on `r`; and, when `MH` floats, a profile scan of `MH`.  It passes
no parameter range of its own: the declaration's ranges apply (a
datacard's `r` gets `text2workspace.py`'s default, [0, 20]).  None of it
is gated.  A declaration that does not build, does not describe the
same observed data, or on which combine cannot compute the goodness of
fit (the observed `GoodnessOfFit` fails, or its bins cannot be counted)
is recorded as such and the checks run on `MODEL.json` as without it;
absent, nothing changes.  A `workspace.root` declaration is accepted but
recorded as such — a workspace is more expressive than a datacard
(arbitrary RooFit constructs), so the two forms do not carry the same
level of structural scrutiny.
<!-- /part -->
