<!-- The corrections kind's parts: corrections as functions (corrections.json, its uncertainty components and domain),
the request (request.json: the kinds, the names a request can carry) and the declared measurement behind each reported
correction (MEASUREMENT.json / MEASUREMENT.npz).
A template (the unified tasks/EVAL_CONTRACT.in.md, or a task's own EVAL_CONTRACT.in.md) includes them as
{{core:<part>}}, {{hep:<part>}}, {{kind:<part>}} (the task's own kind's) or {{kind.<name>:<part>}} (a named kind's);
{{value:<name>}} is filled from the task (Task.contract_values; the unified contract's own names win there).  Each
part is the exact text between its opening marker line and its closing marker (`python -m sbh.contract render`
writes tasks/<task>/EVAL_CONTRACT.md). -->

<!-- part: functions -->
A correction relating the data to the simulation — a momentum scale, a
resolution or an efficiency that depends on the object — can be reported as
a function.  When the dataset
contains `request.json` ({{ref:request}}), write `corrections.json` next to `RESULT.json`
(GATE) with the corrections you report, each under its name in the request,
on every dataset.  A correction's form is yours to choose — a constant,
binned or a formula, one function for several object types or one per type
— and so is the detail of its uncertainty: it needs one component (below),
every further one is optional.  `RESULT.json` and `corrections.json` of a
request are each at most 20 MB.

`corrections.json` is a correctionlib `CorrectionSet` in JSON, schema
version 2: `{"schema_version": 2, "corrections": [<Correction>, ...]}`
(correctionlib's own schema, as correctionlib 2.9.0 reads it).  A file that
is missing, not a JSON object with `"schema_version": 2` and a
`"corrections"` list, or larger than 20 MB fails the run.

Each reported correction is checked on its own, against correctionlib's
schema for a `Correction`, and evaluated with correctionlib: a problem in
one correction affects that correction only.  Entries under other names,
and any `compound_corrections`, are ignored; two entries under a requested
name count as that correction not reported.

A correction's `inputs` are chosen from this set, under exactly these names
and types (use any subset, in any order):

| input | type | value |
|---|---|---|
| `pt` | real | the object's transverse momentum in GeV (e.g. `Muon_pt` / `Electron_pt`) |
| `eta` | real | its pseudorapidity (e.g. `Muon_eta` / `Electron_eta`) |
| `abs_eta` | real | \|`eta`\| |
| `phi` | real | its azimuth (e.g. `Muon_phi` / `Electron_phi`), in [−π, π] |
| `charge` | int | its charge, ±1 |
| `flavour` | string | its type: `"electron"`, `"muon"`, `"tau"`, `"photon"` or `"jet"` |
| `variation` | string | `"nominal"`, `"up"` or `"down"` (below) |

A correction is the same for every value of an input it does not take:
without a `flavour` input it is the same function for every object type
it is requested for ({{ref:request}}), without `phi` the same at every azimuth, and a
constant the same throughout its domain.  The kinematic inputs are those
of the object in the simulation: every correction is a function of the
simulated object it is applied to ({{ref:request}}).
{{value:correction_population}}
Every reported correction states its uncertainty, made of independent components, each a 1σ shift of the
correction:

- the **total band** — with the `variation` input, `"up"` and `"down"` (the correction itself is
  `"nominal"`): the component |up − down| / 2;
- a **source** — with the `variation` input, `"up_<source>"` and `"down_<source>"`: the shift
  (up_<source> − down_<source>) / 2 of that source alone;
- a **parameter covariance** — `RESULT.json["covariances"]["<name>"]`: the covariance matrix of the
  parameters of the correction's formulas, either one matrix or an object of matrices by source
  (`{"<source>": matrix, ...}`).  The parameters are the `parameters` of every `formula` and `formularef`
  node of its `"nominal"` content (of its whole `data` when it has no `variation` input), not those of
  `generic_formulas` entries, concatenated in depth-first order — a binning's or multibinning's `content`
  in order, then its `flow`; a category's values in order, then its `default`; a transform's `rule`, then
  its `content`.  A matrix is P × P for those P parameters (1 ≤ P ≤ 500), symmetric and positive
  semi-definite; its component is the linear propagation √(gᵀ C g), g the gradient of the correction with
  respect to the parameters, computed numerically;
- a **joint covariance** — an entry `{"source": "<source>", "corrections": ["<name>", ...], "matrix":
  [[f, ...], ...]}` of the list `RESULT.json["joint_covariances"]`: the covariance of the listed
  corrections' parameters taken together (their parameters concatenated in the listed order), for
  corrections that come from one fit.  It lists distinct names, each of a correction present once in
  `corrections.json`.

Components of different sources are independent.  A source is one effect wherever it appears: its shifts
are fully correlated across corrections, object types and points, and a joint covariance correlates the
parameters it lists; split a source into two names to make it partly correlated.  A source name is a class,
or a class and a label (`<class>_<label>`, e.g. `bkg_window`; a label is letters and digits, in parts joined
by single underscores): `stat` (the data's statistics), `mcstat` (the
simulation's), `bkg` (background), `tag` (the selection of the measurement's reference objects, such as the
tags of a tag-and-probe), `model` (the fitted model: shapes, windows, binning), `theory`, `other`; a matrix
without a source name is `stat`.  The total uncertainty is the total band when the correction has one, else
the quadrature sum of its components; when it has both, the quadrature sum must agree with the band within
10 % throughout the domain (a resolution's in its extra variance, {{ref:request}}), or only the band is read.  A
`"covariances"` that is not an object, or a `"joint_covariances"` that is not a list, fails the run.  Each of
these counts as its correction not reported (a joint covariance's, every correction it lists): a matrix that
is not the symmetric, positive semi-definite matrix of numbers of the right size; a source name outside these
rules, or in two components of one correction; a `variation` input without `"nominal"`, or with a value other
than `"nominal"`, `"up"`, `"down"`, `"up_<source>"` and `"down_<source>"`; an `"up"` (`"up_<source>"`)
without its `"down"` (`"down_<source>"`), or the reverse; a joint covariance whose names are not distinct, or
name a correction that is not present once in `corrections.json`.

Every correction must evaluate to a finite value, and a finite
uncertainty, everywhere in its domain ({{ref:request}}): every `pt` above the domain's
`pt_min` (with no upper end), every |`eta`| below its `abs_eta_max`, every
`phi`, both charges; for a resolution, the extra variance it stands for
({{ref:request}}) must be finite too.  Outside its domain a correction is never
evaluated.
A binning that ends inside the domain must not raise there: give it
`"flow": "clamp"` (or a content node), not `"error"`.

A reported correction that takes an input outside the set above (or under
another type, or twice), states no uncertainty component, or does not
evaluate as required for an object type counts as not reported for that
type on that dataset.

Examples of the forms (names, inputs, binning and numbers are illustrative
only; use the requested names and whatever form your measurement
supports):

    {"schema_version": 2, "corrections": [
      {"name": "example_constant", "version": 1, "output": {"name": "value", "type": "real"},
       "inputs": [{"name": "variation", "type": "string"}],
       "data": {"nodetype": "category", "input": "variation", "content": [
         {"key": "nominal", "value": 1.002}, {"key": "up", "value": 1.004},
         {"key": "down", "value": 1.000}]}},
      {"name": "example_binned", "version": 1, "output": {"name": "value", "type": "real"},
       "inputs": [{"name": "eta", "type": "real"}, {"name": "variation", "type": "string"}],
       "data": {"nodetype": "category", "input": "variation", "content": [
         {"key": "nominal", "value": {"nodetype": "binning", "input": "eta",
           "edges": [-2.5, -1.2, 0.0, 1.2, 2.5], "content": [0.990, 0.995, 0.993, 0.988],
           "flow": "clamp"}},
         {"key": "up", "value": {"nodetype": "binning", "input": "eta",
           "edges": [-2.5, -1.2, 0.0, 1.2, 2.5], "content": [0.993, 0.997, 0.995, 0.991],
           "flow": "clamp"}},
         {"key": "down", "value": {"nodetype": "binning", "input": "eta",
           "edges": [-2.5, -1.2, 0.0, 1.2, 2.5], "content": [0.987, 0.993, 0.991, 0.985],
           "flow": "clamp"}}]}},
      {"name": "example_formula", "version": 1, "output": {"name": "value", "type": "real"},
       "inputs": [{"name": "phi", "type": "real"}],
       "data": {"nodetype": "formula", "parser": "TFormula", "variables": ["phi"],
                "expression": "[0] + [1]*cos(x) + [2]*sin(x)", "parameters": [0.001, 0.0002, -0.0001]}}]}

with, for the formula, `RESULT.json`:

    {"covariances": {"example_formula": [[1e-8, 0, 0], [0, 4e-8, 0], [0, 0, 4e-8]]}}
<!-- /part -->

<!-- part: request -->
A dataset's `request.json`, when it has one, lists the corrections
requested on that dataset:

    {"request_version": 1,
     "corrections": {"<name>": {"kind": "<kind>", "domain": {"<flavour>": {"pt_min": f, "abs_eta_max": f}, ...},
                                ...}, ...}}

Every entry carries `kind` and `domain`: the object types (`flavour`
values of {{ref:functions}}) the correction is requested for, each with the region where
it is defined, objects with pT > `pt_min` and |η| < `abs_eta_max`.  The
kinds and their further fields:

| `kind` | fields | entries |
|---|---|---|
| `scale` | — | `scale` |
| `resolution` | — | `res_pt`, `res_eta`, `res_phi` |
| `id` | `selection` (string): `"<branch>"` for a boolean branch, `"<branch> >= <n>"` for an integer one | one per requested selection |
| `trigger` | `path`, `filter_bit` (int), `online_pt_min` (GeV), `delta_r_max`, `trigobj_id` (int), `probe_selection` (string, as `selection`) | one per requested leg |

{{value:request_example}}All corrections relate the data to the simulation for the same true
object, as functions of the object's kinematics in the simulation:

- `scale` — momentum scale of the data relative to the simulation: the
  data reconstruct, on average, (1 + value) times the transverse momentum
  the simulation reconstructs for the same true object.  0 = nominal.
- `res_pt` — extra relative transverse-momentum resolution in the data:
  smearing each simulated object's pT with a Gaussian of relative width
  `value` reproduces the data (it adds to the simulated resolution in
  quadrature).  0 = nominal.
- `res_eta`, `res_phi` — the same for the object's direction: the absolute
  width, in η and in φ (radians), of the extra Gaussian smearing.
  0 = nominal.

  A resolution's value stands for the extra variance value × |value|; a
  negative value stands for a negative extra variance (data narrower than
  the simulation).  Its nominal value and both ends of its uncertainty band
  — `"up"` and `"down"`, or the value ± its propagated uncertainty — are
  each mapped through value × |value|, and the extra variance's uncertainty
  is half the distance between the two mapped ends.  The extra variance is
  the whole extra spread of the data at the object's kinematics, tails
  included, not that of a fitted Gaussian core.
- an identification selection (`"kind": "id"`) — data/simulation ratio of
  the efficiency with which a reconstructed object (an entry of the
  `Muon`, `Electron`, … collection) passes `selection`.  1 = nominal.
- a trigger leg (`"kind": "trigger"`; one leg of the HLT path `path`) —
  data/simulation ratio of the efficiency of the leg for a reconstructed
  object that passes `probe_selection`: the probability that a `TrigObj`
  with `TrigObj_id` == `trigobj_id` lies within ΔR < `delta_r_max` of the
  object, with bit `filter_bit` of `TrigObj_filterBits` set (bit 0 = the
  least significant) and `TrigObj_pt` ≥ `online_pt_min`.  It is the leg's
  own efficiency, as measured on objects whose event selection does not
  depend on their own trigger objects.  1 = nominal.
{{value:request_names}}<!-- /part -->

<!-- part: measurement_required -->
A dataset with a request also takes `MEASUREMENT.json` and `MEASUREMENT.npz` next to `corrections.json`: the
binned measurement each reported correction comes from, for each object type it is reported for, and the object
pairs behind every count.  The pairs are checked against the dataset and the measurement is recounted from them; a
reported correction whose measurement is missing or malformed, or fails a check, counts as not reported for that
type.  `MEASUREMENT.json` is at most 20 MB, `MEASUREMENT.npz` at most 1 GB (4 GB unpacked).<!-- /part -->

<!-- part: measurement_optional -->
A dataset with a request may also take, optionally, `MEASUREMENT.json` and `MEASUREMENT.npz` next to
`corrections.json`: the binned measurement each reported correction comes from, for each object type it is reported
for, and the object pairs behind every count.  A declared measurement's pairs are checked against the dataset and
the measurement is recounted from them, and the result is recorded; a correction is evaluated whether or not it is
declared.  `MEASUREMENT.json` is at most 20 MB, `MEASUREMENT.npz` at most 1 GB (4 GB unpacked).<!-- /part -->

<!-- part: measurement -->
{{value:measurement_opening}}

{{value:measurement_budget}}    {"measurements": {"<id>": {<measurement>}, ...},
     "corrections": {"<name>": {"<flavour>": "<id>", ...}, ...}}

An efficiency measurement serves one identification or trigger correction for one object type; a lineshape
measurement serves the `scale` and resolution corrections of one object type that it lists.

**Pairs.**  `MEASUREMENT.npz` (numpy's `.npz`, from `numpy.savez` or `numpy.savez_compressed`, without pickled
objects) holds named lists of object pairs, in the data and in the simulation.  The arrays of a list `<list>` are
named `<list>.<sample>.<field>` with `<sample>` `data` or `sim`, one element per pair, and
`<list>.<sample>.files` names the sample's files (strings, each once: a name in `data/`, or a path below `mc/`):

| field | type | the pair's |
|---|---|---|
| `file` | int | file: an index into `<list>.<sample>.files` |
| `entry` | int | entry index in that file's `Events` tree |
| `first`, `second` | int | objects: indices in the object type's collection (`Muon`, `Electron`, …) — the tag and the probe of a tag-and-probe, or the two objects of a pair |
| `weight` | real | event weight, simulation only: within each process of `mc/`, its `genWeight` times one positive factor |
| `x_<variable>` | real | value the pair is binned by, for each variable its efficiency measurements bin in |
| `pass_<id>` | bool | whether the probe passes, for each efficiency measurement `<id>` that uses the list |

Every listed event is an event of the dataset and every object an object of that event; a pair's two objects
differ, and a list holds each pair at most once.

**Efficiency measurement** — the probes (`second`) of a list's pairs, binned:

    {"type": "efficiency", "pairs": "<list>",
     "bins": {"<variable>": [e0, e1, ..., eN], ...},
     "data": {"pass": [n, ...], "fail": [n, ...]},
     "simulation": {"pass": [w, ...], "fail": [w, ...], "pass_w2": [w2, ...], "fail_w2": [w2, ...]}}

`bins` names one or more variables of the pairs (`x_<variable>`), each with its edges; the bins are every
combination of one interval per variable, the last variable varying fastest, and a pair outside every bin is not
counted.  Per bin: the data's passing and failing pairs, and the simulation's sums of the weights of its passing
and failing pairs and of their squares.  Each pair's `pass_<id>` must be what the dataset holds: for an
identification, whether its probe passes the request's `selection`; for a trigger leg, whether its probe — which
passes the request's `probe_selection` — has the leg's trigger object ({{ref:request}}; a probe within a relative 10⁻⁵ of the
leg's ΔR or pT threshold may be flagged either way).

**Lineshape measurement** — histograms of a list's pairs:

    {"type": "lineshape", "pairs": "<list>", "corrections": ["scale", "res_pt", ...],
     "selection": {"pt_min": f, "abs_eta_max": f},
     "histograms": [{"variable": "<pair variable>", "edges": [e0, ..., eN],
                     "categories": {"<pair variable>": [c0, ..., cK], ...},
                     "data": [[n, ...], ...]}, ...]}

The pair variables are `mass` (the invariant mass of the two objects taken as massless, GeV), `pt_pair` (the
pair's transverse momentum), `delta_eta` (|η₁ − η₂|), `delta_phi` (|Δφ| in [0, π]), `abs_eta_max`,
`abs_eta_min`, `pt_max` and `pt_min`, and each object's own `first_<v>` and `second_<v>`, `<v>` one of `pt`,
`eta`, `abs_eta`, `phi`, `charge`.  A histogram bins its `variable` in `edges`, in categories that bin other pair
variables as `bins` does (without `categories`, one category); `data` holds the data's counts per category and
bin.  A pair counts when both objects have pT > `pt_min` and |η| < `abs_eta_max`, its variables lie in a
category and its `variable` in the edges; a pair may count in several histograms.  In the data the objects are
taken as the dataset holds them, the values computed in double precision (a pair within a relative 10⁻⁵ of an
edge or a threshold may be counted on either side).  In the simulation, the list holds every pair that could count
in a histogram once the corrections the measurement lists are applied to its objects ({{ref:request}}): a selection on their
kinematics must be loose enough that no pair left out could count once corrected.
<!-- /part -->
