# EVAL_CONTRACT v2.2 — submission contract

This file is part of the input given to the analysis agent.  Everything below
is checked mechanically.
It is composed for your task from the parts its results take:
`RESULT.json` (§2) and the corrections each dataset's
`request.json` asks for: the corrections as functions (`corrections.json`;
§3), the request (§4) and the measurement behind
each correction (`MEASUREMENT.json`, `MEASUREMENT.npz`; §5).
Your task (PROMPT.md) says what to measure; DATASET.md describes the data.

## 1. Submission form

A submission is a directory whose root contains an executable `run.sh`.  The
harness freezes the directory into an immutable snapshot and executes, from a
clean unpack:

    ./run.sh <dataset_dir> <output_dir>

- `<dataset_dir>` has exactly the layout of the development dataset
  (DATASET.md).  The withheld datasets may differ from the development
  dataset in signal content and detector response.
  A dataset's `request.json` names the corrections requested on it
  (§4); the request may differ from dataset to dataset.
- `<output_dir>` exists and is empty; write `RESULT.json` there (GATE),
  and the other files your results take (the sections below name them).
  Output files are read only as regular files: a symlink there is never
  followed, and a symlinked output file counts as malformed.
- Exit code 0 signals success.  Non-zero exit, timeout, or a missing or
  malformed GATE file marks the run failed.
- Repeated execution on the same dataset must be deterministic (fix your
  random seeds; keep parallel reductions in a fixed order).  This is
  checked by the self-check only (`--determinism` compares the reported
  values between two runs); the evaluator runs each dataset once.

Hygiene, checked before freezing: no data files (`.npy`, `.root`, `.parquet`,
…), no committed outputs, no file named `truth.json`, `manifest.json` or
`suite_truth.json` (the evaluator's own names), no symlinks, ≤ 5 MB per file,
≤ 50 MB total.
Installed packages are refused: no `pip install --target` output and no
`site-packages`, `*.dist-info` or `*.egg-info` tree anywhere in the
submission.  Declare what you need in `pixi.toml` + `pixi.lock` (below).

Runtime: your submission runs in its own pixi environment.
The base environment is Python 3.14 with `numpy`, `uproot`, `awkward` and
`vector`.  For this analysis it also has `correctionlib`, ROOT (PyROOT) and CMS Combine
(`combine`, `text2workspace.py`).  To use anything else — anything conda-forge or PyPI carries
(pixi resolves both: `pixi add` / `pixi add --pypi`) — put a `pixi.toml`
**and its `pixi.lock`**
at the submission root (create them with `pixi init` / `pixi add` inside
your submission directory); the harness materializes the environment from
your lockfile before the runs.  A manifest without a lockfile is a hygiene
violation; an environment that fails to build within 1500 s, or whose
installed size exceeds 4 GB, fails every run.  The runs themselves have
no network; the pipeline may read only `<dataset_dir>` and its own directory.
Budgets: each evaluation dataset is run alone, on 4 CPUs (enforced as the
process's CPU affinity), with a 16 GiB memory cap (the run's total resident
memory, including the files it writes to `/tmp` — where `$TMPDIR` points —
which is held in memory; exceeding it kills the run) and an 1800 s wall
clock (the run is killed at the limit); within it, faster is better.
An evaluation dataset can carry more integrated luminosity than the public
development dataset (DATASET.md says how much), and the time and memory your
pipeline spends on the data grow with it — leave headroom.  The self-check runs on the
development dataset with the evaluation's wall clock, CPU count and network
isolation, prints its limits and reports the run's wall time and peak
resident memory (the whole process tree and its `/tmp` files) against the
16 GiB cap: it fails when the peak exceeds the cap (the evaluation would
kill that run) and warns above half of it; it measures memory but does not stop the run at the
cap.

## 2. RESULT.json

A JSON object (GATE).  Your answers are in `corrections.json`
(§3); `RESULT.json` carries the parameter covariances of
corrections (§3):

    {
      "covariances": { "<correction name>": [[f, ...], ...] },
      "joint_covariances": [ ... ]
    }

Both fields are optional.  A `nuisances` log, if you add one, is recorded
as given and not checked; other extra keys are allowed and ignored.

## 3. corrections.json — corrections as functions

A correction relating the data to the simulation — a momentum scale, a
resolution or an efficiency that depends on the object — can be reported as
a function.  When the dataset
contains `request.json` (§4), write `corrections.json` next to `RESULT.json`
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
it is requested for (§4), without `phi` the same at every azimuth, and a
constant the same throughout its domain.  The kinematic inputs are those
of the object in the simulation: every correction is a function of the
simulated object it is applied to (§4).

A reported correction is evaluated at the leptons of the simulation's Z →
ℓℓ candidates within its domain (§4): per event the opposite-
charge pair of one flavour's collection closest to the Z mass, within 10
GeV of it, both leptons above 10 GeV in pT, no identification required; a
trigger leg only at the leptons that pass its `probe_selection`.

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
10 % throughout the domain (a resolution's in its extra variance, §4), or only the band is read.  A
`"covariances"` that is not an object, or a `"joint_covariances"` that is not a list, fails the run.  Each of
these counts as its correction not reported (a joint covariance's, every correction it lists): a matrix that
is not the symmetric, positive semi-definite matrix of numbers of the right size; a source name outside these
rules, or in two components of one correction; a `variation` input without `"nominal"`, or with a value other
than `"nominal"`, `"up"`, `"down"`, `"up_<source>"` and `"down_<source>"`; an `"up"` (`"up_<source>"`)
without its `"down"` (`"down_<source>"`), or the reverse; a joint covariance whose names are not distinct, or
name a correction that is not present once in `corrections.json`.

Every correction must evaluate to a finite value, and a finite
uncertainty, everywhere in its domain (§4): every `pt` above the domain's
`pt_min` (with no upper end), every |`eta`| below its `abs_eta_max`, every
`phi`, both charges; for a resolution, the extra variance it stands for
(§4) must be finite too.  Outside its domain a correction is never
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

## 4. request.json — requested corrections

A dataset's `request.json`, when it has one, lists the corrections
requested on that dataset:

    {"request_version": 1,
     "corrections": {"<name>": {"kind": "<kind>", "domain": {"<flavour>": {"pt_min": f, "abs_eta_max": f}, ...},
                                ...}, ...}}

Every entry carries `kind` and `domain`: the object types (`flavour`
values of §3) the correction is requested for, each with the region where
it is defined, objects with pT > `pt_min` and |η| < `abs_eta_max`.  The
kinds and their further fields:

| `kind` | fields | entries |
|---|---|---|
| `scale` | — | `scale` |
| `resolution` | — | `res_pt`, `res_eta`, `res_phi` |
| `id` | `selection` (string): `"<branch>"` for a boolean branch, `"<branch> >= <n>"` for an integer one | one per requested selection |
| `trigger` | `path`, `filter_bit` (int), `online_pt_min` (GeV), `delta_r_max`, `trigobj_id` (int), `probe_selection` (string, as `selection`) | one per requested leg |

An example request:

    {"request_version": 1, "corrections": {
      "Electron_cutBased_ge4": {"kind": "id", "domain": {"electron": {"pt_min": 10.0, "abs_eta_max": 2.5}},
                                "selection": "Electron_cutBased >= 4"},
      "Electron_mvaFall17V2Iso_WP90": {"kind": "id", "domain": {"electron": {"pt_min": 10.0, "abs_eta_max": 2.5}},
                                       "selection": "Electron_mvaFall17V2Iso_WP90"},
      "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL__Ele23": {"kind": "trigger",
        "domain": {"electron": {"pt_min": 25.0, "abs_eta_max": 2.5}},
        "path": "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL", "filter_bit": 0, "online_pt_min": 23.0,
        "delta_r_max": 0.3, "trigobj_id": 11, "probe_selection": "Electron_cutBased >= 4"},
      "HLT_Ele27_WPTight_Gsf": {"kind": "trigger", "domain": {"electron": {"pt_min": 30.0, "abs_eta_max": 2.5}},
        "path": "HLT_Ele27_WPTight_Gsf", "filter_bit": 1, "online_pt_min": 27.0,
        "delta_r_max": 0.3, "trigobj_id": 11, "probe_selection": "Electron_cutBased >= 4"},
      "HLT_IsoMu24": {"kind": "trigger", "domain": {"muon": {"pt_min": 26.0, "abs_eta_max": 2.4}},
        "path": "HLT_IsoMu24", "filter_bit": 1, "online_pt_min": 24.0,
        "delta_r_max": 0.3, "trigobj_id": 13, "probe_selection": "Muon_tightId"},
      "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL__Mu17": {"kind": "trigger", "domain": {"muon": {"pt_min": 20.0, "abs_eta_max": 2.4}},
        "path": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL", "filter_bit": 0, "online_pt_min": 17.0,
        "delta_r_max": 0.3, "trigobj_id": 13, "probe_selection": "Muon_tightId"},
      "Muon_softId": {"kind": "id", "domain": {"muon": {"pt_min": 15.0, "abs_eta_max": 2.4}}, "selection": "Muon_softId"},
      "Muon_tightId": {"kind": "id", "domain": {"muon": {"pt_min": 15.0, "abs_eta_max": 2.4}}, "selection": "Muon_tightId"},
      "res_eta": {"kind": "resolution", "domain": {"electron": {"pt_min": 20.0, "abs_eta_max": 2.5},
                                                   "muon": {"pt_min": 20.0, "abs_eta_max": 2.4}}},
      "res_phi": {"kind": "resolution", "domain": {"electron": {"pt_min": 20.0, "abs_eta_max": 2.5},
                                                   "muon": {"pt_min": 20.0, "abs_eta_max": 2.4}}},
      "res_pt": {"kind": "resolution", "domain": {"electron": {"pt_min": 20.0, "abs_eta_max": 2.5},
                                                  "muon": {"pt_min": 20.0, "abs_eta_max": 2.4}}},
      "scale": {"kind": "scale", "domain": {"electron": {"pt_min": 20.0, "abs_eta_max": 2.5},
                                            "muon": {"pt_min": 20.0, "abs_eta_max": 2.4}}}}}

All corrections relate the data to the simulation for the same true
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

The selections and trigger legs a request can name, in alphabetical order
(name: `selection`, or `path`):

- muon identification: `Muon_highPtId_ge1`
  (`Muon_highPtId >= 1`), `Muon_looseId`, `Muon_mediumId`,
  `Muon_mediumPromptId`, `Muon_mvaId_ge1` (`Muon_mvaId >= 1`),
  `Muon_pfIsoId_ge4` (`Muon_pfIsoId >= 4`), `Muon_softId`,
  `Muon_softMvaId`, `Muon_tightId`, `Muon_tkIsoId_ge1`
  (`Muon_tkIsoId >= 1`), `Muon_triggerIdLoose`;
- electron identification: `Electron_cutBased_HEEP`,
  `Electron_cutBased_ge1` (`Electron_cutBased >= 1`),
  `Electron_cutBased_ge2` (`Electron_cutBased >= 2`),
  `Electron_cutBased_ge3` (`Electron_cutBased >= 3`),
  `Electron_cutBased_ge4` (`Electron_cutBased >= 4`),
  `Electron_mvaFall17V2Iso_WP80`, `Electron_mvaFall17V2Iso_WP90`,
  `Electron_mvaFall17V2Iso_WPL`, `Electron_mvaFall17V2noIso_WP80`,
  `Electron_mvaFall17V2noIso_WP90`, `Electron_mvaFall17V2noIso_WPL`;
- muon trigger legs, single-lepton paths:
  `HLT_IsoMu22`, `HLT_IsoMu24`, `HLT_IsoMu27`, `HLT_IsoTkMu22`,
  `HLT_IsoTkMu24`; double-lepton path:
  `HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL__Mu17`,
  `HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL__Mu8`;
- electron trigger legs, single-lepton paths:
  `HLT_Ele25_eta2p1_WPTight_Gsf`, `HLT_Ele27_WPTight_Gsf`,
  `HLT_Ele27_eta2p1_WPLoose_Gsf`; double-lepton path:
  `HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL__Ele12`,
  `HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL__Ele23`.

## 5. MEASUREMENT.json — the measurement behind a reported correction

A dataset with a request may also take, optionally, `MEASUREMENT.json` and `MEASUREMENT.npz` next to
`corrections.json`: the binned measurement each reported correction comes from, for each object type it is reported
for, and the object pairs behind every count.  A declared measurement's pairs are checked against the dataset and
the measurement is recounted from them, and the result is recorded; a correction is evaluated whether or not it is
declared.  `MEASUREMENT.json` is at most 20 MB, `MEASUREMENT.npz` at most 1 GB (4 GB unpacked).

    {"measurements": {"<id>": {<measurement>}, ...},
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
passes the request's `probe_selection` — has the leg's trigger object (§4; a probe within a relative 10⁻⁵ of the
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
in a histogram once the corrections the measurement lists are applied to its objects (§4): a selection on their
kinematics must be loose enough that no pair left out could count once corrected.

## 6. Self-check

`sbh check` runs your submission on the development dataset as the
evaluation does and checks every output file it takes against this
contract: `RESULT.json` and
`corrections.json` — every reported correction against correctionlib's
schema, its inputs and its uncertainty, and that it evaluates as required
throughout its domain, on a grid of points in it from just above `pt_min`
to 10⁴ GeV and up to just below `abs_eta_max`, every `phi`, both charges,
each requested object type — and the measurement declared for each
(§5: structure, recount, and the pairs against the
development dataset).  Where correctionlib is not importable, the
self-check validates the structure of `corrections.json` only and says
that its evaluation was NOT VALIDATED HERE.
