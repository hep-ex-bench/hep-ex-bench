# EVAL_CONTRACT v2.2 — submission contract

{{core:intro}}It is composed for your task from the parts its results take:
<!-- if: pois -->
`RESULT.json` with the physics quantities you measure ({{ref:result}}),
their statistical model (`MODEL.json`, `EVENTS.json`; {{ref:model}}) and the
log of its nuisance parameters ({{ref:nuisances}}).
<!-- /if -->
<!-- if: corrections -->
`RESULT.json` ({{ref:result}}) and the corrections each dataset's
`request.json` asks for: the corrections as functions (`corrections.json`;
{{ref:functions}}), the request ({{ref:request}}) and the measurement behind
each correction (`MEASUREMENT.json`, `MEASUREMENT.npz`; {{ref:measurement}}).
<!-- /if -->
<!-- if: scaled_model -->
`RESULT.json` with the physics quantities you measure as confidence
intervals ({{ref:result}}), their statistical model (`MODEL.json`,
`EVENTS.json`; {{ref:model}}) and the log of its nuisance parameters
({{ref:nuisances}}).
<!-- /if -->
Your task (PROMPT.md) says what to measure; DATASET.md describes the data.

## {{sec:submission}}. Submission form

{{core:submission}}  The withheld datasets may differ from the development
  dataset in signal content and detector response.
<!-- if: corrections -->
  A dataset's `request.json` names the corrections requested on it
  ({{ref:request}}); the request may differ from dataset to dataset.
<!-- /if -->
{{core:outputs}}
{{core:hygiene}}
{{core:runtime}}{{core:budgets}}An evaluation dataset can carry more integrated luminosity than the public
development dataset (DATASET.md says how much), and the time and memory your
pipeline spends on the data grow with it — leave headroom.{{value:datasets_note}}  {{core:selfcheck_memory}}{{value:selfcheck_note}}{{core:selfcheck_memory_end}}
## {{sec:result}}. RESULT.json

<!-- if: pois -->
A JSON object (GATE) with these fields:

    {
      "pois": [
        {"name": "<poi>", "value": f, "stat": f, "syst": f, "total": f}, ...
      ],
      "significance_obs": f,          # observed local significance, in sigma
      "discovery": bool,              # your discovery claim
      "nuisances": { "<name>": {"value": f, "unc": f, "note": "..."}, ... },   # {{ref:nuisances}}
      "quality": { "<name>": {"value": f, "note": "..."}, ... }
    }

`pois` — the physics quantities you measure, one entry each, under these
names:

| name | quantity |
|---|---|
| `mu` | signal strength: the measured signal cross section over its Standard-Model prediction (for a signal with a mass, at the measured mass) |
| `mu_<proc>` | the same for one production process |
| `xsec_<proc>` | a production cross section, in pb, times the branching fraction into the final state you measure |
| `mH`, `mZ`, `mW`, `mt` | the mass of the Higgs boson, the Z boson, the W boson, the top quark, in GeV |
| `width_H`, `width_Z`, `width_W`, `width_t` | the corresponding total widths, in GeV |

`<proc>` is a process name of {{ref:nuisances}}.  All values finite; `total` > 0 and
consistent with `stat ⊕ syst`.  An analysis that searches for a signal also
reports `significance_obs` and `discovery`, and may report
`"significance_exp"`: the expected (Asimov) local significance of a
Standard-Model signal (`mu` = 1) under your calibrated model — the
sensitivity of your analysis, as opposed to the observed fluctuation.
Report every field your results take on every dataset, including one with
no visible signal (then your best-fit `mu` ≈ 0 with an honest uncertainty,
and a mass at its most signal-like value with its uncertainty).  A POI your
task does not ask for is recorded as given.

`nuisances` — the log of the systematic effects and corrections of your
analysis, {{ref:nuisances}}.  Extra keys are allowed and ignored.

{{kind.binned_likelihood:quality}}
<!-- /if -->
<!-- if: corrections -->
A JSON object (GATE).  Your answers are in `corrections.json`
({{ref:functions}}); `RESULT.json` carries the parameter covariances of
corrections ({{ref:functions}}):

    {
      "covariances": { "<correction name>": [[f, ...], ...] },
      "joint_covariances": [ ... ]
    }

Both fields are optional.  A `nuisances` log, if you add one, is recorded
as given and not checked; other extra keys are allowed and ignored.

<!-- /if -->
<!-- if: intervals -->
{{kind.scan_likelihood:result}}
<!-- /if -->
<!-- if: scaled_model -->
{{kind.scan_likelihood:pieces}}
<!-- /if -->
<!-- if: intervals -->
{{kind.scan_likelihood:result_optional}}
<!-- /if -->
<!-- if: model -->
## {{sec:model}}. MODEL.json — the statistical model

{{kind.binned_likelihood:model}}{{value:observable}} you binned{{kind.binned_likelihood:binning}}Bin {{value:observable}} in GeV and declare the edges in `_meta.binning`;
the bins must cover {{value:gof_window}} GeV, where the evaluator tests the model
against your data.

{{kind.binned_likelihood:gate}}
{{kind.binned_likelihood:event_list}}
<!-- /if -->
<!-- if: mu_point -->
{{kind.binned_likelihood:measured}}
{{kind.binned_likelihood:mass_grid}}
{{kind.binned_likelihood:size_caps}}
<!-- /if -->
<!-- if: mu_marginal -->
{{kind.binned_likelihood:measured_marginal}}
{{kind.binned_likelihood:mass_grid_marginal}}
{{kind.binned_likelihood:size_caps_marginal}}
<!-- /if -->
<!-- if: model -->
{{kind.binned_likelihood:combine}}
<!-- /if -->
<!-- if: scaled_model -->
## {{sec:model}}. MODEL.json — the statistical model

{{kind.scan_likelihood:model}}{{value:observable}} you binned{{kind.binned_likelihood:binning}}Bin {{value:observable}} in GeV and declare the edges in `_meta.binning`;
the bins must cover {{value:gof_window}} GeV, where the evaluator tests the model
against your data.

{{kind.scan_likelihood:scaling}}
{{kind.scan_likelihood:gate}}
{{kind.binned_likelihood:event_list}}
{{kind.scan_likelihood:measured}}
{{kind.scan_likelihood:size_caps}}
<!-- /if -->
<!-- if: nuisances -->
## {{sec:nuisances}}. Nuisance parameters: names and log

The names below cover every effect the evaluation reads, whether or not your
analysis has it: which effects apply is yours to decide.

{{hep:log}}
{{hep:names}}
{{hep:scale_res}}{{hep:eff}}{{value:wp}}{{hep:more}}{{hep:analysis}}
{{hep:population}}
{{hep:parts}}
<!-- /if -->
<!-- if: corrections -->
## {{sec:functions}}. corrections.json — corrections as functions

{{kind.corrections:functions}}
## {{sec:request}}. request.json — requested corrections

{{kind.corrections:request}}
## {{sec:measurement}}. MEASUREMENT.json — the measurement behind a reported correction

{{kind.corrections:measurement}}
<!-- /if -->
## {{sec:selfcheck}}. Self-check

`sbh check` runs your submission on the development dataset as the
evaluation does and checks every output file it takes against this
contract: `RESULT.json`<!-- if: model|scaled_model -->, `MODEL.json` and `EVENTS.json` ({{ref:model}})<!-- /if --><!-- if: corrections --> and
`corrections.json` — every reported correction against correctionlib's
schema, its inputs and its uncertainty, and that it evaluates as required
throughout its domain, on a grid of points in it from just above `pt_min`
to 10⁴ GeV and up to just below `abs_eta_max`, every `phi`, both charges,
each requested object type — and the measurement declared for each
({{ref:measurement}}: structure, recount, and the pairs against the
development dataset).  Where correctionlib is not importable, the
self-check validates the structure of `corrections.json` only and says
that its evaluation was NOT VALIDATED HERE<!-- /if -->.
