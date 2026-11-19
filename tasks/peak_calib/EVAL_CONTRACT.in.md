# EVAL_CONTRACT v2.0-toy — submission contract for `peak_calib`

{{core:intro}}It takes `RESULT.json` ({{ref:result}}) and the corrections each dataset's
`request.json` asks for: the corrections as functions (`corrections.json`;
{{ref:functions}}), the request ({{ref:request}}) and, optionally, the
measurement behind each correction (`MEASUREMENT.json`, `MEASUREMENT.npz`;
{{ref:measurement}}).  PROMPT.md says what to measure; DATASET.md describes
the data.

## {{sec:submission}}. Submission form

{{core:submission}}  The withheld datasets differ from the development dataset
  in their detector response (DATASET.md).
{{core:outputs}}
{{core:hygiene}}
{{core:runtime}}{{core:budgets}}{{core:selfcheck_memory}}{{core:selfcheck_memory_end}}
## {{sec:result}}. RESULT.json

A JSON object (GATE).  Your answers are in `corrections.json`
({{ref:functions}}); `RESULT.json` carries the parameter covariances of
corrections ({{ref:functions}}):

    {
      "covariances": { "<correction name>": [[f, ...], ...] },
      "joint_covariances": [ ... ]
    }

Both fields are optional; extra keys are allowed and ignored.

## {{sec:functions}}. corrections.json — corrections as functions

{{kind:functions}}
## {{sec:request}}. request.json — requested corrections

Every dataset's `request.json` lists the corrections requested on it:

    {"request_version": 1,
     "corrections": {"<name>": {"kind": "<kind>", "domain": {"<flavour>": {"pt_min": f, "abs_eta_max": f}, ...},
                                ...}, ...}}

Every entry carries `kind` and `domain`: the object types (`flavour`
values of {{ref:functions}}) the correction is requested for, each with the region where
it is defined, objects with pT > `pt_min` and |η| < `abs_eta_max`.  An `id`
entry also carries `selection`, the name of the boolean branch whose
efficiency it corrects.  The request of this task, the same on every
dataset:

    {"request_version": 1, "corrections": {
      "Muon_goodId": {"kind": "id", "domain": {"muon": {"pt_min": 15.0, "abs_eta_max": 2.4}},
                      "selection": "Muon_goodId"},
      "res_pt": {"kind": "resolution", "domain": {"muon": {"pt_min": 20.0, "abs_eta_max": 2.4}}},
      "scale": {"kind": "scale", "domain": {"muon": {"pt_min": 20.0, "abs_eta_max": 2.4}}}}}

All corrections relate the data to the simulation for the same true
object, as functions of the object's kinematics in the simulation:

- `scale` — momentum scale of the data relative to the simulation: the
  data reconstruct, on average, (1 + value) times the transverse momentum
  the simulation reconstructs for the same true object.  0 = nominal.
- `res_pt` — extra relative transverse-momentum resolution in the data:
  smearing each simulated object's pT with a Gaussian of relative width
  `value` reproduces the data (it adds to the simulated resolution in
  quadrature).  0 = nominal.

  A resolution's value stands for the extra variance value × |value|; a
  negative value stands for a negative extra variance (data narrower than
  the simulation).  Its nominal value and both ends of its uncertainty band
  — `"up"` and `"down"`, or the value ± its propagated uncertainty — are
  each mapped through value × |value|, and the extra variance's uncertainty
  is half the distance between the two mapped ends.
- `Muon_goodId` (`"kind": "id"`) — data/simulation ratio of the efficiency
  with which a reconstructed muon (an entry of the `Muon` collection)
  passes `selection`.  1 = nominal.

## {{sec:measurement}}. MEASUREMENT.json — the measurement behind a reported correction

{{kind:measurement}}
## {{sec:selfcheck}}. Self-check

`sbh check` runs your submission on the development dataset as the
evaluation does and checks every output file it takes against this
contract: `RESULT.json` and `corrections.json` — every reported correction
against correctionlib's schema, its inputs and its uncertainty, and that it
evaluates as required throughout its domain, on a grid of points in it from
just above `pt_min` to 10⁴ GeV and up to just below `abs_eta_max`, every
`phi`, both charges — and the measurement declared for each
({{ref:measurement}}: structure, recount, and the pairs against the
development dataset).  Where correctionlib is not importable, the
self-check validates the structure of `corrections.json` only and says
that its evaluation was NOT VALIDATED HERE.
