# Prompt tiers

Three tiers, three deployment personas, numbered P0 to P2 (lep_calib has
no blind tier: its baseline is P0).  Each is a strict superset of the one
before, so an inter-tier score delta attributes to the added briefing.  The
prompt files keep their names: `P2_guided.md` is P1, `P3_syst.md` is P2.

- `../PROMPT.md` (**P0, the baseline — the informed model**): the
  analysis target (the lepton calibration the dataset's `request.json`
  asks for), the request envelope (any selection or trigger leg the
  contract lists may be requested, as h4l's physics envelope bounds its
  mass), the deliverable pointer, the physics context (the sample
  composition; the FACT that the detector response and the lepton
  efficiencies of the data may differ from the simulation, without saying
  which or by how much) and the tooling context.  Measures execution given
  awareness of the task.
- `P2_guided.md` (**P1, the specified task**): what a physicist who sat down
  for an hour and wrote exactly what they want would hand over — "measure
  by how much" and an ordered plan anchored on the CMS lepton performance
  measurements (Z→ℓℓ selection with tag-and-probe → scale and resolutions
  from the dilepton lineshapes in categories that separate the
  transverse-momentum and the angular resolutions → tag-and-probe
  efficiencies in bins of the probe's pT and η → a formula in pT and η per
  correction with its parameter covariance, CMS Combine optional →
  uncertainties → closure → outputs), taking methods from the literature
  and every number from the dataset.  Measures faithful execution of a
  specification.
- `P3_syst.md` (**P2, the specified task with its systematics**): P1 with its
  uncertainty step spelt out — every source evaluated for every
  correction and reported as its own source: the data's and the
  simulation's statistics; for the efficiencies the background (the mass
  window or the background shape varied) and the tag selection (a tighter
  tag); for the scale and the resolutions the lineshape model (the fit
  window or the signal shape varied).  It names the sources and how to
  evaluate them, never a magnitude.  Measures whether
  an analysis told which uncertainties to evaluate states them honestly.

No tier names the reference analysis or the injected effects'
functional forms, or says which corrections differ from nominal or which
to report, or how large an uncertainty should be: P1's and P2's plans
measure every requested correction.

Select a tier with `scripts/fleet/make_h4l_sandbox.sh <name> <prompt file>`,
which replaces the bundle's `PROMPT.md`.

Scoring is identical across tiers (`docs/METRICS.md`).
