# Prompt tiers

Three tiers, three deployment personas, as for `h4l`.  Each is a
strict superset of the previous, so an inter-tier score delta attributes
to the added briefing.

- `P0_blind.md` — **the pure model**: the analysis framing (the width
  ratio ξ = Γ_H/Γ_SM with its 68% set and the on-shell signal strength
  μ_on with its 68% interval, the latter not given for a dataset) and the
  deliverable pointer (the result and the statistical model the
  evaluator refits), nothing else.  Measures spontaneous physics
  judgment — does it find the off-shell method, read the sample
  definitions closely, suspect the detector?
- `../PROMPT.md` (**P1, the baseline — the informed model**): + the
  physics envelope (m_H = 125 GeV, ξ at most 100 and μ_on at most 5,
  neither given for a dataset, simulation at ξ = 1 only), the physics
  context (background composition and the gg→ZZ / electroweak
  interference with Higgs production; the recording by a
  lepton-multiplicity trigger menu with prescaled lower-multiplicity
  paths; the FACT that the detector response of the data may
  differ from the simulation, without saying which response or asking to
  act on it) and the published measurements (CMS, arXiv:2202.06923 and
  arXiv:2409.13663, and the 2016 four-lepton analysis, arXiv:1706.09936)
  as references.  Measures execution given awareness, and separates
  "didn't know" from "knew and didn't act".
- `P2_guided.md` — **the specified task**: + "account for the detector
  response" and an ordered plan anchored on the published off-shell
  analysis (selection → corrections and systematics derived from the data
  → normalisation through the menu's prescales → width templates T(ξ),
  with the two electroweak on-shell corrections T′ and T″ → kinematic
  discriminant → backgrounds → joint on-shell + off-shell fit with ξ and
  μ_on floating, the declared model → uncertainties (ξ's 68% set, μ_on's
  interval) → validation → outputs), taking methods from the literature
  and every number from the dataset.  Measures faithful execution of a
  specification.

The template formula and the electroweak corrections are stated only in
P2.  DATASET.md gives every sample's generator-level definition (process,
coupling content and how the width enters, cuts, normalisation, the
130 GeV split), from which a careful analyst can derive them and find the
two electroweak on-shell pieces in `GenPart`.  No tier names the truth
range or grid, or how the evaluation datasets are built, and none lists
which detector effects the suite varies.

Select a tier with `scripts/fleet/make_h4l_sandbox.sh <name> <prompt file>`,
which replaces the bundle's `PROMPT.md`.

Scoring is identical across tiers (`docs/METRICS.md`).
