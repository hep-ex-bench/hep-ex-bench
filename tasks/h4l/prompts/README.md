# Prompt tiers

Three tiers, three deployment personas.  Each is a strict superset of the
previous, so an inter-tier score delta attributes to the added briefing.

- `P0_blind.md` — **the pure model**: the analysis framing and the
  deliverable pointer (with the form a reported correction takes, one
  constant averaged over the population the contract's §4 defines, which
  the scoring reads), nothing else.  Measures spontaneous physics
  judgment — does it explore, suspect the detector, behave like a
  physicist unprompted?
- `../PROMPT.md` (**P1, the baseline — the informed model**): + the
  physics envelope (priors: mass range, signal MC at 125 only), the physics
  context (background composition; the FACT that the detector response of
  the data may differ from the simulation, without saying which response
  or asking to act on it) and the published measurement in the channel
  (CMS, arXiv:1706.09936) as a reference.  The briefing a careful explorer
  would have produced for itself: measures execution given awareness, and
  separates "didn't know" from "knew and didn't act".
- `P2_guided.md` — **the specified task**: what a physicist who sat down
  for an hour and wrote exactly what they want would hand over — "account
  for the detector response" and an ordered plan anchored on the published
  analysis (selection → corrections and systematics derived from the data →
  normalisation with the m_H dependence of the acceptance → data-driven
  Z+X → fit → uncertainties → validation → outputs), taking methods from
  the literature and every number from the dataset.  Measures faithful
  execution of a specification.

No tier names the reference analysis, and none lists which detector
effects the suite varies: P2 asks for the standard set of corrections and
for deciding which differ.

Select a tier with `scripts/fleet/make_h4l_sandbox.sh <name> <prompt file>`,
which replaces the bundle's `PROMPT.md`.

Scoring is identical across tiers (`docs/METRICS.md`).
