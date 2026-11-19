Perform a high-energy-physics analysis of the dataset described in
DATASET.md.  The target of the analysis: the total width of the Higgs
boson, as the ratio ξ = Γ_H/Γ_SM with its 68% confidence set (one or
several intervals), and the on-shell signal strength μ_on with its 68%
confidence interval, in the H→ZZ→4ℓ channel.

Deliver a reproducible analysis directory that conforms to EVAL_CONTRACT.md:
a `run.sh <dataset_dir> <output_dir>` entrypoint that writes ξ and μ_on
to `<output_dir>/RESULT.json`, and the statistical model behind them to
`MODEL.json` and `EVENTS.json`.  The evaluator measures ξ and μ_on by
refitting the model you declare.

Physics envelope: m_H = 125 GeV, ξ is at most 100 and the on-shell
signal strength μ_on at most 5; neither is given for a dataset.
Simulation is provided at ξ = 1 only.

Physics context: the dataset is dominated by Drell–Yan; the backgrounds to
a tight four-lepton selection are dominated by the irreducible ZZ→4ℓ
continuum — qq̄→ZZ, and gg→ZZ and electroweak ZZ production, which
interfere with Higgs production — with reducible Z+X at low mass.  The
data were recorded by a lepton-multiplicity trigger menu with prescaled
lower-multiplicity paths (DATASET.md).  The detector response of the
data may differ from the nominal simulation — account for this.  The
published measurements of the width in this channel are CMS, Nature
Phys. 18 (2022) 1329, arXiv:2202.06923, and CMS, Phys. Rev. D 111 (2025)
092014, arXiv:2409.13663; the four-lepton analysis under the same (2016)
conditions is CMS, JHEP 11 (2017) 047, arXiv:1706.09936.

Tooling context: the field's standard stack applies — columnar,
vectorized analysis rather than per-event Python loops, and a proper
statistical model for the inference.

Analysis plan — follow it.  Take the published off-shell strategy where
this dataset allows it: the on-shell signal strength measured at
m4ℓ ≈ 125 GeV, the off-shell rate above the ZZ threshold, and the width
from their joint fit.  Its lepton calibration methods are described in
detail in CMS, Phys. Rev. D 89 (2014) 092007, arXiv:1312.5353.  Take
methods from the literature, not numbers: every correction is measured
on this dataset.

1. Selection: build 4ℓ candidates per event — lepton identification,
   isolation and impact-parameter requirements, final-state-radiation
   photon recovery, then an opposite-sign same-flavour Z1 closest to m_Z
   and a Z2 — developed and validated on the simulation.  Define an
   on-shell region around 125 GeV and an off-shell region above the ZZ
   threshold.
2. Corrections and systematic uncertainties: derive from the data, per
   lepton flavour and in bins of pT and |η|, every correction the
   published analysis applies to its simulation — the momentum scale and
   resolution from the dilepton lineshapes, the efficiency of each step of
   the lepton selection with tag-and-probe — and decide which of them
   differ from the nominal simulation here.  Apply them to every template
   and yield, and carry their uncertainties, the luminosity uncertainty
   and the theory uncertainties of the predicted processes (including the
   gg→ZZ K-factor) as nuisance parameters of the model.
3. Normalisation: expected yield of a simulated process =
   `effective_xsec_pb × lumi_fb × 1000 × n_selected/n_preselection`
   (entries from each process's `meta.json`; the luminosity from
   `lumi.json`).  Data events are recorded through the trigger menu's
   prescales (DATASET.md).
4. Width templates.  Only ξ = 1 simulation exists; build the off-shell
   prediction at any ξ, bin by bin and separately for the gg and the
   electroweak family, from its samples above 130 GeV:
   T(ξ) = B + √ξ (SBI − S − B) + ξ S, with B = `*_B_high`,
   S = `*_S`, SBI = `*_SBI_xi1`.  The electroweak samples need two
   corrections for their on-shell Higgs pieces (DATASET.md), both
   identifiable in `GenPart`:
   - `EWZZ_S`'s on-shell ZH cross-feed does not depend on the width, so
     keep it fixed: T′(ξ) = T(ξ) − (ξ − 1) S_on, with S_on that part of
     `EWZZ_S`.
   - `EWZZ_SBI_xi1`'s on-shell H → bb̄ Higgs-strahlung: the generator
     scales the hVV couplings but not the b Yukawa, so for ξ ≥ 1 its
     rate goes as ξ^(−1/2), while T and T′ scale it with √ξ:
     T″(ξ) = T′(ξ) − (√ξ − ξ^(−1/2)) X_bb, with X_bb that part of
     `EWZZ_SBI_xi1`.  Below ξ = 1 that scaling would make the unscaled
     partial widths exceed the total width ξ Γ_SM; keep √ξ there
     (T″ = T′).
   Use T for gg and T″ for the electroweak family, and verify that every
   template bin stays non-negative over the ξ range you scan.  With a
   floating on-shell signal strength μ_on, the Higgs terms take μ_on·ξ
   and √(μ_on·ξ) in place of ξ and √ξ, and the cross-feed S_on scales
   with μ_on.
5. Discriminant: in the off-shell region, fit m4ℓ together with a
   kinematic discriminant that separates the gg→ZZ signal from the
   continuum (a matrix-element discriminant as in the published
   analysis, or a learned equivalent trained on the simulation).
6. Backgrounds: qq̄→ZZ from the simulation; the reducible Z+X from data
   control regions (lepton misidentification rates measured in data),
   with the simulation as a cross-check.  Below 130 GeV the on-shell
   Higgs comes from the `*_M125` samples and the gg and electroweak
   continuum from `*_B_low`.
7. Fit: a binned likelihood of the on-shell and off-shell regions
   together, with 0 ≤ ξ ≤ 100 and μ_on floating and the nuisance
   parameters profiled; this likelihood is the model you declare.
8. Uncertainties: the profile-likelihood scan of ξ and its 68% set —
   every piece where it has several — asymmetric where the likelihood
   is, and the 68% interval of μ_on; report the statistical and
   systematic components.
9. Validation: make sure the statistical model is appropriate — the
   corrected simulation against the data in control regions (Z→ℓℓ, Z+ℓ,
   the four-lepton sidebands), closure on the development dataset,
   injection tests on pseudo-data built from the templates at several ξ,
   toy-based goodness of fit and coverage of the declared model, and the
   post-fit nuisance impacts on ξ and μ_on; iterate the model if any of
   them looks wrong.
10. Outputs: everything the contract specifies; self-check and fix until
    it passes.

Track your work for reproducibility: develop the submission as a git
repository with meaningful commits as the analysis evolves, and keep a
log of the decisions taken (selection choices, corrections, model
iterations) alongside the code — the analysis must be auditable end to
end.

Practical notes: read branches selectively with uproot and keep the
event processing columnar/vectorized (awkward) — a per-event Python loop
over these datasets will not finish inside the run budgets of
EVAL_CONTRACT.md.
