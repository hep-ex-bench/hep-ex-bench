Perform a high-energy-physics analysis of the dataset described in
DATASET.md.  The target of the analysis: the mass and the signal
strength of the Higgs boson in the H→ZZ*→4ℓ channel.

Deliver a reproducible analysis directory that conforms to EVAL_CONTRACT.md:
a `run.sh <dataset_dir> <output_dir>` entrypoint that writes
`<output_dir>/RESULT.json`.  If you report corrections such as lepton scale
factors, give each as one constant — its average over the population
EVAL_CONTRACT.md §4 defines.

Physics envelope: a Higgs-like signal, if present, has m_H ∈ [100, 200] GeV.
Signal simulation is provided at m_H = 125 GeV only.

Physics context: the dataset is dominated by Drell–Yan; the backgrounds to
a tight four-lepton selection are dominated by the irreducible ZZ→4ℓ
continuum, with reducible Z+X below it.  The data were recorded by a
lepton-multiplicity trigger menu with prescaled lower-multiplicity paths
(DATASET.md).  The detector response of the data
may differ from the nominal simulation — account for this.  The published
measurement in this channel under the same (2016) conditions is CMS,
JHEP 11 (2017) 047, arXiv:1706.09936.

Tooling context: the field's standard stack applies — columnar,
vectorized analysis rather than per-event Python loops, and a proper
statistical model for the inference.

Analysis plan — follow it.  Take the published measurement's strategy
where this dataset allows it; its lepton calibration methods are
described in detail in CMS, Phys. Rev. D 89 (2014) 092007,
arXiv:1312.5353.  Take methods from the literature, not numbers: every
correction is measured on this dataset.

1. Selection: build 4ℓ candidates per event — lepton identification,
   isolation and impact-parameter requirements, final-state-radiation
   photon recovery, then an opposite-sign same-flavour Z1 closest to m_Z
   and a Z2 — developed and validated on the simulation.
2. Corrections and systematic uncertainties: derive from the data, per
   lepton flavour and in bins of pT and |η|, every correction the
   published analysis applies to its simulation — the momentum scale and
   resolution from the dilepton lineshapes, the efficiency of each step of
   the lepton selection with tag-and-probe — and decide which of them
   differ from the nominal simulation here.  Apply them to the signal and
   background templates and yields, and carry their uncertainties, the
   luminosity uncertainty and the theory uncertainties of the predicted
   processes as nuisance parameters of the model.
3. Normalisation: expected yield of a simulated process =
   `effective_xsec_pb × lumi_fb × 1000 × n_selected/n_preselection`
   (entries from each process's `meta.json`; the luminosity from
   `lumi.json`).  Data events are recorded through the trigger menu's
   prescales (DATASET.md).  The signal changes with m_H through its cross section ×
   branching ratio (LHC Higgs Cross Section Working Group tables, relative
   to the provided 125 GeV point) and through its acceptance × efficiency,
   which the 125 GeV simulation alone does not give: derive it, e.g. from
   your own generator-level simulation (any generator may be installed in
   your environment).
4. Backgrounds: the irreducible ZZ from the simulation; the reducible Z+X
   from data control regions (lepton misidentification rates measured in
   data), with the simulation as a cross-check.
5. Fit: a binned likelihood of the four-lepton mass spectrum — the signal
   template from the 125 GeV simulation, its peak tracking m_H, with your
   corrections applied — with m_H and μ floating, built in a statistical
   framework that can express a floating m_H.
6. Uncertainties: profile-likelihood (or equivalent) intervals; report the
   statistical and systematic components and their quadrature total.
7. Validation: make sure the statistical model is appropriate — the
   corrected simulation against the data in control regions (Z→ℓℓ, Z+ℓ,
   the four-lepton sidebands), toy-based goodness of fit (not just the
   asymptotic saturated test) and the post-fit nuisance impacts on μ;
   iterate the model if any of them looks wrong.
8. Outputs: everything the contract specifies; self-check and fix until it
   passes.

Track your work for reproducibility: develop the submission as a git
repository with meaningful commits as the analysis evolves, and keep a
log of the decisions taken (selection choices, corrections, model
iterations) alongside the code — the analysis must be auditable end to
end.

Practical notes: read branches selectively with uproot and keep the
event processing columnar/vectorized (awkward) — a per-event Python loop
over these datasets will not finish inside the run budgets of
EVAL_CONTRACT.md.
