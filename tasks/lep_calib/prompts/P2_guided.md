Perform a high-energy-physics analysis of the dataset described in
DATASET.md.  The target of the analysis: the calibration of the leptons in
the data relative to the simulation, measured with Z→ℓℓ events — the muon
and electron momentum scale and resolutions, and the efficiency scale
factors of the identification selections and trigger legs the dataset's
`request.json` names.

Request envelope: the evaluation datasets may request other
identification selections and trigger legs than the development dataset —
any that EVAL_CONTRACT.md lists under request.json, each with its own
domain — so the analysis must handle every one of them.

Deliver a reproducible analysis directory that conforms to EVAL_CONTRACT.md:
a `run.sh <dataset_dir> <output_dir>` entrypoint that writes
`<output_dir>/RESULT.json` and `<output_dir>/corrections.json`.

Physics context: the dataset is dominated by Drell–Yan, with small
contributions from top-quark pairs and diboson production.  The detector
response and the lepton efficiencies of the data may differ from the
nominal simulation — measure by how much.

Tooling context: the field's standard stack applies — columnar,
vectorized analysis rather than per-event Python loops, and a proper
statistical treatment of the measurements and their uncertainties.

Analysis plan — follow it.  Take the methods of the CMS lepton performance
measurements: muons in CMS, JINST 13 (2018) P06015, arXiv:1804.04528;
electrons in CMS, JINST 16 (2021) P05014, arXiv:2012.06888.  Take methods
from the literature, not numbers: every correction is measured on this
dataset, the same way in the data and in the simulation.

1. Selection: Z→μμ and Z→ee candidates — opposite-sign same-flavour lepton
   pairs near the Z mass — with a tag-and-probe structure for the
   efficiencies: a tightly identified, isolated tag matched to a
   single-lepton trigger, and an unbiased probe.
2. Momentum scale and resolution: from the dilepton lineshapes, per
   flavour and in categories of the leptons' pT and η — the scale from the
   peak position in the data relative to the simulation, the extra
   resolution from the widths, with the simulation smeared to describe the
   data.  Use categories in which the transverse-momentum and the angular
   resolutions affect the lineshapes differently, so that `res_pt`,
   `res_eta` and `res_phi` are measured separately.
3. Efficiencies: tag-and-probe for every identification selection and
   trigger leg in `request.json`, in the data and in the simulation, in
   bins of the probe's pT and η.
4. Parametrization: per flavour and per correction, fit a formula in the
   lepton's pT and η to the binned measurements and report the formula's
   parameters with their covariance (a fit in CMS Combine,
   Comput. Softw. Big Sci. 8 (2024) 19, arXiv:2404.06614, is one option, not
   a requirement); keep the binned result where no formula describes the
   measurements.  Decide from the data, for each correction, whether the
   muons and the electrons need separate functions.
5. Uncertainties: statistical, including the finite simulation sample,
   and systematic — the background model and the fit model of the
   lineshapes, the tag selection, the mass window.
6. Validation: closure — the simulation corrected with your scale,
   resolutions and efficiency scale factors against the data, in the
   dilepton lineshapes and in the tag-and-probe pass/fail yields; iterate
   the measurement if any of them looks wrong.
7. Outputs: everything the contract specifies; self-check and fix until it
   passes.

Track your work for reproducibility: develop the submission as a git
repository with meaningful commits as the analysis evolves, and keep a
log of the decisions taken (selections, categories, fit models,
parametrizations) alongside the code — the analysis must be auditable end
to end.

Practical notes: read branches selectively with uproot and keep the
event processing columnar/vectorized (awkward) — a per-event Python loop
over these datasets will not finish inside the run budgets of
EVAL_CONTRACT.md.
