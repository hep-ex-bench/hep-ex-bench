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
nominal simulation.

Tooling context: the field's standard stack applies — columnar,
vectorized analysis rather than per-event Python loops, and a proper
statistical treatment of the measurements and their uncertainties.
