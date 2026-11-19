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
may differ from the nominal simulation.  The published measurement in this
channel under the same (2016) conditions is CMS, JHEP 11 (2017) 047,
arXiv:1706.09936.

Tooling context: the field's standard stack applies — columnar,
vectorized analysis rather than per-event Python loops, and a proper
statistical model for the inference.
