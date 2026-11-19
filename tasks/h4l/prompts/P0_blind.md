Perform a high-energy-physics analysis of the dataset described in
DATASET.md.  The target of the analysis: the mass and the signal
strength of the Higgs boson in the H→ZZ*→4ℓ channel.

Deliver a reproducible analysis directory that conforms to EVAL_CONTRACT.md:
a `run.sh <dataset_dir> <output_dir>` entrypoint that writes
`<output_dir>/RESULT.json`.  If you report corrections such as lepton scale
factors, give each as one constant — its average over the population
EVAL_CONTRACT.md §4 defines.
