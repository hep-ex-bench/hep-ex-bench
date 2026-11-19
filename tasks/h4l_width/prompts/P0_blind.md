Perform a high-energy-physics analysis of the dataset described in
DATASET.md.  The target of the analysis: the total width of the Higgs
boson, as the ratio ξ = Γ_H/Γ_SM with its 68% confidence set (one or
several intervals), and the on-shell signal strength μ_on with its 68%
confidence interval, in the H→ZZ→4ℓ channel.  The on-shell signal
strength of a dataset is not given.

Deliver a reproducible analysis directory that conforms to EVAL_CONTRACT.md:
a `run.sh <dataset_dir> <output_dir>` entrypoint that writes ξ and μ_on
to `<output_dir>/RESULT.json`, and the statistical model behind them to
`MODEL.json` and `EVENTS.json`.  The evaluator measures ξ and μ_on by
refitting the model you declare.
