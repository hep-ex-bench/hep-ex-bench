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
data may differ from the nominal simulation.  The published measurements
of the width in this channel are CMS, Nature Phys. 18 (2022) 1329,
arXiv:2202.06923, and CMS, Phys. Rev. D 111 (2025) 092014,
arXiv:2409.13663; the four-lepton analysis under the same (2016)
conditions is CMS, JHEP 11 (2017) 047, arXiv:1706.09936.

Tooling context: the field's standard stack applies — columnar,
vectorized analysis rather than per-event Python loops, and a proper
statistical model for the inference.
