Measure the Higgs width ratio ξ = Γ_H/Γ_SM and the on-shell signal strength
μ_on from the four-lepton mass spectrum in the pseudo-dataset described in
DATASET.md (a toy stand-in for the CMS off-shell H → ZZ → 4ℓ width
measurement).  ξ is at most 100 and μ_on at most 5.

Deliver a reproducible analysis directory that conforms to EVAL_CONTRACT.md:
`run.sh <dataset_dir> <output_dir>` writes the measured ξ and μ_on with
their 68 % intervals to `RESULT.json`, and the statistical model behind
them to `MODEL.json` and `EVENTS.json`.  The evaluator measures ξ and μ_on
from the model you declare.

Develop against the public seed only.  Your frozen pipeline will then run
unchanged on sealed datasets with the same layout but different ξ, μ_on and
background normalisation.  What is scored is whether your declared model
tracks the injected ξ and μ_on across datasets, with honest intervals.
