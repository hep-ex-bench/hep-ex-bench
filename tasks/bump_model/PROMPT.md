Search for and measure a narrow resonance in the pseudo-dataset described in
DATASET.md (a toy).

Deliver a reproducible analysis directory that conforms to EVAL_CONTRACT.md:
a `run.sh <dataset_dir> <output_dir>` entrypoint that writes `RESULT.json`
with the resonance mass `mH` and the signal strength `mu` (= sigma/sigma_SM
at that mass), the statistical model behind them (`MODEL.json`, the mass
grid `MODEL_GRID/` and `EVENTS.json`), the observed local significance and
a discovery flag.  The evaluator measures `mH` and `mu` from the model you
declare.

Physics envelope: a resonance, if present, has mH in [115, 135] GeV and
mu in [0, 3].

Develop against the public dataset only.  Your frozen pipeline will then be
executed unchanged on sealed datasets with the same layout but different
injected mass, signal strength, background normalisation and detector
response.  Do not hard-code any number from the public dataset: what is
scored is whether your pipeline tracks the injected truth across datasets,
with honest uncertainties.

Verify your submission with:

    python3 -m sbh check --task bump_model --submission <your_dir> --dataset datasets/public/seed_000 --determinism
