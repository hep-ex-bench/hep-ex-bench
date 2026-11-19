Calibrate the muons of the pseudo-dataset described in DATASET.md (a toy
stand-in for a lepton calibration with Z → μμ events).

Deliver a reproducible analysis directory that conforms to EVAL_CONTRACT.md:
a `run.sh <dataset_dir> <output_dir>` entrypoint that writes
`<output_dir>/corrections.json` with every correction the dataset's
`request.json` asks for -- the muon momentum scale, the extra pT resolution
and the `Muon_goodId` efficiency scale factor of the data relative to the
simulation -- each as a correctionlib function with its uncertainty, and
`<output_dir>/RESULT.json`.

Develop against the public dataset only.  Your frozen pipeline will then be
executed unchanged on sealed datasets with the same layout and request but a
different scale, resolution and scale factor.  Do not hard-code any number
from the public dataset: what is scored is whether your corrections track the
injected truth across datasets, with honest uncertainties.

The base environment has Python with numpy, uproot and awkward.  Keep the run
under a minute per dataset.  Use only files under <dataset_dir>; there is no
network at run time.  Verify your submission with:

    python3 -m sbh check --task peak_calib --submission <your_dir> --dataset datasets/public/seed_000 --determinism
