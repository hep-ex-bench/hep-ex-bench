#!/bin/bash
# ATTACKER "width_constant": never reads the data; always the Standard-Model answer xi = 1, mu_on = 1, with a fixed model (also the participant bundle's minimal-interface example)
set -euo pipefail
mkdir -p "$2"
python3 - "$2" <<'PY'
import json, sys
out = sys.argv[1]
json.dump({"pois": [{"name": "xi", "value": 1.0, "err_minus": 0.5, "err_plus": 0.5},
                    {"name": "mu_on", "value": 1.0, "err_minus": 0.2, "err_plus": 0.2}],
           "nuisances": {},
           "note": "constant output: this pipeline never opens the dataset"}, open(out + "/RESULT.json", "w"), indent=1)
# a fixed model on three bins of 100-1000 GeV: an on-shell peak scaling with mu_on, an off-shell signal scaling with
# mu_on xi and a background, its expectation as the observations; no EVENTS.json, since no event was read
nf = [{"name": "mu_on", "type": "normfactor", "data": None}, {"name": "xi", "type": "normfactor", "data": None}]
json.dump({"channels": [{"name": "m4l", "samples": [
             {"name": "H_onshell", "data": [40.0, 0.0, 0.0], "modifiers": nf[:1]},
             {"name": "S", "data": [0.0, 5.0, 10.0], "modifiers": nf},
             {"name": "background", "data": [300.0, 1500.0, 500.0], "modifiers": []}]}],
           "observations": [{"name": "m4l", "data": [340, 1505, 510]}],
           "measurements": [{"name": "meas", "config": {"poi": "xi", "parameters": []}}],
           "version": "1.0.0",
           "_meta": {"binning": {"m4l": {"observable": "m4l [GeV]", "edges": [100.0, 182.0, 400.0, 1000.0],
                                         "data_derived": False}},
                     "poi_scaling": {"m4l": {"S": {"factor": 1.0, "powers": {"mu_on": 1, "xi": 1}}}}}},
          open(out + "/MODEL.json", "w"), indent=1)
PY
