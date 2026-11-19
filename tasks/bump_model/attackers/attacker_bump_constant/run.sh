#!/bin/bash
# ATTACKER "constant": never reads the data; always reports the public dataset's answer, with a fixed model.
set -euo pipefail
mkdir -p "$2"
python3 - "$2" <<'PY'
import json, math, sys
out = sys.argv[1]
json.dump({"pois": [{"name": "mH", "value": 125.0, "stat": 0.4, "syst": 0.0, "total": 0.4},
                    {"name": "mu", "value": 1.0, "stat": 0.3, "syst": 0.0, "total": 0.3}],
           "significance_obs": 4.0, "discovery": False,
           "nuisances": {"ana_bkg_norm": {"value": 1.0, "unc": 0.03}},
           "note": "constant output: this pipeline never opens the dataset"},
          open(out + "/RESULT.json", "w"), indent=1)
# a fixed model on 1 GeV bins of 102-158 GeV: a Gaussian bump at 125 GeV on a falling exponential, its expectation
# as the observations; no EVENTS.json, since no event was read
edges = [102.0 + i for i in range(57)]
gauss = lambda x: 0.5 * (1.0 + math.erf((x - 125.0) / (1.5 * math.sqrt(2.0))))
sig = [50.0 * (gauss(b) - gauss(a)) for a, b in zip(edges, edges[1:])]
bkg = [2400.0 / (1.0 - math.exp(-3.2)) * (math.exp(-(a - 90.0) / 25.0) - math.exp(-(b - 90.0) / 25.0))
       for a, b in zip(edges, edges[1:])]
json.dump({"channels": [{"name": "mass", "samples": [
             {"name": "signal", "data": sig, "modifiers": [{"name": "mu", "type": "normfactor", "data": None}]},
             {"name": "background", "data": bkg,
              "modifiers": [{"name": "ana_bkg_norm", "type": "normfactor", "data": None}]}]}],
           "observations": [{"name": "mass", "data": [round(s + b) for s, b in zip(sig, bkg)]}],
           "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
           "version": "1.0.0",
           "_meta": {"binning": {"mass": {"observable": "pair mass [GeV]", "edges": edges,
                                          "data_derived": False}}}},
          open(out + "/MODEL.json", "w"), indent=1)
PY
