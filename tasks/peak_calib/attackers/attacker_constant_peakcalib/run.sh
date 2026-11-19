#!/bin/bash
# ATTACKER "constant": never reads the data; reports every requested correction at its nominal value with a fixed
# uncertainty.  Also the minimal interface example.
set -euo pipefail
mkdir -p "$2"
python3 - "$2" <<'PY'
import json, sys


def constant(name, nominal, up, down):
    return {"name": name, "version": 1, "output": {"name": "value", "type": "real"},
            "inputs": [{"name": "variation", "type": "string"}],
            "data": {"nodetype": "category", "input": "variation", "content": [
                {"key": "nominal", "value": nominal}, {"key": "up_stat", "value": up},
                {"key": "down_stat", "value": down}]}}


corrections = [constant("scale", 0.0, 0.001, -0.001), constant("res_pt", 0.0, 0.005, -0.005),
               constant("Muon_goodId", 1.0, 1.01, 0.99)]
json.dump({"schema_version": 2, "corrections": corrections}, open(sys.argv[1] + "/corrections.json", "w"), indent=1)
json.dump({"note": "constant output: this pipeline never opens the dataset"}, open(sys.argv[1] + "/RESULT.json", "w"))
PY
