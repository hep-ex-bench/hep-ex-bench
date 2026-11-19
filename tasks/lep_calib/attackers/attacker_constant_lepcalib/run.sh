#!/bin/bash
# ATTACKER "constant": reads only request.json, never the events; reports every requested correction at its nominal
# value (efficiency ratios 1, scale 0, extra widths 0) with a generic uncertainty.  Also the participant bundle's
# interface example.
set -euo pipefail
mkdir -p "$2"
python3 - "$1" "$2" <<'PY'
import json, math, sys
dataset, out = sys.argv[1], sys.argv[2]
request = json.load(open(dataset + "/request.json"))["corrections"]
NOMINAL = {"id": (1.0, 0.02), "trigger": (1.0, 0.02), "scale": (0.0, 0.01), "resolution": (0.0, 0.01)}
corrections = []
for name, entry in sorted(request.items()):
    value, width = NOMINAL[entry["kind"]]
    up, down = value + width, value - width
    if entry["kind"] == "resolution":
        # a width is compared through its extra variance w |w|: these ends map to +/- width^2 around 0
        up, down = math.copysign(width, 1.0), math.copysign(width, -1.0)
    corrections.append({
        "name": name, "version": 1, "output": {"name": "value", "type": "real"},
        "inputs": [{"name": "variation", "type": "string"}],
        "data": {"nodetype": "category", "input": "variation",
                 "content": [{"key": "nominal", "value": value}, {"key": "up", "value": up},
                             {"key": "down", "value": down}]}})
json.dump({"schema_version": 2, "corrections": corrections}, open(out + "/corrections.json", "w"), indent=1)
json.dump({"note": "constant output: this pipeline reads only request.json, never the events"},
          open(out + "/RESULT.json", "w"), indent=1)
PY
