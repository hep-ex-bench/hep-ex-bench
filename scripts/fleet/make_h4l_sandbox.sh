#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
BUNDLE=${SBH_BUNDLE:?set SBH_BUNDLE to the packed participant bundle (work/dist/...)}
DEST="work/agents/$1"
[ -d "$BUNDLE" ] || { echo "pack first: python3 -m sbh pack --suite <suite> --out work/dist" \
    "(or --task <task> --public-dataset <dir>)"; exit 1; }
[ -e "$DEST" ] && { echo "$DEST exists"; exit 1; }
mkdir -p "$(dirname "$DEST")"
cp -al "$BUNDLE" "$DEST"
find "$DEST" ! -type d ! -name '*.root' -exec bash -c \
    'for f; do t="$f.unlink.$$"; cp -p "$f" "$t" && mv -f "$t" "$f"; done' _ {} +
find "$DEST" -name '*.root' -type f -exec chmod a-w {} +
if [ -n "${2:-}" ]; then
    rm -f "$DEST/PROMPT.md"
    cp "$2" "$DEST/PROMPT.md"
fi
base=$(.pixi/envs/default/bin/python - "$DEST" <<'PY'
import importlib, sys
from pathlib import Path
dest = Path(sys.argv[1])
sys.path.insert(0, str(dest))
task = next(p.name for p in sorted((dest / "tasks").iterdir()) if (p / "__init__.py").exists())
print(getattr(importlib.import_module(f"tasks.{task}").TASK, "base_env", None) or "agentbase")
PY
) || { echo "cannot read the bundle's task stub under $DEST/tasks"; exit 1; }
[ -d ".pixi/envs/$base" ] || { echo "the task's base environment .pixi/envs/$base is not installed (pixi install -e $base)"; exit 1; }
mkdir -p "$DEST/.pixi/envs"
ln -s "$(cd ".pixi/envs/$base" && pwd -P)" "$DEST/.pixi/envs/$base"
echo "sandbox at $DEST (base environment $base)"
