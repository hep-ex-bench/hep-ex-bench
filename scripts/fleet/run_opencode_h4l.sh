#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$HOME/.opencode/bin:$HOME/.pixi/bin:$PATH"
export PATH="$REPO/.pixi/envs/default/bin:$PATH"
cd -P "$REPO/work/agents/$1" || exit 2
RECORD=("$REPO/.pixi/envs/default/bin/python" -I "$REPO/scripts/fleet/agent_run_record.py")
export XDG_DATA_HOME="$PWD/.xdg_data"
mkdir -p "$XDG_DATA_HOME"
BUDGET_S="${AGENT_TIMEOUT:-7200}"
if "${RECORD[@]}" start "$1" --unless-open --harness opencode --model "${2:-default}" --prompt PROMPT.md \
        ${SBH_AGENT_EFFORT:+--effort "$SBH_AGENT_EFFORT"} --budget "$BUDGET_S"; then
    trap 'st=$?; "${RECORD[@]}" end "$1" --exit "${rc:-$st}"' EXIT
fi
TRAILER="$("$REPO/scripts/fleet/agent_trailer.sh" "$BUDGET_S")" || exit 2
PROMPT="$(cat PROMPT.md)

$TRAILER"
"${RECORD[@]}" version "$1" --harness opencode || echo "run record: version not written"
VARIANT=()
if [ -n "${SBH_AGENT_EFFORT:-}" ]; then
    LISTING=""
    for _ in 1 2; do
        LISTING=$(timeout 120 "$REPO/scripts/fleet/dev_sandbox.sh" "$PWD" -- opencode models "${2%%/*}" --verbose 2>/dev/null) \
            && [ -n "$LISTING" ] && break
    done
    printf '%s\n' "$LISTING" | "${RECORD[@]}" variant-check "$1" --model "$2" --level "$SBH_AGENT_EFFORT" || exit 3
    VARIANT=(--variant "$SBH_AGENT_EFFORT")
fi
timeout "$BUDGET_S" "$REPO/scripts/fleet/dev_sandbox.sh" "$PWD" -- opencode run --auto -m "$2" "${VARIANT[@]}" --print-logs --log-level warn "$PROMPT" > ../$1.opencode.log 2>&1
rc=$?
echo "opencode exit=$rc" >> ../$1.opencode.log
