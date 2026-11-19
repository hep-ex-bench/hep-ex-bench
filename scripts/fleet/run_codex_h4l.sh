#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$HOME/.local/bin:$HOME/.pixi/bin:$PATH"
export PATH="$REPO/.pixi/envs/default/bin:$PATH"
cd -P "$REPO/work/agents/$1" || exit 2
export CODEX_HOME="$PWD/.codex_home"
if [ ! -d "$CODEX_HOME" ]; then
    mkdir -p "$CODEX_HOME"
    cp "$HOME/.codex/auth.json" "$CODEX_HOME/" 2>/dev/null || true
    cp "$REPO/scripts/fleet/codex_config.toml" "$CODEX_HOME/config.toml"
fi
BUDGET_S="${SBH_RESUME_LEFT:-${AGENT_TIMEOUT:-7200}}"
RECORD=("$REPO/.pixi/envs/default/bin/python" -I "$REPO/scripts/fleet/agent_run_record.py")
if "${RECORD[@]}" start "$1" --unless-open --harness codex --model "${2:-default}" --prompt PROMPT.md \
        ${SBH_AGENT_EFFORT:+--effort "$SBH_AGENT_EFFORT"} --budget "$BUDGET_S"; then
    trap 'st=$?; "${RECORD[@]}" end "$1" --exit "${rc:-$st}"' EXIT
fi
RESUME_ARGS=()
LOG="../$1.codex.log"
if [ ${#RESUME_ARGS[@]} -eq 0 ]; then
    TRAILER="$("$REPO/scripts/fleet/agent_trailer.sh" "$BUDGET_S")" || exit 2
    PROMPT="$(cat PROMPT.md)

$TRAILER"
fi
"${RECORD[@]}" version "$1" --harness codex || echo "run record: version not written"
MODEL_ARGS=()
[ -n "${2:-}" ] && MODEL_ARGS=(-m "$2")
[ -n "${SBH_AGENT_EFFORT:-}" ] && MODEL_ARGS+=(-c "model_reasoning_effort=\"$SBH_AGENT_EFFORT\"")
if [ ${#RESUME_ARGS[@]} -eq 0 ]; then
    : > "$LOG"
fi
timeout "$BUDGET_S" "$REPO/scripts/fleet/dev_sandbox.sh" "$PWD" -- codex exec "${RESUME_ARGS[@]}" \
    --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox "${MODEL_ARGS[@]}" "$PROMPT" >> "$LOG" 2>&1
rc=$?
echo "codex exit=$rc" >> "$LOG"
