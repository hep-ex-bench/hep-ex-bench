#!/bin/bash
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$HOME/.local/bin:$HOME/.pixi/bin:$PATH"
export PATH="$REPO/.pixi/envs/default/bin:$PATH"
case "$SBH_CLAUDE_TOKEN_FILE" in /*|"") ;; *) SBH_CLAUDE_TOKEN_FILE="$(pwd -P)/$SBH_CLAUDE_TOKEN_FILE";; esac
cd -P "$REPO/work/agents/$1" || exit 2
RECORD=("$REPO/.pixi/envs/default/bin/python" -I "$REPO/scripts/fleet/agent_run_record.py")
export CLAUDE_CONFIG_DIR="$PWD/.claude_cfg"
if [ -L "$CLAUDE_CONFIG_DIR" ] || { [ -e "$CLAUDE_CONFIG_DIR" ] && [ ! -d "$CLAUDE_CONFIG_DIR" ]; }; then
    "${RECORD[@]}" refuse "$1" --reason "$CLAUDE_CONFIG_DIR is not a plain directory"
    exit 3
fi
mkdir -p "$CLAUDE_CONFIG_DIR"
for f in "$CLAUDE_CONFIG_DIR/.claude.json" PROMPT.md; do
    if [ -L "$f" ] || { [ -e "$f" ] && [ ! -f "$f" ]; }; then
        "${RECORD[@]}" refuse "$1" --reason "$f is not a regular file"
        exit 3
    fi
done
[ -f "$CLAUDE_CONFIG_DIR/.claude.json" ] || echo '{"hasCompletedOnboarding":true}' > "$CLAUDE_CONFIG_DIR/.claude.json"
BUDGET_S="${SBH_RESUME_LEFT:-${AGENT_TIMEOUT:-7200}}"
unset ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY
if [ "${SBH_CLAUDE_AUTH:-}" = token ] || [ -e "$SBH_CLAUDE_TOKEN_FILE" ]; then AUTH=token; else AUTH=login-copy; fi
if "${RECORD[@]}" start "$1" --unless-open --harness claude --model "${2:-default}" --prompt PROMPT.md \
        ${SBH_AGENT_EFFORT:+--effort "$SBH_AGENT_EFFORT"} --budget "$BUDGET_S"; then
    trap 'st=$?; "${RECORD[@]}" end "$1" --exit "${rc:-$st}"' EXIT
fi
"${RECORD[@]}" version "$1" --harness claude --auth "$AUTH" || echo "run record: version not written"
if [ "$AUTH" = token ]; then
    if [ -e "$CLAUDE_CONFIG_DIR/.credentials.json" ]; then
        "${RECORD[@]}" refuse "$1" \
            --reason "$CLAUDE_CONFIG_DIR/.credentials.json holds a copied login: move it out of the sandbox"
        exit 3
    fi
    CLAUDE_CODE_OAUTH_TOKEN="$("${RECORD[@]}" token-check "$1" --file "$SBH_CLAUDE_TOKEN_FILE" --emit)" || exit 3
    export CLAUDE_CODE_OAUTH_TOKEN
else
    "${RECORD[@]}" login-check "$1" --budget "$BUDGET_S" --config "$HOME/.claude" || exit 3
    cp -T --remove-destination "$HOME/.claude/.credentials.json" "$CLAUDE_CONFIG_DIR/.credentials.json" || {
        "${RECORD[@]}" refuse "$1" --reason "the operator's login could not be copied into $CLAUDE_CONFIG_DIR"
        exit 3
    }
    "${RECORD[@]}" login-check "$1" --budget "$BUDGET_S" || exit 3
fi
RESUME_ARGS=()
LOG="../$1.claude.log"
if [ ${#RESUME_ARGS[@]} -eq 0 ]; then
    TRAILER="$("$REPO/scripts/fleet/agent_trailer.sh" "$BUDGET_S")" || exit 2
    PROMPT="$(cat PROMPT.md)

$TRAILER"
fi
EFFORT_ARGS=()
[ -n "${SBH_AGENT_EFFORT:-}" ] && EFFORT_ARGS=(--effort "$SBH_AGENT_EFFORT")
if [ ${#RESUME_ARGS[@]} -eq 0 ]; then
    : > "$LOG"
fi
timeout "$BUDGET_S" "$REPO/scripts/fleet/dev_sandbox.sh" "$PWD" -- claude -p "${RESUME_ARGS[@]}" --model "$2" "${EFFORT_ARGS[@]}" --dangerously-skip-permissions --verbose --output-format stream-json "$PROMPT" >> "$LOG" 2>&1
rc=$?
[ -s "$LOG" ] && [ -n "$(tail -c1 "$LOG")" ] && echo >> "$LOG"
echo "claude exit=$rc" >> "$LOG"
