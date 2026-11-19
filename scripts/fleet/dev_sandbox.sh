#!/bin/bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SB="$(cd "$1" && pwd -P)"; SB_L="$(cd "$1" && pwd)"; shift
[ "${1:-}" = "--" ] && shift
SBH_DEV_HIDE="${SBH_DEV_HIDE:-}"
H="$(getent passwd "$(id -u)" | cut -d: -f6)"; H="$(cd "$H" && pwd -P)"
REPO_P="$(cd "$REPO" && pwd -P)"

args=()
for e in /*; do
    case "$e" in /net|/proc|/dev|/tmp) continue;; esac
    if [ -L "$e" ]; then args+=(--symlink "$(readlink "$e")" "$e")
    elif [ -d "$e" ]; then args+=(--ro-bind "$e" "$e")
    fi
done
args+=(--dir /net --dev /dev --proc /proc --tmpfs /tmp
       --unshare-pid --unshare-ipc --unshare-uts --share-net
       --die-with-parent --new-session)
[ -d /run/munge ] && args+=(--tmpfs /run/munge)
for root in $SBH_DEV_HIDE; do
    [ -d "$root" ] && args+=(--tmpfs "$(cd "$root" && pwd -P)")
done
args+=(--tmpfs "$H")
visible=("$SB" "$SB_L")
ro() { [ -e "$1" ] && args+=(--ro-bind "$1" "$1") && visible+=("$1"); return 0; }
rw() { [ -e "$1" ] && args+=(--bind "$1" "$1") && visible+=("$1"); return 0; }
ro "$REPO_P/.pixi/envs/default"; ro "$REPO_P/.pixi/envs/agentbase"
ENVS_P="$(cd "$REPO_P/.pixi/envs" 2>/dev/null && pwd -P)" || ENVS_P=""
for l in "$SB"/.pixi/envs/*; do
    [ -L "$l" ] && [ -n "$ENVS_P" ] || continue
    t="$(readlink -f "$l")"
    case "$t" in "$ENVS_P"/*) ro "$t";; esac
done
ro "$H/.local/bin"; ro "$H/.local/share/claude"; ro "$H/.opencode"; ro "$H/.pixi"
ro "$H/.codex/packages"; ro "$H/.config/opencode"; ro "$H/.gitconfig"
if [ -L "$SB/.cache" ] || { [ -e "$SB/.cache" ] && [ ! -d "$SB/.cache" ]; }; then
    echo "dev_sandbox.sh: $SB/.cache is not a plain directory: refusing to bind it" >&2
    exit 2
fi
mkdir -p "$SB/.cache"
args+=(--bind "$SB/.cache" "$H/.cache")
args+=(--bind "$SB" "$SB" --chdir "$SB")
for root in $SBH_DEV_HIDE; do
    [ -d "$root" ] && args+=(--remount-ro "$(cd "$root" && pwd -P)")
done
hidden=()
for root in $SBH_DEV_HIDE; do
    [ -d "$root" ] && hidden+=("$root" "$(cd "$root" && pwd -P)")
done
under() {
    local p=$1 r; shift
    for r in "$@"; do case "$p/" in "$r"/*) return 0;; esac; done
    return 1
}
drop=(SLURM_SUBMIT_DIR OMP_NUM_THREADS OMP_THREAD_LIMIT KRB5CCNAME)
for v in $(compgen -e); do
    case "$v" in
        CLAUDE_CODE_OAUTH_TOKEN|ANTHROPIC_AUTH_TOKEN) continue;;
        SBH_*|HF_*|HUGGING_FACE*|*TOKEN*|*SECRET*|*PASSWORD*) drop+=("$v"); continue;;
        _CONDOR_*|BATCH_*) drop+=("$v"); continue;;
        CLAUDE_CONFIG_DIR) under "${!v}" "$SB" "$SB_L" || drop+=("$v"); continue;;
        CLAUDECODE|CLAUDE_*|AI_AGENT|ANTHROPIC_*|SSH_*|TMUX|TMUX_*) drop+=("$v"); continue;;
    esac
    val="${!v}"
    case "$val" in /*) ;; *) continue;; esac
    case "$val" in *:*) continue;; esac
    under "$val" "${hidden[@]}" && ! under "$val" "${visible[@]}" && drop+=("$v")
done
unset_env=()
for v in "${drop[@]}"; do
    unset_env+=(-u "$v")
done
args+=(--setenv PIXI_CACHE_DIR "$SB/.cache/rattler")
args+=(--remount-ro /)
exec env "${unset_env[@]}" bwrap "${args[@]}" -- "$@"
