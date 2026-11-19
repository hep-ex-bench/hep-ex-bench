#!/bin/bash
set -euo pipefail
n() { if (( $1 == 1 )); then echo "1 $2"; else echo "$1 $2s"; fi; }
hm() {
    local t=$1 h m
    [[ "$t" =~ ^[0-9]+$ ]] || { echo "agent_trailer.sh: a time must be whole seconds, got '$t'" >&2; exit 2; }
    t=$((10#$t))
    (( t > 0 )) || [ -n "${2:-}" ] ||
        { echo "agent_trailer.sh: a budget must be positive (timeout 0 disables the limit)" >&2; exit 2; }
    h=$(( t / 3600 )); m=$(( t % 3600 / 60 ))
    if   (( h && m )); then echo "$(n "$h" hour) $(n "$m" minute)"
    elif (( h ));      then n "$h" hour
    elif (( m ));      then n "$m" minute
    else                    n "$t" second
    fi
}
b=$(hm "${1:?usage: agent_trailer.sh <budget_s>}")
sed "s/@SESSION_BUDGET@/$b/g" "$(dirname "$0")/agent_trailer.txt"
