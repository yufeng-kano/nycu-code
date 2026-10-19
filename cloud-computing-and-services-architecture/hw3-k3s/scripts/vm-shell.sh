#!/usr/bin/env bash
# Open a shell on a node, or run one command there:
#   ./scripts/vm-shell.sh server
#   ./scripts/vm-shell.sh worker 'uptime'
# shellcheck source=scripts/common.sh
source "$(dirname "$0")/common.sh" || exit 1

role="${1:-}"; shift || true
for i in "${!NODES[@]}"; do
  if [[ "$(node_field "$i" 2)" == "$role" ]]; then
    name="$(node_field "$i" 1)"
    if (($#)); then vm_exec "$name" "$*"; else exec incus shell "$name"; fi
    exit
  fi
done
die "usage: $0 server|worker [command]"
