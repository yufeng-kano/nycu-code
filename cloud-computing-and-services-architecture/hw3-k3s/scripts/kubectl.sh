#!/usr/bin/env bash
# Run kubectl against the cluster without installing anything on the host:
#   ./scripts/kubectl.sh get nodes -o wide
# shellcheck source=scripts/common.sh
source "$(dirname "$0")/common.sh" || exit 1

exec incus exec "$(server_name)" -- k3s kubectl "$@"
