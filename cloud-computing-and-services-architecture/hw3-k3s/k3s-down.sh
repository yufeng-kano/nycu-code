#!/usr/bin/env bash
# Stop the environment: the application, then K3s, then the VMs. Disks are kept, so the
# PostgreSQL PVC survives and ./k3s-up.sh resumes where it left off.
#   ./k3s-down.sh            shut down both VMs, keep everything
#   ./k3s-down.sh --delete   also delete both VMs, the hw3-k3s profile, the bridge network and
#                            the host firewall rules ./k3s-up.sh added (asks for sudo)
# shellcheck source=scripts/common.sh
source "$(dirname "$0")/scripts/common.sh" || exit 1

delete=false
case "${1:-}" in
  "") ;;
  --delete) delete=true ;;
  -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
  *) die "unknown option: $1 (use --delete or --help)" ;;
esac

check_prereqs

# Stop in order: application (so PostgreSQL flushes to the PVC), then K3s, then the VMs.
# Skipped when the server is already off.
if vm_running "$(server_name)"; then
  stop_app
  stop_k3s
fi

# Worker first, then the control plane, so the server is the last thing to go away.
for ((i = ${#NODES[@]} - 1; i >= 0; i--)); do
  name="$(node_field "$i" 1)"
  vm_exists "$name" || continue
  stop_vm "$name"
  if $delete; then
    log "Deleting $name and its disk"
    incus delete "$name"
  fi
done

if $delete; then
  incus profile show "$PROFILE" &>/dev/null && { log "Deleting profile $PROFILE"; incus profile delete "$PROFILE" >/dev/null; }
  incus network show "$NET_NAME" &>/dev/null && { log "Deleting network $NET_NAME"; incus network delete "$NET_NAME" >/dev/null; }
  close_firewall
fi

# The downloaded Ubuntu image stays in Incus's image cache, so the next ./k3s-up.sh is fast.
log "Done."
incus list hw3- --columns ns4tS
