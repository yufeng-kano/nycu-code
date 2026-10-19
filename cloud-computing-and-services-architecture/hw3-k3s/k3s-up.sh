#!/usr/bin/env bash
# Bring the two-node environment up: create the Incus VMs on first run, start them otherwise.
# Safe to run repeatedly; every step skips what already exists.
# shellcheck source=scripts/common.sh
source "$(dirname "$0")/scripts/common.sh" || exit 1

check_prereqs up
# Build first: a broken Dockerfile or source tree stops here, before any VM is touched.
build_images
ensure_network
open_firewall
ensure_profile

for i in "${!NODES[@]}"; do
  name="$(node_field "$i" 1)"
  if ! vm_exists "$name"; then
    create_vm "$i"
  fi
  if ! vm_running "$name"; then
    log "Starting $name"
    incus start "$name"
  else
    log "$name is already running"
  fi
done

for i in "${!NODES[@]}"; do
  wait_for_vm "$(node_field "$i" 1)" "$(node_field "$i" 3)"
done

install_k3s_server
for i in "${!NODES[@]}"; do
  [[ "$(node_field "$i" 2)" == worker ]] && install_k3s_agent "$i"
done
wait_for_nodes

# Application Pods only run on workers (the server is tainted), so only workers need the images.
for i in "${!NODES[@]}"; do
  [[ "$(node_field "$i" 2)" == worker ]] && import_images "$(node_field "$i" 1)"
done

deploy_app

log "Cluster and application are up:"
incus list hw3- --columns ns4tS
echo "    app:     http://$(node_field 1 3):$NODE_PORT/   (any node IP works)"
echo "    shell:   ./scripts/vm-shell.sh server|worker"
echo "    kubectl: ./scripts/kubectl.sh -n $APP_NS get pods"
