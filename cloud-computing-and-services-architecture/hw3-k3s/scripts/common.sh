# shellcheck shell=bash
# Shared settings and helpers for k3s-up.sh / k3s-down.sh.
# Sourced, not executed. The VMs are Incus virtual machines (KVM underneath); members of the
# `incus-admin` group can manage them without sudo.

set -euo pipefail

# ---- Cluster layout ---------------------------------------------------------
# name        role    IP
NODES=(
  "hw3-server server 192.168.56.10"
  "hw3-worker worker 192.168.56.11"
)

# Incus-managed bridge with NAT to the outside. It is also the host interface name, and Linux
# caps those at 15 characters.
NET_NAME="nycu-cc-hw3-br"
NET_CIDR="192.168.56.1/24"
PROFILE="hw3-k3s"
IMAGE="images:ubuntu/24.04"

# Pinned so every run (and the grader's) installs the same Kubernetes version.
K3S_VERSION="${K3S_VERSION:-v1.36.5+k3s1}"

# Application images: built on the host from src/<tier>, then imported into the worker's containerd.
# The manifests use imagePullPolicy: Never, so K3s only ever runs these local copies.
HW3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="1.0"
APP_TIERS=(web api db)
APP_NS="hw3"
# shellcheck disable=SC2034 # used by k3s-up.sh and scripts/verify.sh
NODE_PORT=30080                        # must match k8s/web.yaml

VM_CPUS="${VM_CPUS:-2}"
VM_MEMORY="${VM_MEMORY:-3GiB}"
VM_DISK="${VM_DISK:-20GiB}"

# ---- Helpers ----------------------------------------------------------------
log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

node_field() { # node_field <index> <field: 1=name 2=role 3=ip>
  echo "${NODES[$1]}" | awk -v f="$2" '{print $f}'
}

vm_exists()  { incus info "$1" &>/dev/null; }
vm_state()   { incus list "$1" --format csv --columns s 2>/dev/null; }
vm_running() { [[ "$(vm_state "$1")" == "RUNNING" ]]; }

vm_exec() { # vm_exec <name> <shell command>: run as root inside the VM via the Incus agent
  local name="$1"; shift
  incus exec "$name" -- bash -lc "$*"
}

check_prereqs() { # check_prereqs [up]: stop before touching anything if the host cannot work
  # Incus VMs need a Linux kernel with KVM; nothing else is worth checking without one.
  [[ "$(uname -s)" == Linux ]] \
    || die "this environment needs a Linux host with KVM (found $(uname -s)); see README.md, 前置需求"

  # Report every remaining problem at once, so one run tells the user everything to fix.
  local problems=0
  problem() { warn "$1"; problems=$((problems + 1)); }

  [[ -e /dev/kvm ]] \
    || problem "/dev/kvm is missing: enable virtualisation (VT-x/AMD-V) in the BIOS, or nested virtualisation on a VM"
  if ! command -v incus &>/dev/null; then
    problem "Incus is not installed: sudo apt install -y incus qemu-system-x86 && sudo incus admin init --minimal"
  elif ! incus info &>/dev/null; then
    problem "cannot talk to the Incus daemon: sudo usermod -aG incus-admin \$USER, then log out and back in"
  fi
  if [[ "${1:-}" == up ]]; then
    if ! command -v docker &>/dev/null; then
      problem "Docker is not installed (it builds the images): https://docs.docker.com/engine/install/"
    elif ! docker info &>/dev/null; then
      problem "cannot talk to the Docker daemon: sudo usermod -aG docker \$USER, then log out and back in"
    fi
    local cmd
    for cmd in curl openssl; do
      command -v "$cmd" &>/dev/null || problem "$cmd is not installed: sudo apt install -y $cmd"
    done
  fi
  ((${#NET_NAME} <= 15)) || problem "NET_NAME '$NET_NAME' is longer than the 15-character interface limit"

  ((problems == 0)) || die "$problems prerequisite(s) missing; nothing was changed (see README.md, 前置需求)"
}

# ---- Host firewall ----------------------------------------------------------
# UFW's default INPUT DROP swallows the VMs' DHCP/DNS requests to the bridge, and Docker's
# FORWARD DROP stops their traffic to the outside. These rules let the bridge through.
# They need root, so these functions call sudo (which may ask for a password).

# Both checks avoid sudo so hosts without UFW or Docker never get a password prompt.
ufw_active()   { grep -qs '^ENABLED=yes' /etc/ufw/ufw.conf; }
docker_chain() { ip link show docker0 &>/dev/null; }

need_sudo() { # returns 1 (caller skips the firewall step) when sudo cannot prompt
  sudo -n true 2>/dev/null && return 0
  if [[ -t 0 ]]; then
    sudo -v || die "sudo is needed to change the host firewall for $NET_NAME (see README.md, 防火牆)"
  else
    warn "no terminal for a sudo password; skipping firewall rules for $NET_NAME (see README.md, 防火牆)"
    return 1
  fi
}

DOCKER_RULES=(
  "-i $NET_NAME -j ACCEPT"
  "-o $NET_NAME -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT"
)

open_firewall() {
  ufw_active || docker_chain || return 0
  need_sudo || return 0
  if ufw_active; then
    log "Allowing $NET_NAME through UFW (sudo)"
    sudo ufw allow in on "$NET_NAME" >/dev/null
    sudo ufw route allow in on "$NET_NAME" >/dev/null
    sudo ufw route allow out on "$NET_NAME" >/dev/null
  fi
  # DOCKER-USER rules do not survive a host reboot, so this runs on every ./k3s-up.sh.
  if docker_chain; then
    local rule
    for rule in "${DOCKER_RULES[@]}"; do
      # shellcheck disable=SC2086 # the rule is deliberately split into iptables arguments
      sudo iptables -C DOCKER-USER $rule 2>/dev/null || {
        log "Allowing $NET_NAME through Docker's DOCKER-USER chain (sudo)"
        sudo iptables -I DOCKER-USER $rule
      }
    done
  fi
}

close_firewall() {
  ufw_active || docker_chain || return 0
  need_sudo || return 0
  if ufw_active; then
    log "Removing UFW rules for $NET_NAME (sudo)"
    sudo ufw delete allow in on "$NET_NAME" >/dev/null || true
    sudo ufw route delete allow in on "$NET_NAME" >/dev/null || true
    sudo ufw route delete allow out on "$NET_NAME" >/dev/null || true
  fi
  if docker_chain; then
    local rule
    for rule in "${DOCKER_RULES[@]}"; do
      # shellcheck disable=SC2086
      while sudo iptables -C DOCKER-USER $rule 2>/dev/null; do
        log "Removing DOCKER-USER rule for $NET_NAME (sudo)"
        sudo iptables -D DOCKER-USER $rule
      done
    done
  fi
}

# ---- Incus resources --------------------------------------------------------
ensure_network() {
  incus network show "$NET_NAME" &>/dev/null && return
  log "Creating Incus network $NET_NAME ($NET_CIDR, NAT)"
  incus network create "$NET_NAME" \
    ipv4.address="$NET_CIDR" ipv4.nat=true ipv6.address=none >/dev/null
}

ensure_profile() {
  incus profile show "$PROFILE" &>/dev/null || {
    log "Creating Incus profile $PROFILE (${VM_CPUS} vCPU, $VM_MEMORY RAM, $VM_DISK disk)"
    incus profile create "$PROFILE" >/dev/null
  }
  # Re-applied every run so edits here reach the profile; only VMs created afterwards pick up
  # a new disk size, while CPU and memory apply on the next boot.
  incus profile edit "$PROFILE" <<EOF
description: HW3 K3s nodes
config:
  limits.cpu: "$VM_CPUS"
  limits.memory: $VM_MEMORY
devices:
  eth0:
    type: nic
    name: eth0
    network: $NET_NAME
  root:
    type: disk
    path: /
    pool: default
    size: $VM_DISK
EOF
}

create_vm() { # create_vm <index>
  local name ip
  name="$(node_field "$1" 1)"; ip="$(node_field "$1" 3)"
  log "Creating VM $name ($ip)"
  incus init "$IMAGE" "$name" --vm --profile "$PROFILE" >/dev/null
  # Pin the address with a DHCP reservation; K3s needs node IPs that never change.
  incus config device override "$name" eth0 ipv4.address="$ip" >/dev/null
}

wait_for_vm() { # wait_for_vm <name> <ip>: until the agent answers and the network is up
  local name="$1" ip="$2" deadline=$((SECONDS + 300))
  log "Waiting for $name ($ip) to boot"
  until vm_exec "$name" "ip -4 -o addr | grep -q ' $ip/'" &>/dev/null; do
    if ((SECONDS >= deadline)); then
      vm_exec "$name" true &>/dev/null \
        && die "$name booted but never got $ip; the host firewall is probably blocking DHCP on $NET_NAME (see README.md, 防火牆)"
      die "$name did not boot within 5 minutes (try: incus console $name)"
    fi
    sleep 3
  done
}

stop_vm() { # stop_vm <name>: clean shutdown, force off after 2 minutes
  local name="$1"
  vm_running "$name" || return 0
  log "Shutting down $name"
  incus stop "$name" --timeout 120 2>/dev/null || {
    warn "$name ignored the shutdown request, forcing it off"
    incus stop "$name" --force
  }
}

# ---- K3s --------------------------------------------------------------------
server_name() { node_field 0 1; }
server_ip()   { node_field 0 3; }

k3s_installed() { # k3s_installed <name> <unit: k3s|k3s-agent>
  vm_exec "$1" "systemctl list-unit-files $2.service | grep -q $2.service" &>/dev/null
}

vm_iface() { # vm_iface <name> <ip>: the NIC that owns <ip>, for flannel and the node IP
  vm_exec "$1" "ip -4 -o addr | awk -v ip='$2' '\$4 ~ \"^\"ip\"/\" {print \$2}'"
}

run_installer() { # run_installer <name> <shell command>: quiet unless the K3s installer fails
  local out
  out="$(vm_exec "$1" "$2" 2>&1)" || { printf '%s\n' "$out" >&2; die "K3s installation failed on $1"; }
}

install_k3s_server() {
  local name ip iface
  name="$(server_name)"; ip="$(server_ip)"
  if k3s_installed "$name" k3s; then log "K3s server already installed on $name"; return; fi
  iface="$(vm_iface "$name" "$ip")"
  log "Installing K3s $K3S_VERSION server on $name (control-plane, embedded etcd, NoSchedule taint)"
  # No K3S_TOKEN given: K3s generates a random one, which install_k3s_agent reads back.
  # --cluster-init keeps cluster state in embedded etcd instead of SQLite, so more servers can
  # join later for an HA control plane.
  # The taint keeps application Pods off the control plane, as production clusters do.
  # Traefik and ServiceLB are disabled because the app is exposed through a NodePort.
  run_installer "$name" "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION='$K3S_VERSION' sh -s - server \
    --cluster-init \
    --node-ip '$ip' --flannel-iface '$iface' \
    --node-taint node-role.kubernetes.io/control-plane:NoSchedule \
    --disable traefik --disable servicelb \
    --write-kubeconfig-mode 0644"
}

install_k3s_agent() { # install_k3s_agent <index>
  local name ip iface token
  name="$(node_field "$1" 1)"; ip="$(node_field "$1" 3)"
  if k3s_installed "$name" k3s-agent; then log "K3s agent already installed on $name"; return; fi
  iface="$(vm_iface "$name" "$ip")"
  token="$(vm_exec "$(server_name)" 'cat /var/lib/rancher/k3s/server/node-token')"
  log "Installing K3s $K3S_VERSION agent on $name and joining https://$(server_ip):6443"
  run_installer "$name" "curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION='$K3S_VERSION' \
    K3S_URL='https://$(server_ip):6443' K3S_TOKEN='$token' sh -s - agent \
    --node-ip '$ip' --flannel-iface '$iface'"
}

kubectl_server() { # kubectl_server <args...>: kubectl on the control plane
  incus exec "$(server_name)" -- k3s kubectl "$@"
}

wait_for_k3s_services() {
  # After a reboot the API can still report the nodes as Ready from before the shutdown, so
  # first wait for each node's own K3s service and containerd socket to be up.
  local i name unit deadline=$((SECONDS + 300))
  for i in "${!NODES[@]}"; do
    name="$(node_field "$i" 1)"
    [[ "$(node_field "$i" 2)" == server ]] && unit=k3s || unit=k3s-agent
    until vm_exec "$name" "systemctl is-active --quiet $unit && test -S /run/k3s/containerd/containerd.sock" &>/dev/null; do
      ((SECONDS < deadline)) || die "$unit on $name did not start within 5 minutes (try: incus exec $name -- journalctl -u $unit)"
      sleep 2
    done
  done
}

wait_for_nodes() {
  wait_for_k3s_services
  local deadline=$((SECONDS + 300)) want=${#NODES[@]} ready
  log "Waiting for all $want nodes to be Ready"
  while true; do
    # The API server answers with errors for a few seconds after boot; keep polling.
    ready="$( (kubectl_server get nodes --no-headers 2>/dev/null || true) | awk '$2 == "Ready"' | wc -l)"
    ((ready == want)) && break
    ((SECONDS < deadline)) || { kubectl_server get nodes -o wide || true; die "only $ready/$want nodes Ready after 5 minutes"; }
    sleep 3
  done
  kubectl_server get nodes -o wide
}

# ---- Application images -----------------------------------------------------
app_image() { echo "hw3/$1:$IMAGE_TAG"; }
workload() { [[ "$1" == db ]] && echo "statefulset/db" || echo "deployment/$1"; }

# Tiers whose image import_images had to copy (a new build). deploy_app restarts them.
CHANGED_TIERS=()

build_images() {
  local tier
  for tier in "${APP_TIERS[@]}"; do
    log "Building image $(app_image "$tier") from src/$tier"
    docker build --quiet --tag "$(app_image "$tier")" "$HW3_DIR/src/$tier" >/dev/null \
      || die "docker build failed for src/$tier"
  done
}

import_images() { # import_images <vm name>: copy changed images into the node's containerd
  local name="$1" tier image want have
  for tier in "${APP_TIERS[@]}"; do
    image="$(app_image "$tier")"
    # Docker's image ID and the CRI image ID are both the digest of the image config, so equal
    # IDs mean the node already has exactly this build.
    want="$(docker image inspect --format '{{.Id}}' "$image")"
    have="$(vm_exec "$name" "k3s crictl inspecti -o go-template --template '{{.status.id}}' docker.io/$image 2>/dev/null" || true)"
    if [[ "$want" == "$have" ]]; then
      log "$image is already on $name"
      continue
    fi
    log "Importing $image into $name"
    docker save "$image" | incus exec "$name" -- k3s ctr --namespace k8s.io images import - >/dev/null
    [[ " ${CHANGED_TIERS[*]} " == *" $tier "* ]] || CHANGED_TIERS+=("$tier")
  done
}

# ---- Application ------------------------------------------------------------
ensure_db_secret() {
  kubectl_server -n "$APP_NS" get secret db-credentials &>/dev/null && return
  # Generated once per cluster and never written to git. PostgreSQL stores the password in its
  # data directory on first start, so the Secret must survive restarts (it lives in etcd).
  log "Generating database credentials (Secret $APP_NS/db-credentials)"
  kubectl_server -n "$APP_NS" create secret generic db-credentials \
    --from-literal=POSTGRES_USER=names \
    --from-literal=POSTGRES_DB=names \
    --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 24)" >/dev/null
}

apply_manifests() {
  # kubectl -k needs a directory, so copy k8s/ onto the server first (fresh, so files deleted
  # here do not linger there).
  vm_exec "$(server_name)" 'rm -rf /root/k8s'
  incus file push --recursive --quiet "$HW3_DIR/k8s" "$(server_name)/root/"
  kubectl_server apply -k /root/k8s >/dev/null
}

deploy_app() {
  log "Deploying the application (kubectl apply -k k8s/)"
  # The namespace has to exist before the Secret can be created in it.
  kubectl_server apply -f - >/dev/null < "$HW3_DIR/k8s/namespace.yaml"
  ensure_db_secret

  # The tag stays hw3/<tier>:1.0, so a running Pod keeps the old build after a new import.
  # Remember which changed workloads already exist; brand-new ones start on the new image anyway.
  local tier restart=()
  for tier in "${CHANGED_TIERS[@]}"; do
    kubectl_server -n "$APP_NS" get "$(workload "$tier")" &>/dev/null && restart+=("$(workload "$tier")")
  done

  apply_manifests

  if ((${#restart[@]})); then
    log "Restarting ${restart[*]} to pick up the new image"
    kubectl_server -n "$APP_NS" rollout restart "${restart[@]}" >/dev/null
    local w
    for w in "${restart[@]}"; do
      kubectl_server -n "$APP_NS" rollout status "$w" --timeout=180s >/dev/null
    done
  fi
  wait_for_app
  kubectl_server -n "$APP_NS" get pods,svc,pvc -o wide
}

wait_for_app() {
  # `rollout status` can return before a 0 -> 1 scale-up is even observed, so wait for the real
  # thing instead: a request through the NodePort, nginx, the api and PostgreSQL.
  local url deadline=$((SECONDS + 300))
  url="http://$(node_field 1 3):$NODE_PORT/api/health"
  log "Waiting for the app to answer at $url"
  until curl -sf -m 3 "$url" 2>/dev/null | grep -q '"database":"ok"'; do
    if ((SECONDS >= deadline)); then
      kubectl_server -n "$APP_NS" get pods -o wide || true
      die "the app did not become healthy within 5 minutes"
    fi
    sleep 2
  done
}

stop_app() { # scale everything to 0 so PostgreSQL shuts down cleanly before the VMs do
  kubectl_server get namespace "$APP_NS" &>/dev/null || return 0
  log "Stopping the application (scale web, api, db to 0)"
  kubectl_server -n "$APP_NS" scale deployment/web deployment/api statefulset/db --replicas=0 >/dev/null
  kubectl_server -n "$APP_NS" wait --for=delete pod --all --timeout=120s >/dev/null 2>&1 || true
}

stop_k3s() {
  local i name
  for ((i = ${#NODES[@]} - 1; i >= 0; i--)); do
    name="$(node_field "$i" 1)"
    vm_running "$name" || continue
    log "Stopping K3s on $name"
    vm_exec "$name" 'systemctl stop k3s-agent 2>/dev/null; systemctl stop k3s 2>/dev/null; true' >/dev/null
  done
}
