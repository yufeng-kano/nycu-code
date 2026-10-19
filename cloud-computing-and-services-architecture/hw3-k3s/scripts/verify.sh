#!/usr/bin/env bash
# Run the five verification steps from the assignment and save each transcript to evidence/.
#   ./scripts/verify.sh          all five (step 5 shuts the environment down and up again)
#   ./scripts/verify.sh 2 3 4    only the listed steps
# Every command is echoed with a leading "$ " before its output, so the files read as a session log.
# shellcheck source=scripts/common.sh
source "$(dirname "$0")/common.sh" || exit 1

EVIDENCE="$HW3_DIR/evidence"
WEB="http://$(node_field 1 3):$NODE_PORT"
API="$WEB/api"
mkdir -p "$EVIDENCE"
failures=0

run() { # run <command...>: echo it like a shell prompt, then run it
  printf '\n$ %s\n' "$*"
  bash -c "$*" 2>&1 || true
}

check() { # check <description> <command...>: PASS when the command succeeds
  local what="$1"; shift
  if bash -c "$*" &>/dev/null; then
    printf '\n[PASS] %s\n' "$what"
  else
    printf '\n[FAIL] %s\n' "$what"
  fi
}

header() { printf '# %s\n# %s\n' "$1" "$(date -Is)"; }

K="./scripts/kubectl.sh"   # relative, so the evidence shows short commands
cd "$HW3_DIR"

step1() {
  header "1. Cluster startup: one command brings up the cluster and the app; both nodes Ready"
  run ./k3s-up.sh
  run "$K get nodes -o wide"
  check "both nodes are Ready" "[[ \$($K get nodes --no-headers | awk '\$2==\"Ready\"' | wc -l) == 2 ]]"
  run "$K -n $APP_NS get deploy,statefulset,pods -o wide"
}

step2() {
  header "2. Application functionality through the nginx NodePort ($WEB)"
  run "curl -s $WEB/ | grep -o '<title>.*</title>'"
  run "curl -s -X POST -H 'Content-Type: application/json' -d '{\"name\":\"Alice\"}' $API/add_name"
  run "curl -s -X POST -H 'Content-Type: application/json' -d '{\"name\":\"Bob\"}' $API/add_name"
  run "curl -s $API/get_names"
  run "curl -s -X POST -H 'Content-Type: application/json' -d '{\"name\":\"Alice\"}' $API/add_name   # duplicate"
  run "curl -s -X DELETE '$API/remove_name?name=Bob'"
  run "curl -s $API/get_names"
  check "Alice is listed and Bob was removed" \
    "curl -s $API/get_names | grep -q '\"Alice\"' && ! curl -s $API/get_names | grep -q '\"Bob\"'"
  run "curl -s -X DELETE '$API/remove_name?name=Alice'"
  run "curl -s -o /dev/null -w 'NodePort on the server node (192.168.56.10): HTTP %{http_code}\n' http://$(server_ip):$NODE_PORT/"
}

step3() {
  header "3. Inter-tier connectivity through Kubernetes Services"
  run "$K -n $APP_NS get svc,endpointslices -o wide"
  echo; echo "## web -> api: nginx proxies /api/ to the 'api' Service"
  run "$K -n $APP_NS exec deploy/web -- grep -n 'server api' /etc/nginx/conf.d/default.conf"
  run "$K -n $APP_NS exec deploy/web -- wget -qO- http://api:8000/api/health"
  echo; echo "## api -> db: Flask connects to the 'db' Service"
  run "$K -n $APP_NS exec deploy/api -- printenv PGHOST PGPORT PGDATABASE"
  run "$K -n $APP_NS exec deploy/api -- getent hosts db"
  run "$K -n $APP_NS exec deploy/api -- /srv/.venv/bin/python -c \"import psycopg; c = psycopg.connect(); print(c.info.host, c.execute('select version()').fetchone()[0])\""
  check "web reaches api, and api reaches db (health reports database ok)" \
    "$K -n $APP_NS exec deploy/web -- wget -qO- http://api:8000/api/health | grep -q '\"database\":\"ok\"'"
}

step4() {
  local name old new
  name="persist-$(date +%s)"
  header "4. Data persistence: insert, delete the PostgreSQL Pod, verify after it is recreated"
  run "$K -n $APP_NS get pvc data-db-0"
  run "curl -s -X POST -H 'Content-Type: application/json' -d '{\"name\":\"$name\"}' $API/add_name"
  old="$($K -n "$APP_NS" get pod db-0 -o jsonpath='{.metadata.uid}')"
  run "$K -n $APP_NS get pod db-0 -o custom-columns=POD:.metadata.name,UID:.metadata.uid,NODE:.spec.nodeName"
  run "$K -n $APP_NS delete pod db-0"
  run "$K -n $APP_NS wait --for=condition=Ready pod/db-0 --timeout=120s"
  run "$K -n $APP_NS rollout status deployment/api --timeout=60s"
  new="$($K -n "$APP_NS" get pod db-0 -o jsonpath='{.metadata.uid}')"
  run "$K -n $APP_NS get pod db-0 -o custom-columns=POD:.metadata.name,UID:.metadata.uid,NODE:.spec.nodeName"
  run "$K -n $APP_NS get pvc data-db-0"
  # The api may need a moment to notice the new db Pod is Ready; retry briefly.
  run "for i in \$(seq 20); do curl -sf $API/get_names && break; sleep 2; done"
  check "db-0 was recreated (UID changed)" "[[ '$old' != '$new' ]]"
  check "record '$name' survived the Pod recreation" "curl -s $API/get_names | grep -q '$name'"
}

step5() {
  local name
  name="restart-$(date +%s)"
  header "5. Cluster restart: ./k3s-down.sh then ./k3s-up.sh; app and data still work"
  run "curl -s -X POST -H 'Content-Type: application/json' -d '{\"name\":\"$name\"}' $API/add_name"
  run ./k3s-down.sh
  run "incus list hw3- --columns ns4t"
  run ./k3s-up.sh
  run "$K get nodes"
  run "curl -s $API/get_names"
  check "both nodes are Ready after restart" "[[ \$($K get nodes --no-headers | awk '\$2==\"Ready\"' | wc -l) == 2 ]]"
  check "record '$name' survived the restart" "curl -s $API/get_names | grep -q '$name'"
  check "the app still accepts writes" \
    "curl -s -X POST -H 'Content-Type: application/json' -d '{\"name\":\"after-$name\"}' $API/add_name | grep -q after-"
}

files=([1]=1-startup [2]=2-app-functionality [3]=3-inter-tier-connectivity [4]=4-data-persistence [5]=5-cluster-restart)
steps=("$@"); ((${#steps[@]})) || steps=(1 2 3 4 5)
for n in "${steps[@]}"; do
  [[ -n "${files[$n]:-}" ]] || die "unknown step: $n (use 1-5)"
  out="$EVIDENCE/${files[$n]}.txt"
  log "Step $n -> evidence/${files[$n]}.txt"
  # Strip colour codes so the evidence files stay plain text. The step runs in a pipeline
  # subshell, so failures are counted from the [FAIL] lines in its file.
  "step$n" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tee "$out" | grep -E '^\[(PASS|FAIL)\]' || true
  grep -q '^\[FAIL\]' "$out" && failures=$((failures + 1))
done

if ((failures > 0)); then die "$failures step(s) had failures; see evidence/"; fi
log "All checks passed."
