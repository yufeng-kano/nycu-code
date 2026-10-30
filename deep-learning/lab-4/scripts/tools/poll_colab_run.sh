#!/usr/bin/env bash
# Keep a Colab CLI session alive and mirror one notebook run's log and resumable checkpoint to this machine.
#
# usage (from the lab root, in the background):  scripts/tools/poll_colab_run.sh SESSION NAME [INTERVAL_S]
#   SESSION  colab session name (e.g. lab4)       NAME  experiment name (e.g. 002-mini-sam-g4)
#
# Every INTERVAL_S (default 600) it runs a status script on the VM (which also counts as session activity),
# mirrors logs/NAME.log, and every third round checkpoints/NAME-last.pt. When the VM reports the nbconvert
# process gone it downloads the executed notebook, the best checkpoint and the CSV, prints DONE and exits 0.
# Exits 2 after six consecutive failed status calls (session reclaimed): recreate the session, re-fetch the
# data, upload checkpoints/NAME-last.pt + logs/NAME.log + notebooks/NAME.ipynb and relaunch; the run resumes.
set -u
SESSION=$1; NAME=$2; INTERVAL=${3:-600}
export PATH="$HOME/.local/bin:$PATH"
STATUS_PY=$(mktemp "${TMPDIR:-/tmp}/vm_status_${NAME}_XXXX.py")
cat > "$STATUS_PY" <<PYEOF
import os, subprocess
from pathlib import Path
lab = Path("/content/lab-4"); name = "$NAME"
pidf = lab / "logs" / f"{name}.pid"
pid = int(pidf.read_text()) if pidf.exists() else -1
def is_alive(pid: int) -> bool:
    """A finished nbconvert stays in /proc as a zombie until the kernel that launched it reaps it; reap and treat Z as dead."""
    if pid <= 0 or not Path(f"/proc/{pid}").exists():
        return False
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, IndexError):
        return False
    return state not in ("Z", "X")
alive = is_alive(pid)
print(f"[{name}] nbconvert pid {pid} alive={alive}")
log = lab / "logs" / f"{name}.log"
if log.exists():
    lines = log.read_text().splitlines()
    keep = [l for l in lines if l.startswith(("epoch ", "RESULT", "Resumed", "Traceback", "Saved")) or "Error" in l]
    print(f"log {len(lines)} lines; last: " + " || ".join(l[:160] for l in keep[-2:]))
nbl = lab / "logs" / f"{name}.nbconvert.log"
if nbl.exists():
    tail = [l for l in nbl.read_text(errors="replace").splitlines() if "colab_request" not in l and "frozen" not in l and "Debugging" not in l and l.strip()][-3:]
    if tail:
        print("nbconvert.log: " + " || ".join(l[:160] for l in tail))
r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True)
print("gpu " + r.stdout.strip())
PYEOF
dl() {  # remote path relative to /content/lab-4, local path
  if timeout 900 colab download -s "$SESSION" "/content/lab-4/$1" "$2.part" >/dev/null 2>&1; then mv -f "$2.part" "$2"; else rm -f "$2.part"; echo "download failed: $1"; fi
}
i=0; fails=0
while true; do
  i=$((i + 1))
  out=$(timeout 300 colab exec -s "$SESSION" -f "$STATUS_PY" --timeout 120 2>&1); rc=$?
  echo "--- $(date '+%m-%d %H:%M') poll $i rc=$rc"; echo "$out" | grep -v "colab_request"
  if ! echo "$out" | grep -q "alive="; then
    fails=$((fails + 1)); echo "status call failed ($fails/6)"
    if [ "$fails" -ge 6 ]; then echo "SESSION LOST $SESSION"; exit 2; fi
    sleep 60; continue
  fi
  fails=0
  dl "logs/$NAME.log" "logs/$NAME.log"
  if echo "$out" | grep -q "alive=False"; then
    echo "nbconvert gone: downloading artifacts"
    dl "notebooks/$NAME.ipynb" "notebooks/$NAME.ipynb" && /app/.venv/bin/python scripts/tools/fix_colab_notebook.py "notebooks/$NAME.ipynb"
    dl "checkpoints/$NAME.pt" "checkpoints/$NAME.pt"
    dl "checkpoints/$NAME-last.pt" "checkpoints/$NAME-last.pt"
    dl "result/$NAME.csv" "result/$NAME.csv"
    echo "DONE $NAME $(grep -c '^epoch ' "logs/$NAME.log" 2>/dev/null) epochs in log; $(grep '^RESULT' "logs/$NAME.log" 2>/dev/null | tail -1)"
    exit 0
  fi
  if [ $((i % 3)) -eq 0 ]; then dl "checkpoints/$NAME-last.pt" "checkpoints/$NAME-last.pt"; fi
  sleep "$INTERVAL"
done
