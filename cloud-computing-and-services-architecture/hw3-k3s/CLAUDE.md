# CLAUDE.md: HW3 K3s migration

Notes for Claude working in this folder. README.md is for the grader and stays short and readable. Status, decisions, logs and to-dos go here, never in README.

## Status (2026-10-19)

- **Done** everything the assignment lists: scripts, `src/`, `k8s/`, `evidence/` (all 5 checks pass), `docs/architecture.pdf`, `comparison.md`, plus `docs/guide.md` and `docs/critique.md`. Merged into `main` and pushed.
- **Submission** `./scripts/package.sh` rebuilds `HW3_315834011.zip` from the committed tree (zip files are git-ignored). Commit first; the script refuses a dirty tree.
- **Optional** the user reruns `./scripts/verify.sh` in a real terminal, so the evidence loses the "skipping firewall rules" warning that no-TTY runs print; then commit `evidence/` and rebuild the zip.

## Decisions (made by the user, do not reopen)

- Assignment Option 2 on Incus VMs (KVM underneath). Vagrant rejected, see `docs/critique.md`. Plain libvirt was tried first and replaced by Incus.
- Bridge `nycu-cc-hw3-br` (Linux caps interface names at 15 chars). Subnet `192.168.56.0/24` is fixed on purpose, no env override. Server `.10`, worker `.11`.
- K3s pinned to `v1.36.5+k3s1`. The server runs embedded etcd (`--cluster-init`) and carries a `NoSchedule` taint, so app Pods only run on the worker.
- PostgreSQL is a StatefulSet although the assignment says Deployment; README states the deviation.
- Every tier has its own Dockerfile, db included, for consistency.
- Images are built on the host first (fail fast before any VM work) and imported into the worker's containerd. No registry.
- Manifests use Kustomize (`kubectl apply -k k8s/`). No numeric filename prefixes anywhere.
- Prefer production practice wherever it is cheap, and list every deviation (`docs/guide.md`, 與 production 的差異).
- Other machines reach the app through `ssh -L`; no LAN port forward.

## Conventions

- Docs are Traditional Chinese and follow repo memory `writing-style` and `document-readability`: short sentences, flat `##` plus bullets, abbreviations spelled out on first use, no emoji.
- Code and comments are English. Python is fully typed.
- Shell: `scripts/common.sh` sets `set -euo pipefail`; every step is idempotent; sudo steps warn and skip when there is no TTY.
- After changing scripts or manifests, rerun `./scripts/verify.sh` (or the affected steps) so `evidence/` stays true.

## Running things from a Claude session

- sudo needs the user's password and the session has no TTY, so the firewall step is skipped; the user runs it.
- If `incus` reports permission denied, the session predates the `incus-admin` group: wrap the call in `sg incus-admin -c '...'`.
- `docs/build.sh` needs `rsvg-convert` (apt `librsvg2-bin`); if it is missing, point `RSVG_CONVERT` at a local copy. XeTeX picks the HK face of `PingFang.ttc` by name, so `architecture.tex` selects the TC faces by `FontIndex`.

## Verification log

- 2026-10-19: from scratch (`./k3s-down.sh --delete`, then `./scripts/verify.sh`) all 5 checks pass in about 2 minutes.
- Bugs those runs caught, now fixed: after a reboot the API still reports the nodes Ready from before shutdown, so up waits for each node's K3s service and containerd socket; `kubectl rollout status` returns before a 0 → 1 scale-up starts, so up waits for an end-to-end `/api/health` instead; a rebuilt image keeps its tag, so up now restarts workloads whose image it re-imported.
- 2026-10-19, after the api code-style pass (keyword arguments everywhere the API allows them): ruff, mypy and the 4 PostgreSQL tests pass; from-scratch `verify.sh` passes all 5 checks again (about 2 minutes).

## Packaging (HW3_315834011.zip)

`./scripts/package.sh` runs `git archive` on this folder at HEAD, leaves out CLAUDE.md, Homework3.pdf and the script itself, and adds `docs/architecture.pdf` at the zip root as the assignment asks.
