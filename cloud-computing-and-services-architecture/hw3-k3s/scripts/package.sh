#!/usr/bin/env bash
# Build HW3_315834011.zip from the committed tree (git archive), so the zip matches git exactly.
set -euo pipefail
cd "$(dirname "$0")/.."

out="$PWD/HW3_315834011.zip"
prefix="$(git rev-parse --show-prefix)"
git diff --quiet HEAD -- . || { echo "uncommitted changes in $PWD; commit first" >&2; exit 1; }

rm -f "$out"
trap 'rm -f "$out"' ERR
# Pathspecs of `git archive HEAD:<dir>` only resolve from the repository root.
git -C "$(git rev-parse --show-toplevel)" archive --format=zip --output="$out" "HEAD:$prefix" -- \
  ':(exclude)CLAUDE.md' ':(exclude)Homework3.pdf' ':(exclude)scripts/package.sh'
# The assignment lists architecture.pdf at the top level of the zip.
zip -q -j "$out" docs/architecture.pdf
echo "built $(basename "$out") from commit $(git rev-parse --short HEAD)"
