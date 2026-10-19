#!/usr/bin/env bash
# Build HW3_315834011.zip from the committed tree (git archive), so the zip matches git exactly.
set -euo pipefail
cd "$(dirname "$0")/.."

out="HW3_315834011.zip"
git diff --quiet HEAD -- . || { echo "uncommitted changes in $(pwd); commit first" >&2; exit 1; }

rm -f "$out"
git archive --format=zip --output="$out" "HEAD:$(git rev-parse --show-prefix)" -- . \
  ':(exclude)CLAUDE.md' ':(exclude)Homework3.pdf' ':(exclude)scripts/package.sh'
# The assignment lists architecture.pdf at the top level of the zip.
zip -q -j "$out" docs/architecture.pdf
echo "built $out from commit $(git rev-parse --short HEAD)"
