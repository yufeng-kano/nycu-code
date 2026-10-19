#!/usr/bin/env bash
# Build architecture.pdf from architecture.svg + architecture.tex.
# Needs rsvg-convert (apt: librsvg2-bin), xelatex with xeCJK, and the PingFang TC / SF Pro fonts.
set -euo pipefail
cd "$(dirname "$0")"

RSVG_CONVERT="${RSVG_CONVERT:-rsvg-convert}"
# architecture.tex loads PingFang.ttc by file name (to pick the TC face by index), so let XeTeX's
# file search see the user and system font directories.
export OSFONTDIR="${OSFONTDIR:-$HOME/.local/share/fonts//:/usr/share/fonts//:/usr/local/share/fonts//}"

# LaTeX cannot read SVG directly; convert it to a vector PDF first (text stays text).
"$RSVG_CONVERT" --format=pdf --output=architecture-diagram.pdf architecture.svg
# Twice, so any cross-references settle.
xelatex -interaction=nonstopmode -halt-on-error architecture.tex >/dev/null
xelatex -interaction=nonstopmode -halt-on-error architecture.tex >/dev/null
rm -f architecture.aux architecture.log architecture.out
echo "built docs/architecture.pdf"
