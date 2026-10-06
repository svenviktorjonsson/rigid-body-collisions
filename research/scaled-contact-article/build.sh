#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TFMFONTS="$(pwd)/fonts/blackboard:${TFMFONTS-}"
export T1FONTS="$(pwd)/fonts/blackboard:${T1FONTS-}"
pdflatex -interaction=nonstopmode -halt-on-error article.tex
pdflatex -interaction=nonstopmode -halt-on-error article.tex
