#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TFMFONTS="$(pwd)/../scaled-contact-article/fonts/blackboard:${TFMFONTS-}"
export T1FONTS="$(pwd)/../scaled-contact-article/fonts/blackboard:${T1FONTS-}"
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
