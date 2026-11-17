#!/bin/bash
# Tier 1: rebuild every summary table, the figure data and the figures from the shipped per-run CSVs, then compare
# each regenerated table with the shipped one. CPU only, a few minutes. Run from the repo root (or `make figures`).
set -euo pipefail
cd "$(dirname "$0")/.."
X=experiments; REF=.reference
SUMMARIES="$X/base/results/summary.csv $X/regen/results/train_summary.csv $X/regen/results/recovery_summary.csv
  $X/regen/results/grid_summary.csv $X/epopt/results/final.csv $X/close/results/train_final.csv
  $X/fig6c/results/final.csv $X/fig6c/results/panel.csv"
# 1. keep the shipped versions aside for the comparison (re-running restores them first, so it is idempotent)
if [ -d $REF ]; then (cd $REF && find . -type f) | while read -r f; do cp "$REF/$f" "$f"; done; fi
rm -rf $REF; mkdir -p $REF
for f in $SUMMARIES $X/regen/figures/data/*; do mkdir -p "$REF/$(dirname "$f")"; cp "$f" "$REF/$f"; done
# 2. summaries (order matters: EPOPT feeds CLOSE, CLOSE feeds 6C)
python3 $X/base/aggregate.py
python3 $X/regen/aggregate.py
python3 $X/epopt/summarize.py
python3 $X/close/summarize.py
python3 $X/fig6c/summarize.py
# 3. figure tables and macros, then the figures (PDF + SVG next to the captions)
python3 $X/regen/figures/export.py
python3 $X/regen/figures/src/fig6_basin.py
python3 $X/regen/figures/src/fig7_recovery.py
# 4. compare with the shipped versions
python3 scripts/compare.py $REF $SUMMARIES $X/regen/figures/data/*
for f in fig6_basin fig7_recovery; do
  n=$(pdffonts $X/regen/figures/$f.pdf | grep -c "Type 3" || true); echo "$f.pdf: $n Type 3 fonts"
done
