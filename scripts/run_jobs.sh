#!/bin/bash
# Tiers 2 and 3: run job files (jobs/*.txt; one command per line, generated from each shipped run's provenance).
# Each run rewrites its shipped CSV in place; `git diff --stat experiments` shows what changed and `make figures` rebuilds
# the figures from the new numbers. Usage:
#   scripts/run_jobs.sh [-P parallel] [-n first-N-lines] [-g grep-pattern] jobs/fig7_2_damage.txt [more job files...]
# Files run one after the other (fig6b_1_train before fig6b_2_probe, fig7_1_pretrain before fig7_2_damage); the lines in
# a file run P at a time. Logs go to logs/. Most jobs need an NVIDIA GPU (fig6c's ReLU runs, base/ and fig6a are CPU).
set -uo pipefail
cd "$(dirname "$0")/.."
P=1; N=0; G=""
while getopts "P:n:g:" o; do case $o in P) P=$OPTARG;; n) N=$OPTARG;; g) G=$OPTARG;; *) exit 2;; esac; done
shift $((OPTIND - 1)); [ $# -gt 0 ] || { sed -n 2,9p "$0"; exit 2; }
mkdir -p logs
one() {
  tag=$(echo "$*" | sed 's/.*experiments\///; s/[^A-Za-z0-9._-]/_/g' | cut -c1-150)
  echo "$(date +%T) start $tag"; bash -c "$*" > "logs/$tag.log" 2>&1; rc=$?
  echo "$(date +%T) end   $tag rc=$rc"; return $rc
}
export -f one
for f in "$@"; do
  grep -v '^#' "$f" | grep -v '^\s*$' | { [ -n "$G" ] && grep -- "$G" || cat; } | { [ "$N" -gt 0 ] && head -n "$N" || cat; } \
    | xargs -P "$P" -I{} bash -c 'one {}'
done
