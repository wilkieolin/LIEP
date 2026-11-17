#!/bin/bash
# Make the Julia depot usable from a read-only image (Apptainer) or as a non-root user: the user's depot comes first,
# the image's depot (packages, precompiled caches) second; FashionMNIST is copied into the user's scratch space once.
set -e
U=${JULIA_USER_DEPOT:-$HOME/.julia-liep}
export JULIA_DEPOT_PATH="$U:/opt/julia-depot"
uuid=$(ls /opt/julia-depot/scratchspaces 2>/dev/null | head -1)
if [ -n "$uuid" ] && [ ! -d "$U/scratchspaces/$uuid/fashion_mnist" ]; then
    mkdir -p "$U/scratchspaces/$uuid" && cp -r /opt/data/fashion_mnist "$U/scratchspaces/$uuid/"
fi
exec "$@"
