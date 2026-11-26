#!/usr/bin/env bash
# Regression experiments on a dataset (<language>/experiments/regression/<dataset>: data and main);
# results/metrics.csv (every part), figures and the log in its results/. The language is set up by scripts/setup_env.sh;
# on a cluster, submit this script with its arguments to the scheduler (README.md, On a cluster).
#
#   scripts/run_regression.sh <python|matlab|julia|all> <dataset> [parts] [replot] [name=value]     # toy parts: posterior_1d accuracy_1d scaling_1d sparse_grid scaling_nd
# Without parts, every part of the dataset's data PARTS runs.

set -euo pipefail

[ $# -ge 2 ] || { echo "usage: $0 <python|matlab|julia|all> <dataset> [parts] [replot] [name=value]" >&2; exit 2; }
lang="$1"; dataset="$2"; shift 2
source "$(dirname "$0")/setup_env.sh"
if [ "$lang" = all ]; then
    for l in python matlab julia; do "$0" "$l" "$dataset" "$@" || echo "$l: failed" >&2; done
    exit
fi
tag=$([ "$KP_DEVICE" = gpu ] && echo _gpu || true)
dir="$lang/experiments/regression/$dataset"
mkdir -p "$dir/results"
# results/log.txt, or log_<parts>.txt when only some parts run; _gpu appended on a GPU node
log="$dir/results/log${*:+_$(IFS=_; echo "$*")}$tag.txt"
case "$lang" in
    python) python/.venv/bin/python -u "$dir/main.py" "$@" | tee "$log" ;;
    matlab)
        args=""
        if [ $# -gt 0 ]; then args=$(printf "'%s'," "$@"); args=${args%,}; fi
        matlab_cmd -batch "maxNumCompThreads($KP_NUM_THREADS); addpath(genpath('matlab/src')); cd('$dir'); main(${args})" | tee "$log" ;;
    julia) .julia/julia-*/bin/julia --project=julia "$dir/main.jl" "$@" | tee "$log" ;;
    *) echo "unknown language: $lang" >&2; exit 2 ;;
esac
