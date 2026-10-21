#!/bin/bash
# Shared environment preamble for the experiment launchers, sourced by every scripts/<experiment>.sh.
#
# Everything is overridable from the environment:
#   REPO_DIR      repo root (default: the directory sbatch was submitted from, else the cwd)
#   CONDA_ENV     conda env name or path to activate (default: semantic-ot, see environment.yml)
#   CUDA_MODULE   environment module to load first, e.g. "cuda/12.8.1" (default: none)
#   CACHE_ROOT    if set, HuggingFace / torch / pip caches are redirected under it
#   DEVICE_ID     GPU index the scripts use as cuda:<DEVICE_ID> (default: 0)
#
# No `set -e`/`set -u` here: `conda activate` trips on them.

if [[ -n "${CUDA_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
    module load "$CUDA_MODULE"
fi

export REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

if [[ -n "${CACHE_ROOT:-}" ]]; then
    export PIP_CACHE_DIR="$CACHE_ROOT/pip"
    export HF_HOME="$CACHE_ROOT/huggingface"
    export TORCH_HOME="$CACHE_ROOT/torch"
    export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
    export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
    mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
fi

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-semantic-ot}"
fi

cd "$REPO_DIR"

# Under SLURM only the allocated GPUs are visible, so a single-GPU allocation is always device 0.
DEVICE_ID="${DEVICE_ID:-0}"
