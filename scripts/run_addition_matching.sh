#!/bin/bash
#SBATCH --job-name=addition_match
#SBATCH --output=logs/run_addition_matching_%j.out
#SBATCH --error=logs/run_addition_matching_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#
# Two-digit addition case study: intra-SAE feature matching.
# WARNING: METHOD_NAMES="llm" fans out paid OpenAI calls.
# Submit:  sbatch scripts/run_addition_matching.sh

# sbatch copies this script to the compute node's spool dir, so BASH_SOURCE no longer points
# into the repo. Fall back to the directory sbatch was submitted from (the repo root).
_COMMON="$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[ -f "$_COMMON" ] || _COMMON="${SLURM_SUBMIT_DIR:-.}/scripts/_common.sh"
source "$_COMMON"
mkdir -p logs

# ---- model / SAE -----------------------------------------------------------------------------
# Presets: gemma_mlp | gpt2_ds
PRESET="gemma_mlp"

# ---- addition matching -----------------------------------------------------------------------
# Methods: ot | naive | feat_flow | mech_permute | llm  (several may be listed)
METHOD_NAMES="${METHOD_NAMES:-ot feat_flow mech_permute naive llm}"
SAE_LAYER=19
K=1
BATCH_SIZE=32
DATA_RANGE="0 9"            # MIN_VAL MAX_VAL, inclusive operand range
ACT_FREQ_RANGE="5 50"       # LO HI, last-token firing count kept
LLM_CHUNK_SIZE=20

# ---- activation cache ------------------------------------------------------------------------
CACHE_DIR="data/cache_data"

OUTPUT_DIR="."

python run_addition_matching.py \
    --preset "$PRESET" \
    --device_id "$DEVICE_ID" \
    --method_names $METHOD_NAMES \
    --sae_layer "$SAE_LAYER" \
    --k "$K" \
    --batch_size "$BATCH_SIZE" \
    --data_range $DATA_RANGE \
    --act_freq_range $ACT_FREQ_RANGE \
    --llm_chunk_size "$LLM_CHUNK_SIZE" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR"
