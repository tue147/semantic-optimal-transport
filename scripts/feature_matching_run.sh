#!/bin/bash
#SBATCH --job-name=feat_match_run
#SBATCH --output=logs/feature_matching_run_%j.out
#SBATCH --error=logs/feature_matching_run_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#
# Table 1 / Table 6: produce cross-layer SAE feature matchings for one layer pair (Table 3).
# Submit:  sbatch --export=ALL,SAE_A_ID=13,SAE_B_ID=12 scripts/feature_matching_run.sh
#          sbatch --export=ALL,PRESET=gpt2_ds,NUM_BATCHES=1000,SAE_A_ID=6,SAE_B_ID=5 scripts/feature_matching_run.sh
#          sbatch --export=ALL,PRESET=gpt2_ds,NUM_BATCHES=1000,SAE_A_ID=11,SAE_B_ID=5,CACHING_HOOK=ln_final.hook_normalized scripts/feature_matching_run.sh

# sbatch copies this script to the compute node's spool dir, so BASH_SOURCE no longer points
# into the repo. Fall back to the directory sbatch was submitted from (the repo root).
_COMMON="$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[ -f "$_COMMON" ] || _COMMON="${SLURM_SUBMIT_DIR:-.}/scripts/_common.sh"
source "$_COMMON"
mkdir -p logs

# ---- model / SAE -----------------------------------------------------------------------------
# Presets: gemma_mlp | gemma_ds | gpt2_ds | llama_post | qwen_mlp
PRESET="${PRESET:-gemma_mlp}"

# ---- activation cache ------------------------------------------------------------------------
DATA_PATH="monology/pile-uncopyrighted"
BATCHSIZE=64
MAX_LENGTH=128
NUM_BATCHES="${NUM_BATCHES:-10000}"   # 10000 sequences for Gemma-2-2B, 1000 for GPT2
CACHE_DIR="data/cache_data"

# ---- matching --------------------------------------------------------------------------------
# Methods: ot | mech_permute | feat_flow | naive | mech_permute_ml | feat_flow_ml
# Several may be listed; they share the loaded model and activation cache.
# (causal is not produced here; the eval scripts compute it on the fly from the sampled features.)
METHOD_NAME="${METHOD_NAME:-ot feat_flow mech_permute naive feat_flow_ml mech_permute_ml}"
SAE_A_ID="${SAE_A_ID:-13}"            # target (later) layer, must exceed SAE_B_ID
SAE_B_ID="${SAE_B_ID:-12}"            # source (earlier) layer
K=25
MAX_DENSITY=0.1
BATCH_SIZE=32
CHUNK_SIZE=2048
METRIC="euclidean"

# ---- ot only ---------------------------------------------------------------------------------
M=25
MAX_POINTS_PER_FEATURE=500
OT_METHOD="sinkhorn"
SINKHORN_IMPL="old"          # the Sinkhorn helper behind the paper's numbers; "fixed" = true W_2
CACHING_HOOK="${CACHING_HOOK:-}"   # reference space; empty = layer A. GPT2 target layer 11: ln_final.hook_normalized

# ---- mech_permute only -----------------------------------------------------------------------
USE_LAP=true

python feature_matching_run.py \
    --preset "$PRESET" \
    --device_id "$DEVICE_ID" \
    --data_path "$DATA_PATH" \
    --batchsize "$BATCHSIZE" \
    --max_length "$MAX_LENGTH" \
    --num_batches "$NUM_BATCHES" \
    --cache_dir "$CACHE_DIR" \
    --method_name $METHOD_NAME \
    --sae_a_id "$SAE_A_ID" \
    --sae_b_id "$SAE_B_ID" \
    --k "$K" \
    --max_density "$MAX_DENSITY" \
    --batch_size "$BATCH_SIZE" \
    --chunk_size "$CHUNK_SIZE" \
    --metric "$METRIC" \
    --M "$M" \
    --max_points_per_feature "$MAX_POINTS_PER_FEATURE" \
    --ot_method "$OT_METHOD" \
    --sinkhorn_impl "$SINKHORN_IMPL" \
    ${CACHING_HOOK:+--caching_hook "$CACHING_HOOK"} \
    --use_lap "$USE_LAP"
