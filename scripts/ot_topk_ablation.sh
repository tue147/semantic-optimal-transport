#!/bin/bash
#SBATCH --job-name=ot_topk_abl
#SBATCH --output=logs/ot_topk_ablation_%j.out
#SBATCH --error=logs/ot_topk_ablation_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#
# Appendix D (Table 4): OT matching LLM Eval / Acc vs support size K in {1,16,256,500}, on the
# 100 query features of Table 1, 5 judge runs. Run feature_matching_test.py for the pair first:
# QUERY_PREV_FILE points at its output for our method.
# WARNING: fans out paid LLM-judge calls.
# Submit:  sbatch scripts/ot_topk_ablation.sh
#          sbatch --export=ALL,PRESET=gpt2_ds,NUM_BATCHES=1000,PAIRS=11,5,CACHING_HOOK=ln_final.hook_normalized scripts/ot_topk_ablation.sh

_COMMON="$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[ -f "$_COMMON" ] || _COMMON="${SLURM_SUBMIT_DIR:-.}/scripts/_common.sh"
source "$_COMMON"
mkdir -p logs

# ---- model / SAE -----------------------------------------------------------------------------
PRESET="${PRESET:-gemma_mlp}"         # gemma_mlp | gpt2_ds

# ---- token extraction ------------------------------------------------------------------------
DATA_PATH="monology/pile-uncopyrighted"
MAX_LENGTH=128
NUM_BATCHES="${NUM_BATCHES:-10000}"   # 10000 sequences for Gemma-2-2B, 1000 for GPT2

# ---- ablation --------------------------------------------------------------------------------
PAIRS="${PAIRS:-13,12 25,12 25,0}"    # layer pairs A,B (A later)
K_GRID="1 16 256 500"                 # support sizes (max_points_per_feature)
NUMBER_OF_PAIRS=100
MIN_ACTIVATION_EXAMPLES=30
SEEDS="1 2 3 4 5"
QUERY_PREV_FILE="${QUERY_PREV_FILE:-ot_{a}_{b}_1_100_30_run1.pt}"   # Table 1 query features
CACHING_HOOK="${CACHING_HOOK:-}"      # empty = layer A; GPT2 target layer 11 used ln_final.hook_normalized

# ---- ot --------------------------------------------------------------------------------------
M=25
MAX_DENSITY=0.1
METRIC="euclidean"
OT_METHOD="sinkhorn"
SINKHORN_IMPL="old"                   # the helper behind the paper's numbers
CHUNK_SIZE=2048
BATCH_SIZE=96

# ---- LLM judge -------------------------------------------------------------------------------
JUDGE="gpt-oss-120b"

python ot_topk_ablation.py \
    --preset "$PRESET" \
    --device_id "$DEVICE_ID" \
    --data_path "$DATA_PATH" \
    --max_length "$MAX_LENGTH" \
    --num_batches "$NUM_BATCHES" \
    --pairs $PAIRS \
    --K_grid $K_GRID \
    --number_of_pairs "$NUMBER_OF_PAIRS" \
    --min_activation_examples "$MIN_ACTIVATION_EXAMPLES" \
    --seeds $SEEDS \
    --query_prev_file "$QUERY_PREV_FILE" \
    ${CACHING_HOOK:+--caching_hook "$CACHING_HOOK"} \
    --M "$M" \
    --max_density "$MAX_DENSITY" \
    --metric "$METRIC" \
    --ot_method "$OT_METHOD" \
    --sinkhorn_impl "$SINKHORN_IMPL" \
    --chunk_size "$CHUNK_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --judge "$JUDGE"
