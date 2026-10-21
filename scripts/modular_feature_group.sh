#!/bin/bash
#SBATCH --job-name=modular_feat
#SBATCH --output=logs/modular_feature_group_%j.out
#SBATCH --error=logs/modular_feature_group_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#
# Inspect one behaviour cluster and build its modular feature groups.
# Requires the circuit_behavior_clustering output identified by the CLUSTER_* variables below.
# Submit:  sbatch scripts/modular_feature_group.sh

# sbatch copies this script to the compute node's spool dir, so BASH_SOURCE no longer points
# into the repo. Fall back to the directory sbatch was submitted from (the repo root).
_COMMON="$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[ -f "$_COMMON" ] || _COMMON="${SLURM_SUBMIT_DIR:-.}/scripts/_common.sh"
source "$_COMMON"
mkdir -p logs

# ---- model / SAE -----------------------------------------------------------------------------
# Presets: gemma_mlp | gemma_ds | gpt2_ds | llama_post | qwen_mlp
PRESET="gemma_mlp"

# ---- activation cache ------------------------------------------------------------------------
DATA_PATH="monology/pile-uncopyrighted"
BATCHSIZE=64
MAX_LENGTH=128
NUM_BATCHES=10000
CACHE_DIR="data/cache_data"

# ---- behaviour cluster input -----------------------------------------------------------------
# These three reconstruct the circuit_behavior_clustering file name; set CLUSTERING_FILE to
# override it outright.
CLUSTER_ID=307
CLUSTERING_DIR="."
CLUSTER_NODE_THRESHOLD=50
CLUSTER_N_CLUSTERS=500
CLUSTER_DATA_RANGE="0 18000"
SHOW_MAX_TO_SHOW=20

# ---- circuit construction / pruning (node pruning only) --------------------------------------
METHOD_NAME="fc"
USE_ERROR_TERM=true
PASS_THROUGH_GRAD=true
REVERSE_PRUNING=false
GRADIENT_MODE="standard"
EDGE_GRADIENT_MODE="gradient"
IG_STEPS=10
TRANSFER_GRAD=true
SAVE_NODE_GRADS=false

PRUNE_TYPE="attrib"
SCORING_MODE="less"

NODE_THRESHOLD_TYPE="number"
NODE_THRESHOLD=200

DIRECT_EFFECT=false

# ---- supernode clustering --------------------------------------------------------------------
N_CLUSTERS=10
ALPHA=0.8
LAMBDA_PEN=1.0
N_INTERVALS=5
K_PER_INTERVAL=5
BUFFER=10

OUTPUT_DIR="."

python modular_feature_group.py \
    --preset "$PRESET" \
    --device_id "$DEVICE_ID" \
    --data_path "$DATA_PATH" \
    --batchsize "$BATCHSIZE" \
    --max_length "$MAX_LENGTH" \
    --num_batches "$NUM_BATCHES" \
    --cache_dir "$CACHE_DIR" \
    --cluster_id "$CLUSTER_ID" \
    --clustering_dir "$CLUSTERING_DIR" \
    --cluster_node_threshold "$CLUSTER_NODE_THRESHOLD" \
    --cluster_n_clusters "$CLUSTER_N_CLUSTERS" \
    --cluster_data_range $CLUSTER_DATA_RANGE \
    --show_max_to_show "$SHOW_MAX_TO_SHOW" \
    --method_name "$METHOD_NAME" \
    --use_error_term "$USE_ERROR_TERM" \
    --pass_through_grad "$PASS_THROUGH_GRAD" \
    --reverse_pruning "$REVERSE_PRUNING" \
    --gradient_mode "$GRADIENT_MODE" \
    --edge_gradient_mode "$EDGE_GRADIENT_MODE" \
    --ig_steps "$IG_STEPS" \
    --transfer_grad "$TRANSFER_GRAD" \
    --save_node_grads "$SAVE_NODE_GRADS" \
    --prune_type "$PRUNE_TYPE" \
    --scoring_mode "$SCORING_MODE" \
    --node_threshold_type "$NODE_THRESHOLD_TYPE" \
    --node_threshold "$NODE_THRESHOLD" \
    --direct_effect "$DIRECT_EFFECT" \
    --n_clusters "$N_CLUSTERS" \
    --alpha "$ALPHA" \
    --lambda_pen "$LAMBDA_PEN" \
    --n_intervals "$N_INTERVALS" \
    --k_per_interval "$K_PER_INTERVAL" \
    --buffer "$BUFFER" \
    --output_dir "$OUTPUT_DIR"
