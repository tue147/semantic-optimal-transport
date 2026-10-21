#!/bin/bash
#SBATCH --job-name=behavior_cluster
#SBATCH --output=logs/circuit_behavior_clustering_%j.out
#SBATCH --error=logs/circuit_behavior_clustering_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#
# Cluster prompts by last-token hidden state, then score each cluster by circuit-feature entropy.
# The entropy stage rebuilds a circuit for every prompt in every cluster - budget accordingly, or
# set CLUSTER_IDS / SKIP_ENTROPY.
# Submit:  sbatch scripts/circuit_behavior_clustering.sh

# sbatch copies this script to the compute node's spool dir, so BASH_SOURCE no longer points
# into the repo. Fall back to the directory sbatch was submitted from (the repo root).
_COMMON="$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[ -f "$_COMMON" ] || _COMMON="${SLURM_SUBMIT_DIR:-.}/scripts/_common.sh"
source "$_COMMON"
mkdir -p logs

# ---- model / SAE -----------------------------------------------------------------------------
# Presets: gemma_mlp | gpt2_ds
PRESET="gemma_mlp"

# ---- prompt set / hidden states --------------------------------------------------------------
LOSS_THRESHOLD="0.1"                        # picks loss_below_<threshold>_<model>_sequences.json
HOOK_TEMPLATE="blocks.{layer}.hook_resid_post"
POSITION=-1                                 # token position cached (-1 = last)
DATA_RANGE="0 18000"                        # START END

# ---- kmeans ----------------------------------------------------------------------------------
N_CLUSTERS=500
KMEANS_BATCH_SIZE=1000
RANDOM_STATE=42
KMEANS_N_INIT="auto"

# ---- inspection ------------------------------------------------------------------------------
SHOW_CLUSTER_ID=9                           # -1 disables the print
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
NODE_THRESHOLD=50

CUT_MODE="edge"
THRESHOLD_TYPE="percen"
THRESHOLD=1

DIRECT_EFFECT=false

# ---- entropy stage ---------------------------------------------------------------------------
# Leave CLUSTER_IDS empty to score every cluster.
CLUSTER_IDS=""
SKIP_ENTROPY=false

OUTPUT_DIR="."

EXTRA_ARGS=()
if [[ -n "$CLUSTER_IDS" ]]; then
    EXTRA_ARGS+=(--cluster_ids $CLUSTER_IDS)
fi

python circuit_behavior_clustering.py \
    --preset "$PRESET" \
    --device_id "$DEVICE_ID" \
    --loss_threshold "$LOSS_THRESHOLD" \
    --hook_template "$HOOK_TEMPLATE" \
    --position "$POSITION" \
    --data_range $DATA_RANGE \
    --n_clusters "$N_CLUSTERS" \
    --kmeans_batch_size "$KMEANS_BATCH_SIZE" \
    --random_state "$RANDOM_STATE" \
    --kmeans_n_init "$KMEANS_N_INIT" \
    --show_cluster_id "$SHOW_CLUSTER_ID" \
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
    --cut_mode "$CUT_MODE" \
    --threshold_type "$THRESHOLD_TYPE" \
    --threshold "$THRESHOLD" \
    --direct_effect "$DIRECT_EFFECT" \
    --skip_entropy "$SKIP_ENTROPY" \
    --output_dir "$OUTPUT_DIR" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
