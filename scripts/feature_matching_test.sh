#!/bin/bash
#SBATCH --job-name=feat_match_test
#SBATCH --output=logs/feature_matching_test_%j.out
#SBATCH --error=logs/feature_matching_test_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#
# Table 1 / Table 6: evaluate cross-layer feature matchings (LLM Eval / Acc with the gpt-oss-120b
# judge, CE / VE by layer skipping). Run feature_matching_run.sh for the pair first.
# The paper judges the same 100 query features 5 times: run seed 1 alone to sample them, then
# seeds 2-5 reusing the seed-1 file:
#   sbatch --export=ALL,SAE_A_ID=13,SAE_B_ID=12,SEEDS=1 scripts/feature_matching_test.sh
#   sbatch --export=ALL,SAE_A_ID=13,SAE_B_ID=12,SEEDS="2 3 4 5",EXTRACT_FROM_PREV=true scripts/feature_matching_test.sh
# GPT2: add PRESET=gpt2_ds,NUM_BATCHES=1000.
# WARNING: fans out paid LLM-judge calls, once per seed.

# sbatch copies this script to the compute node's spool dir, so BASH_SOURCE no longer points
# into the repo. Fall back to the directory sbatch was submitted from (the repo root).
_COMMON="$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[ -f "$_COMMON" ] || _COMMON="${SLURM_SUBMIT_DIR:-.}/scripts/_common.sh"
source "$_COMMON"
mkdir -p logs

# ---- model / SAE -----------------------------------------------------------------------------
# Presets: gemma_mlp | gpt2_ds
PRESET="${PRESET:-gemma_mlp}"

# ---- token extraction ------------------------------------------------------------------------
DATA_PATH="monology/pile-uncopyrighted"
MAX_LENGTH=128
NUM_BATCHES="${NUM_BATCHES:-10000}"   # 10000 sequences for Gemma-2-2B, 1000 for GPT2

# ---- layer pair and methods ------------------------------------------------------------------
# Pairs defined in exp_config.LAYER_PAIR_PRESETS: (1,0) (6,5) (11,0) (11,5) (13,12) (25,0) (25,12)
# METHODS must be a subset of the methods listed for that pair, e.g.
#   METHODS="ot feat_flow mech_permute naive feat_flow_ml mech_permute_ml causal"
SAE_A_ID="${SAE_A_ID:-13}"
SAE_B_ID="${SAE_B_ID:-12}"
METHODS="${METHODS:-ot feat_flow mech_permute naive feat_flow_ml mech_permute_ml causal}"

# ---- evaluation ------------------------------------------------------------------------------
BATCH_SIZE=96
MIN_ACTIVATION_EXAMPLES=30
NUMBER_OF_PAIRS=100
TOP=1

# ---- repeat runs -----------------------------------------------------------------------------
# One full evaluation per seed; the seed becomes the runN suffix in the output file name.
# e.g. SEEDS="1 2 3 4" writes ..._run1.pt ... ..._run4.pt
SEEDS="${SEEDS:-1 2 3 4 5}"
RUN_PREFIX="run"

# ---- feature selection -----------------------------------------------------------------------
# true reuses the feature set recorded in PREV_FILE (looked up in results/<model>_<sae>/, then the
# judge dir); false samples fresh keys (seed-dependent).
EXTRACT_FROM_PREV="${EXTRACT_FROM_PREV:-false}"
PREV_FILE="${PREV_FILE:-ot_${SAE_A_ID}_${SAE_B_ID}_1_100_30_run1.pt}"

# ---- causal (attribution) baseline -----------------------------------------------------------
# causal is computed on the fly over the sampled feature set (list "causal" in METHODS above).
CAUSAL_GRAD_ONLY=true
CAUSAL_USE_ERROR_TERM=true
CAUSAL_DIRECT_EFFECT=true

# ---- LLM judge -------------------------------------------------------------------------------
# gpt5-mini | gpt5-nano  -> OpenAI public API, key in OPENAI_API_KEY (cheap; gpt5-mini recommended)
# llama70b | gpt-oss-120b -> OpenAI-compatible endpoint JUDGE_BASE_URL, key in JUDGE_API_KEY (the paper's judges)
# JUDGE_EFFORT (gpt5-* only): minimal | low | medium | high; empty = preset default (low)
JUDGE="${JUDGE:-gpt-oss-120b}"
JUDGE_EFFORT=""

# ---- output ----------------------------------------------------------------------------------
# Every run also updates the per-pair details log next to the .pt files,
# results/<model>_<sae>_<judge>/matching_{a}_{b}_{top}_{npairs}_{minact}_seedagg.json: per-seed
# LLM score / accuracy / CE / VE plus mean, std and 90% CI across seeds. CE/VE are always in the
# JSON; SAVE_SKIPPING only adds them to the .pt as well.
# To rebuild that JSON from existing .pt files (login node, no GPU, no API calls):
#   python feature_matching_test.py --backfill true --judge gpt5-mini --sae_a_id 25 --sae_b_id 12 \
#       --backfill_logs "logs/feature_matching_test_*.out"
SAVE_SKIPPING=true

JUDGE_ARGS=(--judge "$JUDGE")
if [[ -n "$JUDGE_EFFORT" ]]; then
    JUDGE_ARGS+=(--judge_effort "$JUDGE_EFFORT")
fi

python feature_matching_test.py \
    --preset "$PRESET" \
    --device_id "$DEVICE_ID" \
    --data_path "$DATA_PATH" \
    --max_length "$MAX_LENGTH" \
    --num_batches "$NUM_BATCHES" \
    --sae_a_id "$SAE_A_ID" \
    --sae_b_id "$SAE_B_ID" \
    --methods $METHODS \
    --batch_size "$BATCH_SIZE" \
    --min_activation_examples "$MIN_ACTIVATION_EXAMPLES" \
    --number_of_pairs "$NUMBER_OF_PAIRS" \
    --top "$TOP" \
    --seeds $SEEDS \
    --run_prefix "$RUN_PREFIX" \
    --extract_from_prev "$EXTRACT_FROM_PREV" \
    --prev_file "$PREV_FILE" \
    --causal_grad_only "$CAUSAL_GRAD_ONLY" \
    --causal_use_error_term "$CAUSAL_USE_ERROR_TERM" \
    --causal_direct_effect "$CAUSAL_DIRECT_EFFECT" \
    --save_skipping "$SAVE_SKIPPING" \
    "${JUDGE_ARGS[@]}"
