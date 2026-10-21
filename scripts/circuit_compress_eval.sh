#!/bin/bash
# Section 5.2 (Table 2) and Appendix E (Table 7): circuit compression benchmark.
# Each array task builds the circuits for one slice of the 200 prompts, clusters them with every
# method in CLUSTER_TYPES, and scores each clustering with one LLM-judge eval per circuit (seed 1).
# The judge is Llama-3.3-70B-Instruct served by vLLM on the job's own GPU (JUDGE=llama70b-vllm).
#
# 50 nodes -> 5 supernodes (200 circuits = 8 slices of 25):
#   sbatch --array=0-7 --export=ALL,PRESET=gemma_mlp scripts/circuit_compress_eval.sh
#   sbatch --array=0-7 --export=ALL,PRESET=gpt2_ds scripts/circuit_compress_eval.sh
# 200 nodes -> 20 supernodes (20 slices of 10; LLM Sel does not fit the context window):
#   sbatch --array=0-19 --export=ALL,PRESET=gemma_mlp,NODE_THRESHOLD=200,N_CLUSTERS=20,SLICE=10,CLUSTER_TYPES="ot_global_10 feat_flow_llama mech_permute_llama causal naive" scripts/circuit_compress_eval.sh
# LLM Sel (cluster type `llm`) calls gpt-oss-120b. Either point LLMSEL_URL at an OpenAI-compatible
# endpoint serving it, or set LLMSEL_SELFHOST=true and submit with --gpus=3 to serve it here too.
# Merge the per-slice results with analysis/summarize_compression.py.
#SBATCH --job-name=circuit_compress
#SBATCH --output=logs/circuit_compress_eval_%A_%a.out
#SBATCH --error=logs/circuit_compress_eval_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#
# JUDGE=llama70b-vllm (the default) serves Llama-3.3-70B from vLLM on this node's own GPU, so the
# eval has no API cost or rate limit. It needs two large GPUs (we used 180 GB B200s): the 70B is
# ~132 GB of bf16 weights and wants essentially a whole card, and the circuits need the other one.
# GPU 0 serves the judge, GPU 1 runs the experiment; the python process has GPU 0 hidden from it,
# so it still addresses its card as cuda:0. Set EXP_GPU=0 (and --gpus=1) to squeeze both onto one card -- see the note there.

# sbatch copies this script to the compute node's spool dir, so BASH_SOURCE no longer points
# into the repo. Fall back to the directory sbatch was submitted from (the repo root).
_COMMON="$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[ -f "$_COMMON" ] || _COMMON="${SLURM_SUBMIT_DIR:-.}/scripts/_common.sh"
source "$_COMMON"
mkdir -p logs

# ---- model / SAE -----------------------------------------------------------------------------
# Presets: gemma_mlp | gemma_ds | gpt2_ds | llama_post | qwen_mlp
PRESET="${PRESET:?set PRESET}"

# ---- prompt set ------------------------------------------------------------------------------
LOSS_THRESHOLD="0.1"        # picks loss_below_<threshold>_<model>_sequences.json
# Array task i covers prompts [i*SLICE, (i+1)*SLICE) of the paper's prompt file.
SLICE="${SLICE:-25}"
_START=$(( ${SLURM_ARRAY_TASK_ID:-0} * SLICE ))
DATA_RANGE="$_START $(( _START + SLICE ))"

# ---- activation cache ------------------------------------------------------------------------
DATA_PATH="monology/pile-uncopyrighted"
BATCHSIZE=64
MAX_LENGTH=128
# 10000 sequences for both models (Table 2).
NUM_BATCHES="${NUM_BATCHES:-10000}"
CACHE_DIR="data/cache_data"

# ---- circuit construction / pruning ----------------------------------------------------------
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
NODE_THRESHOLD="${NODE_THRESHOLD:-50}"

CUT_MODE="edge"
THRESHOLD_TYPE="percen"
THRESHOLD=1                 # 1 with percen keeps all edges

DIRECT_EFFECT=false         # false = total effect, which the causal clustering needs

# ---- clustering benchmark --------------------------------------------------------------------
# naive | ot | ot_01 | ot_05 | ot_10 | ot_0
#       | ot_global | ot_global_01 | ot_global_05 | ot_global_10 | ot_global_0
#       | mech_permute_llama | feat_flow_llama | llm | causal
CLUSTER_TYPES="${CLUSTER_TYPES:-ot_global_10 feat_flow_llama mech_permute_llama llm causal naive}"
N_CLUSTERS="${N_CLUSTERS:-5}"

# ---- repeat runs -----------------------------------------------------------------------------
# Leave empty for a single unseeded run under the original file name. Set e.g. SEEDS="1 2 3" to
# run the whole sweep once per seed; each output then gets a _seed<N> suffix.
SEEDS="1"                   # paper protocol: ONE eval per circuit (seeded for reproducibility)

# Fast seed sweep: build the circuit + graph + clustering ONCE per prompt (with CLUSTER_SEED) and
# only re-run the LLM eval per seed, aggregating into one JSON per cluster type (mean/std/90%CI +
# every model decision). Requires SEEDS to be non-empty. Leaves the .pt path untouched when false.
AGGREGATE_SEEDS=false
CLUSTER_SEED=0

# ---- LLM eval --------------------------------------------------------------------------------
# JUDGE: llama70b-vllm          -> self-hosted on this job's GPU (no key, no cost, no rate limit)
#        gpt5-mini | gpt5-nano  -> OpenAI public API, key in OPENAI_API_KEY (cheap; gpt5-mini rec.)
#        llama70b | gpt-oss-120b -> UF gateway, key in UF_AI_API_KEY (the original judges)
# JUDGE_EFFORT (gpt5-* only): minimal | low | medium | high; empty = preset default (low)
# NOTE llama70b-vllm and llama70b are the SAME weights but different judge names, so they write to
# different results dirs (results/<model>_<sae>_llama70b-vllm vs ..._llama70b). Point --save_dir
# at the gateway dir if you want a self-hosted run to extend an existing llama70b sweep.
JUDGE="${JUDGE:-llama70b-vllm}"       # same Llama-3.3-70B weights as the paper's gateway judge
JUDGE_EFFORT=""
LLM_N_CLUSTER_FEATURE=3
LLM_N_RAND_FEATURE=3
LLM_N_INTERVALS=3
LLM_MAX_PARALLEL_REQUESTS=20

# ---- circuit cache ---------------------------------------------------------------------------
# Off by default: the matching fg.save() in circuit_compress_eval.py is still commented out, so
# circuits are always rebuilt. Enable both together to reuse circuits across runs.
USE_CIRCUIT_CACHE=false
CIRCUIT_CACHE_DIR="data/circuit_file"

SUPPRESS_OUTPUT=false

# ---- self-hosted judge (JUDGE=llama70b-vllm) -------------------------------------------------
# vLLM may live in its own environment: point VLLM_BIN at its executable. JUDGE_HF_HOME is the
# HuggingFace cache holding the judge weights (~132 GB); it can differ from the experiment's cache.
# Set JUDGE_OFFLINE=1 if the weights are already downloaded and the node has no internet.
VLLM_BIN="${VLLM_BIN:-$(command -v vllm)}"
JUDGE_HF_HOME="${JUDGE_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
JUDGE_OFFLINE="${JUDGE_OFFLINE:-0}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"   # e.g. --kernel-config '{"enable_flashinfer_autotune": false}'
JUDGE_MODEL="unsloth/Llama-3.3-70B-Instruct"
# Port derived from the job id so array tasks sharing a node do not collide.
JUDGE_PORT="${JUDGE_PORT:-$(( 8100 + ${SLURM_JOB_ID:-0} % 800 ))}"
JUDGE_MAX_LEN=32768         # the cluster prompts carry 9 features x 3 intervals of token windows
# Which of the job's GPUs serves the judge, which runs the experiment, and how much of the judge's
# card vLLM may take (0.92 of a 180 GB B200 = ~165 GB: ~132 GB of weights, the rest KV cache).
#   JUDGE_GPU=0 EXP_GPU=1 (default, needs --gpus=2): a card each.
#   JUDGE_GPU=0 EXP_GPU=0 (with --gpus=1): both share GPU 0 -- also drop GPU_MEM_UTIL to ~0.72 so
#   the circuits get ~50 GB. Fine for 50-node circuits; expect OOM at 200 nodes.
JUDGE_GPU=0
EXP_GPU=1
GPU_MEM_UTIL=0.92

if [[ "$JUDGE" == "llama70b-vllm" ]]; then
    [ -x "$VLLM_BIN" ] || { echo "no vllm at $VLLM_BIN"; exit 1; }

    # JUDGE_GPU / EXP_GPU index into the job's OWN allocation, but CUDA_VISIBLE_DEVICES holds
    # device ids, which are only the same thing when SLURM renumbers them from 0. Resolve through
    # the allocated list so a "2,5" style allocation lands on the right cards either way.
    IFS=',' read -ra _ALLOC_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
    if [[ -z "${_ALLOC_GPUS[$EXP_GPU]:-}" || -z "${_ALLOC_GPUS[$JUDGE_GPU]:-}" ]]; then
        echo "GPU index out of range: allocation is [${_ALLOC_GPUS[*]}] but JUDGE_GPU=$JUDGE_GPU"
        echo "EXP_GPU=$EXP_GPU. Ask for --gpus=2, or set EXP_GPU=$JUDGE_GPU to share one card."
        exit 1
    fi
    JUDGE_DEV="${_ALLOC_GPUS[$JUDGE_GPU]}"
    EXP_DEV="${_ALLOC_GPUS[$EXP_GPU]}"
    echo "[vllm] judge on GPU $JUDGE_DEV, experiment on GPU $EXP_DEV (allocation: ${_ALLOC_GPUS[*]})"

    export VLLM_JUDGE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
    export VLLM_JUDGE_MODEL="$JUDGE_MODEL"
    LOG="logs/vllm_server_${SLURM_JOB_ID:-local}.log"

    # Subshell: the server needs its OWN HF_HOME (the judge weights are in the other repo's cache,
    # the target model google/gemma-2-2b is in this one's -- offline, one HF_HOME cannot serve
    # both), its own CUDA module, and its own GPU. None of that leaks back to the experiment.
    # TMPDIR is node-local and short because vLLM binds a ZMQ IPC socket under it and unix socket
    # paths are capped at 107 chars.
    (
        export HF_HOME="$JUDGE_HF_HOME"
        export HUGGINGFACE_HUB_CACHE="$JUDGE_HF_HOME/hub"
        unset TRANSFORMERS_CACHE
        export HF_HUB_OFFLINE="$JUDGE_OFFLINE"
        export TMPDIR="/tmp/vllm_${SLURM_JOB_ID:-$$}"
        mkdir -p "$TMPDIR"
        export CUDA_VISIBLE_DEVICES="$JUDGE_DEV"
        export PATH="$(dirname "$VLLM_BIN"):$PATH"
                # The flashinfer autotune warmup JIT-compiles kernels and the venv's CUDA pieces are not a
        # coherent toolchain; attention kernels come precompiled, so turn the warmup off.
        exec "$VLLM_BIN" serve "$JUDGE_MODEL" \
            --port "$JUDGE_PORT" \
            --max-model-len "$JUDGE_MAX_LEN" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            $VLLM_EXTRA_ARGS
    ) > "$LOG" 2>&1 &
    SERVER_PID=$!
    trap 'kill $SERVER_PID 2>/dev/null; pkill -P $SERVER_PID 2>/dev/null; true' EXIT

    echo "[vllm] waiting for judge (pid $SERVER_PID, port $JUDGE_PORT), log: $LOG"
    for i in $(seq 1 240); do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "[vllm] server died during startup -- see $LOG"; tail -40 "$LOG"; exit 1
        fi
        curl -sf "$VLLM_JUDGE_URL/models" >/dev/null && break
        sleep 10
    done
    curl -sf "$VLLM_JUDGE_URL/models" >/dev/null || { echo "[vllm] startup timeout"; tail -40 "$LOG"; exit 1; }
    echo "[vllm] judge up after ~$((i * 10))s"

    # ---- optional: self-host the LLM Sel clustering model too (LLMSEL_SELFHOST=true, --gpus=3) ----
    # Serves gpt-oss-120b for LLM Sel on its own card (LLMSEL_GPU indexes the allocation like
    # JUDGE_GPU / EXP_GPU) and points LLMSEL_URL at it.
    if [[ "${LLMSEL_SELFHOST:-false}" == "true" ]]; then
        LLMSEL_GPU="${LLMSEL_GPU:-2}"
        LLMSEL_DEV="${_ALLOC_GPUS[$LLMSEL_GPU]:?LLMSEL_GPU out of range -- submit with --gpus=3}"
        LLMSEL_PORT=$(( JUDGE_PORT + 1000 ))
        LLMSEL_HF_HOME="${LLMSEL_HF_HOME:-$JUDGE_HF_HOME}"
        LLMSEL_LOG="logs/vllm_llmsel_${SLURM_JOB_ID:-local}.log"
        (
            export HF_HOME="$LLMSEL_HF_HOME"
            export HUGGINGFACE_HUB_CACHE="$LLMSEL_HF_HOME/hub"
            unset TRANSFORMERS_CACHE
            export HF_HUB_OFFLINE="$JUDGE_OFFLINE"
            export TMPDIR="/tmp/vllm_sel_${SLURM_JOB_ID:-$$}"
            mkdir -p "$TMPDIR"
            export CUDA_VISIBLE_DEVICES="$LLMSEL_DEV"
            export PATH="$(dirname "$VLLM_BIN"):$PATH"
                        exec "$VLLM_BIN" serve openai/gpt-oss-120b \
                --served-model-name gpt-oss-120b \
                --port "$LLMSEL_PORT" \
                --max-model-len 131072 \
                --gpu-memory-utilization 0.90 \
                $VLLM_EXTRA_ARGS
        ) > "$LLMSEL_LOG" 2>&1 &
        LLMSEL_PID=$!
        trap 'kill $SERVER_PID $LLMSEL_PID 2>/dev/null; pkill -P $SERVER_PID 2>/dev/null; pkill -P $LLMSEL_PID 2>/dev/null; true' EXIT
        export LLMSEL_URL="http://127.0.0.1:${LLMSEL_PORT}/v1"
        for j in $(seq 1 240); do
            kill -0 $LLMSEL_PID 2>/dev/null || { echo "[vllm] LLM Sel server died -- see $LLMSEL_LOG"; tail -40 "$LLMSEL_LOG"; exit 1; }
            curl -sf "$LLMSEL_URL/models" >/dev/null && break
            sleep 10
        done
        curl -sf "$LLMSEL_URL/models" >/dev/null || { echo "[vllm] LLM Sel startup timeout"; tail -40 "$LLMSEL_LOG"; exit 1; }
        echo "[vllm] LLM Sel gpt-oss-120b up on GPU $LLMSEL_DEV after ~$((j * 10))s ($LLMSEL_URL)"
    fi

    # Hide every other card from the experiment, so --device_id 0 is unambiguously EXP_GPU and no
    # library can wander onto the judge's memory by defaulting to cuda:0.
    export CUDA_VISIBLE_DEVICES="$EXP_DEV"
    DEVICE_ID=0
fi

EXTRA_ARGS=()
if [[ -n "$SEEDS" ]]; then
    EXTRA_ARGS+=(--seeds $SEEDS)
fi
EXTRA_ARGS+=(--judge "$JUDGE")
if [[ -n "${SAVE_DIR:-}" ]]; then
    EXTRA_ARGS+=(--save_dir "$SAVE_DIR")   # run file names carry no num_batches, so separate dirs
fi
if [[ -n "$JUDGE_EFFORT" ]]; then
    EXTRA_ARGS+=(--judge_effort "$JUDGE_EFFORT")
fi
EXTRA_ARGS+=(--aggregate_seeds "$AGGREGATE_SEEDS" --cluster_seed "$CLUSTER_SEED")

python circuit_compress_eval.py \
    --preset "$PRESET" \
    --device_id "$DEVICE_ID" \
    --loss_threshold "$LOSS_THRESHOLD" \
    --data_range $DATA_RANGE \
    --data_path "$DATA_PATH" \
    --batchsize "$BATCHSIZE" \
    --max_length "$MAX_LENGTH" \
    --num_batches "$NUM_BATCHES" \
    --cache_dir "$CACHE_DIR" \
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
    --cluster_types $CLUSTER_TYPES \
    --n_clusters "$N_CLUSTERS" \
    --llm_n_cluster_feature "$LLM_N_CLUSTER_FEATURE" \
    --llm_n_rand_feature "$LLM_N_RAND_FEATURE" \
    --llm_n_intervals "$LLM_N_INTERVALS" \
    --llm_max_parallel_requests "$LLM_MAX_PARALLEL_REQUESTS" \
    --use_circuit_cache "$USE_CIRCUIT_CACHE" \
    --circuit_cache_dir "$CIRCUIT_CACHE_DIR" \
    --suppress_output "$SUPPRESS_OUTPUT" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
