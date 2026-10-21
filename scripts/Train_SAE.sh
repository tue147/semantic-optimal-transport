#!/bin/bash
#SBATCH --job-name=train_sae
#SBATCH --output=logs/train_sae_%j.out
#SBATCH --error=logs/train_sae_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#
# Appendix D.4: TopK SAEs on the GPT2 residual stream (hook_resid_pre of every layer), trained on
# the Pile for 500M tokens, lr 1e-4, auxiliary-loss coefficient 0.03125, Adam.
# Submit:  sbatch scripts/Train_SAE.sh     (LAYERS="0 1 2" to train a subset)

_COMMON="$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[ -f "$_COMMON" ] || _COMMON="${SLURM_SUBMIT_DIR:-.}/scripts/_common.sh"
source "$_COMMON"
mkdir -p logs

model_name="gpt2-small"
tok_name="gpt2"
batch_size=32
max_tokens=500000000
sae_name="topk_custom"
lr=0.0001
sparsity_coef=0.1
expansion_factor=32
k=32
auxk=256
auxk_coef=0.03125
max_epochs=1
dead_tokens_threshold=10000000
log_every_n_steps=50
save_every_n_training_steps=1000
save_step=false
use_loss_var=true
num_workers=16
LAYERS="${LAYERS:-0 1 2 3 4 5 6 7 8 9 10 11}"

for layer in $LAYERS
do
    hook_name="blocks.${layer}.hook_resid_pre"

    python Train_SAE.py \
        --model_name "$model_name" \
        --tok_name "$tok_name" \
        --layer "$layer" \
        --batch_size "$batch_size" \
        --hook_name "$hook_name" \
        --sae_name "$sae_name" \
        --lr "$lr" \
        --sparsity_coef "$sparsity_coef" \
        --expansion_factor "$expansion_factor" \
        --k "$k" \
        --auxk "$auxk" \
        --device_id "$DEVICE_ID" \
        --max_epochs "$max_epochs" \
        --dead_tokens_threshold "$dead_tokens_threshold" \
        --log_every_n_steps "$log_every_n_steps" \
        --save_every_n_training_steps "$save_every_n_training_steps" \
        --save_step "$save_step" \
        --use_loss_var "$use_loss_var" \
        --max_tokens "$max_tokens" \
        --auxk_coef "$auxk_coef" \
        --num_workers "$num_workers"
done
