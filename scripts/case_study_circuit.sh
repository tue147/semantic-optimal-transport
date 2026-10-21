#!/bin/bash
#SBATCH --job-name=case_study
#SBATCH --output=logs/case_study_%j.out
#SBATCH --error=logs/case_study_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#
# Section 6.1: automated circuit compression of the Dallas / Rhythm / Opposite circuits (Table 5).
# Writes case_studies/<circuit>_circuit.html and case_studies/<circuit>_compressed.html.
# Submit:  sbatch scripts/case_study_circuit.sh        (CIRCUITS="dallas" for one circuit)

_COMMON="$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[ -f "$_COMMON" ] || _COMMON="${SLURM_SUBMIT_DIR:-.}/scripts/_common.sh"
source "$_COMMON"
mkdir -p logs

CIRCUITS="${CIRCUITS:-dallas rhythm opposite}"

for circuit in $CIRCUITS; do
    python case_study_circuit.py --preset gemma_mlp --device_id "$DEVICE_ID" --circuit "$circuit"
done
