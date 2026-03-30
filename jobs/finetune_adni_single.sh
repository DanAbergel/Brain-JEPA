#!/bin/bash
# =====================================================================
# SLURM Job — Fine-tune Brain-JEPA on a single ADNI degradation horizon
# =====================================================================
# Usage: sbatch jobs/finetune_adni_single.sh <horizon>
# =====================================================================

#SBATCH --job-name=adni-ft
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/finetune_adni_%x_%j.out
#SBATCH --error=logs/finetune_adni_%x_%j.err

set -euo pipefail

HORIZON=${1:-1y}

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
PROJECT_DIR="$LAB_DIR/repos/Brain-JEPA"
VENV_DIR="$LAB_DIR/torch_env"
CKPT="$PROJECT_DIR/logs/pretrained/BrainJEPA-Checkpoints/Pretraining/jepa-ep300.pth.tar"

mkdir -p "$PROJECT_DIR/logs"

echo "=== Fine-tuning degradation_${HORIZON} — Job ${SLURM_JOB_ID:-local} on $(hostname) ==="

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

bash scripts/classification/run_downstream_FT_adni_degradation.sh ${HORIZON} ${CKPT}

echo "=== Done degradation_${HORIZON}: $(date) ==="
