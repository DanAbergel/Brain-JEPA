#!/bin/bash
# =====================================================================
# SLURM Job — Fine-tune Brain-JEPA on ADNI degradation (all 3 horizons)
# =====================================================================
#
# HOW TO USE:
#   sbatch jobs/finetune_adni.sh
#
# Requires GPU. Runs fine-tuning for 1y, 2y, 3y degradation.
# Pretrained checkpoint must be at:
#   /sci/labs/arieljaffe/dan.abergel1/repos/Brain-JEPA/logs/pretrained/
# =====================================================================

#SBATCH --job-name=adni-ft
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune_adni_%j.out
#SBATCH --error=logs/finetune_adni_%j.err

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
PROJECT_DIR="$LAB_DIR/repos/Brain-JEPA"
VENV_DIR="$LAB_DIR/torch_env"
CKPT="$PROJECT_DIR/logs/pretrained/BrainJEPA-Checkpoints/Pretraining/jepa-ep300.pth.tar"

mkdir -p "$PROJECT_DIR/logs"

echo "=== ADNI Fine-tuning — Job $SLURM_JOB_ID on $(hostname) ==="

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

for HORIZON in 1y 2y 3y; do
    echo ""
    echo "=========================================="
    echo "  Fine-tuning: degradation ${HORIZON}"
    echo "=========================================="
    bash scripts/classification/run_downstream_FT_adni_degradation.sh ${HORIZON} ${CKPT}
done

echo ""
echo "=== All fine-tuning complete: $(date) ==="
echo ""
python3 -u scripts/adni/summarize_results.py ./output_dir
