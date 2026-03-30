#!/bin/bash
# =====================================================================
# SLURM Job — Summarize results after all 3 fine-tuning jobs complete
# =====================================================================

#SBATCH --job-name=adni-summary
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/finetune_adni_summary_%j.out
#SBATCH --error=logs/finetune_adni_summary_%j.err

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
PROJECT_DIR="$LAB_DIR/repos/Brain-JEPA"

source "$LAB_DIR/torch_env/bin/activate"
cd "$PROJECT_DIR"

python3 -u scripts/adni/summarize_results.py ./output_dir
