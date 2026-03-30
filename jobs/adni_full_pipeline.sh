#!/bin/bash
# =====================================================================
# SLURM Job — Full ADNI pipeline: parcellate 450 ROIs + prepare splits
# =====================================================================
#
# HOW TO USE:
#   sbatch jobs/adni_full_pipeline.sh
#
# Step 1: Parcellate ADNI → (812, 450, 140) tensor
# Step 2: Prepare train/valid/test splits with degradation labels
#
# Output: /sci/labs/arieljaffe/dan.abergel1/data/adni_processed/
#         adni_degradation_{1y,2y,3y}_{train,valid,test}_{x,y}.pt
# =====================================================================

#SBATCH --job-name=adni-pipeline
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/adni_pipeline_%j.out
#SBATCH --error=logs/adni_pipeline_%j.err

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
PROJECT_DIR="$LAB_DIR/repos/Brain-JEPA"
VENV_DIR="$LAB_DIR/torch_env"

mkdir -p "$PROJECT_DIR/logs"

echo "=== ADNI Full Pipeline — Job $SLURM_JOB_ID on $(hostname) ==="

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

pip install --quiet nilearn nibabel

echo ""
echo "=== Step 1: Parcellate ADNI 450 ROIs ==="
python3 -u scripts/adni/parcellate_adni_450.py

echo ""
echo "=== Step 2: Prepare train/valid/test splits ==="
python3 -u scripts/adni/prepare_adni_splits.py

echo ""
echo "=== Pipeline complete: $(date) ==="
