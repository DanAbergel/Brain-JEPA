#!/bin/bash
# =====================================================================
# SLURM Job — Parcellate ADNI with Tian S3 (50) + Schaefer 400 = 450 ROIs
# =====================================================================
#
# HOW TO USE:
#   sbatch jobs/parcellate_adni_450.sh
#
# CPU only, ~100 GB RAM.
# Output: /sci/labs/arieljaffe/dan.abergel1/data/adni_parcellated_450.pt
# =====================================================================

#SBATCH --job-name=parcellate-450
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/parcellate_450_%j.out
#SBATCH --error=logs/parcellate_450_%j.err

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
PROJECT_DIR="$LAB_DIR/repos/Brain-JEPA"
VENV_DIR="$LAB_DIR/torch_env"

mkdir -p "$PROJECT_DIR/logs"

echo "Parcellate ADNI 450 ROIs — Job $SLURM_JOB_ID on $(hostname)"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

pip install --quiet nilearn

python3 -u scripts/adni/parcellate_adni_450.py

echo "Done: $(date)"
