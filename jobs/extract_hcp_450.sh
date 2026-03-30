#!/bin/bash
# =====================================================================
# SLURM Job — Extract 450 ROI time series for HCP subjects
# =====================================================================
#
# HOW TO USE:
#   sbatch jobs/extract_hcp_450.sh
#
# CPU only, ~32 GB RAM.
# Output: HCP_data/subject_XXXXX/.../rfMRI_REST1_LR_450.npy per subject
# =====================================================================

#SBATCH --job-name=hcp-450
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/extract_hcp_450_%j.out
#SBATCH --error=logs/extract_hcp_450_%j.err

set -euo pipefail

LAB_DIR="/sci/labs/arieljaffe/dan.abergel1"
PROJECT_DIR="$LAB_DIR/repos/Brain-JEPA"
VENV_DIR="$LAB_DIR/torch_env"

mkdir -p "$PROJECT_DIR/logs"

echo "Extract HCP 450 ROIs — Job $SLURM_JOB_ID on $(hostname)"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

pip install --quiet nilearn nibabel

python3 -u scripts/hcp/extract_hcp_450.py

echo "Done: $(date)"
