#!/bin/bash
# =====================================================================
# Launch 3 fine-tuning jobs in parallel (1y, 2y, 3y) + summary job
# =====================================================================
#
# HOW TO USE:
#   bash jobs/finetune_adni.sh
#
# This is NOT a SLURM job itself — it submits 4 jobs:
#   3 parallel fine-tuning jobs + 1 summary job that waits for all 3
# =====================================================================

set -eo pipefail

PROJECT_DIR="/sci/labs/arieljaffe/dan.abergel1/repos/Brain-JEPA"
cd "$PROJECT_DIR"

JIDS=""
for HORIZON in 1y 2y 3y; do
    JID=$(sbatch --parsable jobs/finetune_adni_single.sh ${HORIZON})
    echo "Submitted degradation_${HORIZON} — Job ${JID}"
    JIDS="${JIDS}:${JID}"
done

# Submit summary job that waits for all 3
SUMMARY_JID=$(sbatch --parsable --dependency=afterok${JIDS} jobs/finetune_adni_summary.sh)
echo "Submitted summary — Job ${SUMMARY_JID} (waits for${JIDS})"
echo ""
echo "Monitor: squeue -u \$USER"
