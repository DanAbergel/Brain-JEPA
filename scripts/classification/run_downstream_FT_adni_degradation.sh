#!/bin/bash
# =====================================================================
# Fine-tune Brain-JEPA on ADNI cognitive degradation prediction
# =====================================================================
# Usage:
#   bash scripts/classification/run_downstream_FT_adni_degradation.sh <horizon> <checkpoint_path>
#
# Examples:
#   bash scripts/classification/run_downstream_FT_adni_degradation.sh 1y logs/pretrained/jepa-ep300.pth.tar
#   bash scripts/classification/run_downstream_FT_adni_degradation.sh 2y logs/pretrained/jepa-ep300.pth.tar
#   bash scripts/classification/run_downstream_FT_adni_degradation.sh 3y logs/pretrained/jepa-ep300.pth.tar
# =====================================================================

HORIZON=${1:-1y}
LOAD_PATH=${2:-logs/pretrained/jepa-ep300.pth.tar}

echo "=== Fine-tuning Brain-JEPA on ADNI degradation ${HORIZON} ==="
echo "Checkpoint: ${LOAD_PATH}"

python downstream_eval.py \
    --downstream_task fine_tune \
    --task classification \
    --batch_size 4 \
    --nb_classes 2 \
    --num_seed 5 \
    --epochs 50 \
    --blr 0.001 \
    --min_lr 0.000001 \
    --smoothing 0.0 \
    --config configs/downstream/fine_tune.yaml \
    --output_root "./output_dir/degradation_${HORIZON}" \
    --model_name vit_base \
    --data_make_fn adni_degradation \
    --processed_dir /sci/labs/arieljaffe/dan.abergel1/data/adni_processed \
    --horizon ${HORIZON} \
    --load_path ${LOAD_PATH} \
    --use_normalization \
    --crop_size 450,160 \
    --patch_size 16 \
    --downsample \
    --pred_depth 12 \
    --pred_emb_dim 384 \
    --attn_mode normal \
    --add_w mapping \
    --gradient_checkpointing
