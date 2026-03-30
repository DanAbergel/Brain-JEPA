"""
Prepare ADNI 450 ROI data for Brain-JEPA fine-tuning.
=====================================================

Takes the parcellated ADNI tensor + labels and creates train/valid/test
splits in the format expected by Brain-JEPA downstream evaluation.

Split is done at the SUBJECT level (not image level) to prevent data
leakage when a subject has multiple scans. The same subject-level split
is reused across all horizons for fair comparison.

Input:
    adni_parcellated_450.pt         (812, 450, 140)
    index_to_name.json              index → image_id mapping
    imageID_to_labels.json          image_id → labels

Output (for each degradation horizon):
    adni_degradation_1y_{train,valid,test}_x.pt   (N, 450, 140)
    adni_degradation_1y_{train,valid,test}_y.pt    (N,)
    adni_degradation_2y_...
    adni_degradation_3y_...

Usage:
    python3 scripts/adni/prepare_adni_splits.py
"""

import json
import math
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================
LAB_DIR = Path("/sci/labs/arieljaffe/dan.abergel1")
SHARED_DIR = Path("/sci/nosnap/arieljaffe/sagi.nathan/shared_fmri_data")
PARCELLATED_PATH = LAB_DIR / "data" / "adni_parcellated_450.pt"
INDEX_TO_NAME_PATH = SHARED_DIR / "index_to_name.json"
LABELS_PATH = SHARED_DIR / "imageID_to_labels.json"
OUTPUT_DIR = LAB_DIR / "data" / "adni_processed"

TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

HORIZONS = {
    "1y": "degradation_binary_1year",
    "2y": "degradation_binary_2years",
    "3y": "degradation_binary_3years",
}


def main():
    print("=" * 70)
    print("PREPARE ADNI SPLITS FOR BRAIN-JEPA FINE-TUNING")
    print("=" * 70)

    # 1. Load parcellated data
    print(f"\nLoading {PARCELLATED_PATH} ...")
    data = torch.load(PARCELLATED_PATH, weights_only=True)
    N = data.shape[0]
    print(f"  Shape: {data.shape}")

    # 2. Load mappings
    print("Loading index_to_name ...")
    with open(INDEX_TO_NAME_PATH) as f:
        index_to_name = json.load(f)

    print("Loading imageID_to_labels ...")
    with open(LABELS_PATH) as f:
        image_labels = json.load(f)

    # 3. Group indices by subject_id to prevent leakage
    subject_to_indices = defaultdict(list)
    for idx in range(N):
        info = index_to_name[str(idx)]
        subject_id = info["subject_id"]
        subject_to_indices[subject_id].append(idx)

    subjects = sorted(subject_to_indices.keys())
    print(f"\n  Total images: {N}")
    print(f"  Unique subjects: {len(subjects)}")
    multi = sum(1 for s in subjects if len(subject_to_indices[s]) > 1)
    if multi > 0:
        print(f"  Subjects with multiple scans: {multi}")

    # 4. Split at subject level (ONCE, shared across all horizons)
    rng = np.random.RandomState(SEED)
    subjects_arr = np.array(subjects)
    rng.shuffle(subjects_arr)

    n_train = int(len(subjects_arr) * TRAIN_RATIO)
    n_valid = int(len(subjects_arr) * (TRAIN_RATIO + VALID_RATIO))

    train_subjects = set(subjects_arr[:n_train])
    valid_subjects = set(subjects_arr[n_train:n_valid])
    test_subjects = set(subjects_arr[n_valid:])

    print(f"\n  Subject split: train={len(train_subjects)}, valid={len(valid_subjects)}, test={len(test_subjects)}")

    # 5. For each horizon, collect valid samples and save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for horizon_name, label_key in HORIZONS.items():
        print(f"\n{'─' * 70}")
        print(f"HORIZON: {horizon_name} ({label_key})")
        print(f"{'─' * 70}")

        # Map each index to its split and label
        split_indices = {"train": [], "valid": [], "test": []}
        split_labels = {"train": [], "valid": [], "test": []}
        skipped = 0

        for idx in range(N):
            info = index_to_name[str(idx)]
            subject_id = info["subject_id"]
            image_id = info["image_id"]

            # Get label
            if image_id not in image_labels:
                skipped += 1
                continue
            label_val = image_labels[image_id].get(label_key)
            if label_val is None or (isinstance(label_val, float) and math.isnan(label_val)):
                skipped += 1
                continue

            # Assign to split based on subject
            if subject_id in train_subjects:
                split = "train"
            elif subject_id in valid_subjects:
                split = "valid"
            else:
                split = "test"

            split_indices[split].append(idx)
            split_labels[split].append(int(label_val))

        print(f"  Valid samples: {N - skipped}/{N} (skipped {skipped} with missing labels)")

        prefix = f"adni_degradation_{horizon_name}"

        for split_name in ["train", "valid", "test"]:
            indices = np.array(split_indices[split_name])
            y = torch.tensor(split_labels[split_name], dtype=torch.long)

            # Shuffle within split
            perm = np.random.RandomState(SEED).permutation(len(indices))
            indices = indices[perm]
            y = y[perm]

            x = data[indices]

            x_path = OUTPUT_DIR / f"{prefix}_{split_name}_x.pt"
            y_path = OUTPUT_DIR / f"{prefix}_{split_name}_y.pt"

            torch.save(x, x_path)
            torch.save(y, y_path)

            n0 = (y == 0).sum().item()
            n1 = (y == 1).sum().item()
            print(f"  {split_name}: {len(y)} samples (0={n0}, 1={n1}) → {x_path.name}")

    # 6. Verify no leakage
    print(f"\n{'─' * 70}")
    print("LEAKAGE CHECK")
    print(f"{'─' * 70}")
    overlap_tv = train_subjects & valid_subjects
    overlap_tt = train_subjects & test_subjects
    overlap_vt = valid_subjects & test_subjects
    if overlap_tv or overlap_tt or overlap_vt:
        print(f"  WARNING: Subject overlap detected!")
        print(f"    train ∩ valid: {len(overlap_tv)}")
        print(f"    train ∩ test:  {len(overlap_tt)}")
        print(f"    valid ∩ test:  {len(overlap_vt)}")
    else:
        print("  No subject overlap between splits. No data leakage.")

    print(f"\nAll splits saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
