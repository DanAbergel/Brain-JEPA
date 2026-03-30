"""
Parcellate ADNI 4D volumes with Tian S3 (50 subcortical) + Schaefer 400 (cortical).

Input:  all_4d_downsampled.pt  (812, 45, 54, 45, 140)
Output: adni_parcellated_450.pt  (812, 450, 140) — ROIs x Timepoints
        Order: 50 subcortical (Tian S3) + 400 cortical (Schaefer 400)
        Same ROI ordering as Brain-JEPA expects.

Usage:
    python3 scripts/adni/parcellate_adni_450.py
"""

import os
import urllib.request
import torch
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.image import resample_img
from pathlib import Path

DATA_ROOT = Path("/sci/nosnap/arieljaffe/sagi.nathan/shared_fmri_data")
LAB_DIR = Path("/sci/labs/arieljaffe/dan.abergel1")
OUTPUT_PATH = LAB_DIR / "data" / "adni_parcellated_450.pt"
NILEARN_DATA_DIR = str(LAB_DIR / "cache" / "nilearn_data")

os.environ["NILEARN_DATA"] = NILEARN_DATA_DIR

TIAN_S3_URL = "https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S3_3T_1mm.nii.gz"


def fetch_tian_s3():
    """Download Tian S3 atlas if not already cached."""
    cache_dir = os.path.join(NILEARN_DATA_DIR, "tian_2020")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, "Tian_Subcortex_S3_3T_1mm.nii.gz")
    if not os.path.exists(local_path):
        print("  Downloading Tian S3 atlas from GitHub...")
        urllib.request.urlretrieve(TIAN_S3_URL, local_path)
    return local_path


def fetch_and_resample(atlas_img, data_affine, target_shape):
    """Resample an atlas to the data grid."""
    resampled = resample_img(
        atlas_img,
        target_affine=data_affine,
        target_shape=target_shape,
        interpolation="nearest",
    )
    return resampled.get_fdata().astype(int)


def main():
    # 1. Load 4D volumes
    pt_path = DATA_ROOT / "all_4d_downsampled.pt"
    print(f"Loading {pt_path} ...")
    data = torch.load(pt_path, weights_only=True)
    N, X, Y, Z, T = data.shape
    print(f"  Shape: ({N}, {X}, {Y}, {Z}, {T})")

    # MNI152 4mm affine
    data_affine = np.array([
        [-4.,  0.,  0.,  90.],
        [ 0.,  4.,  0., -126.],
        [ 0.,  0.,  4.,  -72.],
        [ 0.,  0.,  0.,    1.]
    ])

    # 2. Fetch Tian S3 subcortical atlas (50 ROIs)
    print("Fetching Tian S3 subcortical atlas (50 ROIs) ...")
    tian_path = fetch_tian_s3()
    tian_img = nib.load(tian_path)
    tian_labels = fetch_and_resample(tian_img, data_affine, (X, Y, Z))

    tian_regions = np.unique(tian_labels)
    tian_regions = tian_regions[tian_regions > 0]
    print(f"  Tian S3: {len(tian_regions)}/50 regions found")

    # 3. Fetch Schaefer 400 cortical atlas
    print("Fetching Schaefer 400 cortical atlas ...")
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    schaefer_img = nib.load(schaefer.maps)
    schaefer_labels = fetch_and_resample(schaefer_img, data_affine, (X, Y, Z))

    schaefer_regions = np.unique(schaefer_labels)
    schaefer_regions = schaefer_regions[schaefer_regions > 0]
    print(f"  Schaefer 400: {len(schaefer_regions)}/400 regions found")

    if len(tian_regions) < 45:
        print("  WARNING: many Tian regions missing!")
    if len(schaefer_regions) < 380:
        print("  WARNING: many Schaefer regions missing!")

    # 4. Precompute masks — subcortical first, then cortical
    masks = []

    # Tian S3: labels 1..50
    tian_order = sorted(tian_regions)
    for r in tian_order:
        mask = tian_labels == r
        if mask.sum() > 0:
            masks.append(torch.from_numpy(mask))
    n_subcortical = len(masks)
    print(f"  Subcortical masks: {n_subcortical}")

    # Schaefer 400: labels 1..400
    schaefer_order = sorted(schaefer_regions)
    for r in schaefer_order:
        mask = schaefer_labels == r
        if mask.sum() > 0:
            masks.append(torch.from_numpy(mask))
    n_cortical = len(masks) - n_subcortical
    n_total = len(masks)
    print(f"  Cortical masks: {n_cortical}")
    print(f"  Total ROIs: {n_total}")

    # 5. Parcellate
    print(f"Parcellating {N} subjects ...")
    result = torch.zeros(N, n_total, T, dtype=torch.float32)

    for i in range(N):
        if i % 100 == 0:
            print(f"  {i}/{N}")
        vol = data[i].float()  # (X, Y, Z, T)
        for j, mask in enumerate(masks):
            voxels = vol[mask]  # (n_voxels, T)
            result[i, j, :] = voxels.mean(dim=0)

    # 6. Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Shape: {result.shape}  (N={N}, ROIs={n_total} [{n_subcortical} sub + {n_cortical} cort], T={T})")


if __name__ == "__main__":
    main()
