"""
Extract 450 ROI time series for HCP subjects (Tian S3 50 + Schaefer 400)
=========================================================================

Adds rfMRI_REST1_LR_450.npy alongside existing Schaefer files:
    HCP_data/subject_XXXXX/.../rfMRI_REST1_LR_schaefer100.npy  (existing)
    HCP_data/subject_XXXXX/.../rfMRI_REST1_LR_schaefer200.npy  (existing)
    HCP_data/subject_XXXXX/.../rfMRI_REST1_LR_schaefer400.npy  (existing)
    HCP_data/subject_XXXXX/.../rfMRI_REST1_LR_450.npy          (NEW)

ROI order: [50 subcortical (Tian S3), 400 cortical (Schaefer 400)]
Output shape per subject: (T, 450)

Usage:
    python3 scripts/hcp/extract_hcp_450.py
"""

import os
import urllib.request
from pathlib import Path
import numpy as np
from tqdm import tqdm

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker


# =============================================================================
# CONFIGURATION
# =============================================================================
HCP_ROOT = Path("/sci/labs/arieljaffe/dan.abergel1/HCP_data")
NILEARN_DATA_DIR = str(HCP_ROOT / "nilearn_cache")
SKIP_EXISTING = True


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_nifti_path(subject_id: str) -> Path | None:
    nifti_path = (
        HCP_ROOT / f"subject_{subject_id}" / "MNINonLinear" / "Results"
        / "rfMRI_REST1_LR" / "rfMRI_REST1_LR.nii.gz"
    )
    return nifti_path if nifti_path.exists() else None


def get_output_path(subject_id: str) -> Path:
    return (
        HCP_ROOT / f"subject_{subject_id}" / "MNINonLinear" / "Results"
        / "rfMRI_REST1_LR" / "rfMRI_REST1_LR_450.npy"
    )


TIAN_S3_URL = "https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S3_3T_1mm.nii.gz"


def fetch_tian_s3(cache_dir: str) -> str:
    """Download Tian S3 atlas if not already cached."""
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, "Tian_Subcortex_S3_3T_1mm.nii.gz")
    if not os.path.exists(local_path):
        print(f"  Downloading Tian S3 atlas from GitHub...")
        urllib.request.urlretrieve(TIAN_S3_URL, local_path)
    return local_path


def load_maskers():
    print("  Loading Tian S3 subcortical atlas (50 ROIs)...")
    tian_path = fetch_tian_s3(os.path.join(NILEARN_DATA_DIR, "tian_2020"))
    tian_masker = NiftiLabelsMasker(
        labels_img=tian_path,
        resampling_target="data",
        standardize=False,
        verbose=0,
    )

    print("  Loading Schaefer 400 cortical atlas...")
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=400, resolution_mm=2, data_dir=NILEARN_DATA_DIR
    )
    schaefer_masker = NiftiLabelsMasker(
        labels_img=schaefer.maps,
        standardize=False,
        verbose=0,
    )

    return tian_masker, schaefer_masker


def extract_450(nifti_path: Path, tian_masker, schaefer_masker) -> np.ndarray:
    subcortical = tian_masker.fit_transform(str(nifti_path))  # (T, 50)
    cortical = schaefer_masker.fit_transform(str(nifti_path))  # (T, 400)
    return np.concatenate([subcortical, cortical], axis=1).astype(np.float32)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("HCP 450 ROI EXTRACTION (Tian S3 50 + Schaefer 400)")
    print("=" * 70)

    print("\nDiscovering subjects...")
    subject_dirs = sorted(HCP_ROOT.glob("subject_*"))
    subject_ids = [d.name.replace("subject_", "") for d in subject_dirs]
    print(f"Subjects found: {len(subject_ids)}")

    tian_masker, schaefer_masker = load_maskers()

    stats = {"processed": 0, "skipped": 0, "failed": 0, "errors": []}

    for subject_id in tqdm(subject_ids, desc="450 ROIs"):
        output_path = get_output_path(subject_id)

        if SKIP_EXISTING and output_path.exists():
            stats["skipped"] += 1
            continue

        nifti_path = get_nifti_path(subject_id)
        if nifti_path is None:
            stats["failed"] += 1
            stats["errors"].append(f"{subject_id}: NIfTI not found")
            continue

        try:
            timeseries = extract_450(nifti_path, tian_masker, schaefer_masker)
            np.save(output_path, timeseries)
            stats["processed"] += 1
        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append(f"{subject_id}: {str(e)[:50]}")

    print(f"\nResult:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped:   {stats['skipped']}")
    print(f"  Failed:    {stats['failed']}")

    if stats["errors"] and len(stats["errors"]) <= 10:
        for err in stats["errors"]:
            print(f"    - {err}")

    print("\nDone!")


if __name__ == "__main__":
    main()
