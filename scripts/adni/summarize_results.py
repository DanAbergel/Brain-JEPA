"""
Summarize Brain-JEPA fine-tuning results for ADNI degradation.
Same format as BNT summarize_results.py for easy comparison.

Reads log.txt (one JSON per line) from each horizon's output dir.
Extracts best epoch (by val AUC) metrics.

Usage:
    python3 scripts/adni/summarize_results.py [output_root]
"""

import sys
import json
import numpy as np
from pathlib import Path


LABELS = [
    "degradation_binary_1year",
    "degradation_binary_2years",
    "degradation_binary_3years",
]

HORIZON_MAP = {
    "degradation_binary_1year": "1y",
    "degradation_binary_2years": "2y",
    "degradation_binary_3years": "3y",
}

METRICS = [
    ("test_acc1", "Test Acc"),
    ("test_precision", "Test Prec"),
    ("test_recall", "Test Recall"),
    ("test_f1", "Test F1"),
]


def fmt(val):
    if val is not None:
        return f"{val:.4f}"
    return "N/A"


def load_log(log_path):
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('{'):
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def best_epoch(entries, metric='val_auc'):
    """Find best epoch by given metric (default: val AUC)."""
    valid = [e for e in entries if metric in e]
    if not valid:
        return entries[-1] if entries else None
    return max(valid, key=lambda e: e.get(metric, 0))


def main():
    output_root = sys.argv[1] if len(sys.argv) > 1 else './output_dir'

    print()
    print("=" * 130)
    print("  Brain-JEPA on ADNI — Results Summary (best epoch by Val F1)")
    print("=" * 130)

    # Header
    header = f"  {'Label':<30}"
    for _, display in METRICS:
        header += f"  {display:>16}"
    print(header)
    print(f"  {'-' * 125}")

    for label in LABELS:
        h = HORIZON_MAP[label]
        log_path = Path(output_root) / f'degradation_{h}' / 'fine_tune' / 'adni_degradation' / 'jepa-ep300' / 'ft_output' / 'log.txt'

        if not log_path.exists():
            print(f"  {label:<30}   (no results)")
            continue

        entries = load_log(log_path)
        if not entries:
            print(f"  {label:<30}   (empty log)")
            continue

        best = best_epoch(entries, 'val_f1')
        if best is None:
            print(f"  {label:<30}   (no valid entries)")
            continue

        row = f"  {label:<30}"
        for key, _ in METRICS:
            val = best.get(key)
            if val is not None:
                if 'acc' in key.lower():
                    row += f"  {val / 100:>16.4f}"  # convert % to 0-1 scale
                else:
                    row += f"  {val:>16.4f}"
            else:
                row += f"  {'N/A':>16}"
        print(row)

    print(f"\n{'=' * 130}")
    print()


if __name__ == '__main__':
    main()
