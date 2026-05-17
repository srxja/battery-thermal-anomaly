# src/ground_truth.py

import numpy as np
import pandas as pd
from config import (
    GT_MAX_TEMP_PERCENTILE,
    GT_GRADIENT_PERCENTILE,
)

# ─────────────────────────────────────────
# PHYSICS-BASED GROUND TRUTH LABELING
# ─────────────────────────────────────────

def assign_ground_truth(df):
    """
    Assign binary ground truth labels using physics-based thresholds.

    Logic:
        An image is labeled anomalous (1) if BOTH conditions hold:
            1. max_temp  >= 95th percentile of max_temp  across ALL images
            2. gradient  >= 90th percentile of gradient  across ALL images

        Why both conditions?
            max_temp alone flags hot images but not necessarily stressed ones.
            gradient alone flags sharp boundaries but not necessarily hot ones.
            Together they identify images that are BOTH unusually hot AND
            show sharp thermal boundaries — the physical signature of a
            developing hotspot or thermal runaway precursor.

        Think of it like a weather alert system:
            Temperature alone doesn't cause a flood warning.
            Rainfall rate alone doesn't either.
            Both together crossing a threshold triggers the alert.

    This labeling is:
        - Independent of model output (no circularity)
        - Physically motivated (not arbitrary)
        - Defensible in a paper as "domain-informed proxy ground truth"

    Args:
        df: DataFrame with at least max_temp and gradient columns

    Returns:
        df with new columns:
            gt_label        : 1 = anomalous, 0 = normal
            gt_max_thresh   : threshold used for max_temp
            gt_grad_thresh  : threshold used for gradient
    """

    df = df.copy()

    # Compute thresholds across ALL images regardless of C-rate
    # This is important — thresholds are global, not per C-rate
    # So a 4C image isn't anomalous just because 4C runs hotter overall
    # It's anomalous if it's an outlier within the full distribution
    max_temp_thresh = np.percentile(df["max_temp"], GT_MAX_TEMP_PERCENTILE)
    gradient_thresh = np.percentile(df["gradient"], GT_GRADIENT_PERCENTILE)

    df["gt_label"] = (
        (df["max_temp"] >= max_temp_thresh) &
        (df["gradient"] >= gradient_thresh)
    ).astype(int)

    # Store thresholds as columns so they're visible in the output CSV
    df["gt_max_thresh"]  = max_temp_thresh
    df["gt_grad_thresh"] = gradient_thresh

    # ── Summary ──────────────────────────────
    total     = len(df)
    n_anomaly = df["gt_label"].sum()

    print("===== GROUND TRUTH SUMMARY =====")
    print(f"Max temp threshold  (p{GT_MAX_TEMP_PERCENTILE}): {max_temp_thresh:.4f}")
    print(f"Gradient threshold  (p{GT_GRADIENT_PERCENTILE}): {gradient_thresh:.4f}")
    print(f"Total images        : {total}")
    print(f"Labeled anomalous   : {n_anomaly} ({100*n_anomaly/total:.1f}%)")
    print(f"Labeled normal      : {total - n_anomaly} ({100*(total-n_anomaly)/total:.1f}%)")
    print()
    print("Per C-rate breakdown:")
    for c_rate, group in df.groupby("c_rate"):
        n = group["gt_label"].sum()
        print(f"  {c_rate}: {n} anomalous / {len(group)} total ({100*n/len(group):.1f}%)")

    return df


# ─────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────

def check_ground_truth_quality(df):
    """
    Checks that ground truth labels are reasonable before
    using them for evaluation.

    Flags:
        - If < 2% labeled anomalous: thresholds may be too strict
        - If > 20% labeled anomalous: thresholds may be too loose
        - If one C-rate has 0 anomalies: worth investigating
        - If all anomalies are in one C-rate: also worth investigating

    These don't break anything — they're warnings for you to
    think about before writing the paper.
    """

    total     = len(df)
    n_anomaly = df["gt_label"].sum()
    rate      = n_anomaly / total

    print("===== GROUND TRUTH QUALITY CHECK =====")

    if rate < 0.02:
        print(f"  WARNING: Only {100*rate:.1f}% labeled anomalous.")
        print(f"  Consider lowering GT_MAX_TEMP_PERCENTILE or GT_GRADIENT_PERCENTILE in config.py")
    elif rate > 0.20:
        print(f"  WARNING: {100*rate:.1f}% labeled anomalous — this is quite high.")
        print(f"  Consider raising GT_MAX_TEMP_PERCENTILE or GT_GRADIENT_PERCENTILE in config.py")
    else:
        print(f"  OK: {100*rate:.1f}% anomalous — reasonable range.")

    print()
    per_crate = df.groupby("c_rate")["gt_label"].sum()
    for c_rate, count in per_crate.items():
        if count == 0:
            print(f"  WARNING: {c_rate} has 0 anomalies. Check if images loaded correctly.")
        else:
            print(f"  OK: {c_rate} has {count} anomalies.")

    # Check if anomalies increase with C-rate (expected physically)
    rates_list = [C for C in ["2C", "3C", "4C"] if C in per_crate.index]
    counts     = [per_crate[c] for c in rates_list]

    if len(counts) == 3:
        if counts[0] <= counts[1] <= counts[2]:
            print()
            print("  GOOD: Anomaly count increases with C-rate as expected physically.")
        else:
            print()
            print(f"  NOTE: Anomaly counts {dict(zip(rates_list, counts))} don't")
            print(f"  strictly increase with C-rate. This is worth discussing in the paper.")