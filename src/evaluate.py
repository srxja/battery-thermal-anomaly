# src/evaluate.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from config import (
    EVALUATION_CSV, ANOMALIES_CSV
)

# ─────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────

def evaluate_all(df):
    """
    Full evaluation of all three methods against
    physics-based ground truth labels.

    Metrics explained simply:

        Precision = of everything we flagged as anomalous,
                    how many actually were?
                    "When we raise the alarm, are we right?"

        Recall    = of everything that was actually anomalous,
                    how many did we catch?
                    "Are we missing any real anomalies?"

        F1        = harmonic mean of precision and recall.
                    Balances both — useful when you care about
                    both false alarms AND missed detections.
                    For battery safety, recall matters more
                    (missing a real anomaly is worse than a
                    false alarm) — worth noting in your paper.

        Confusion matrix:
                        Predicted Normal | Predicted Anomalous
            Actual Normal     TN         |        FP
            Actual Anomalous  FN         |        TP

            FP = false alarm (annoying but safe)
            FN = missed anomaly (potentially dangerous)

    Args:
        df: fully merged DataFrame with gt_label,
            km_anomaly, ae_anomaly, hybrid_anomaly

    Returns:
        summary_df : per-method metrics DataFrame (for paper table)
        full_report: dict with detailed results
    """

    print("===== EVALUATION AGAINST GROUND TRUTH =====\n")

    y_true   = df["gt_label"].values
    methods  = {
        "KMeans"      : df["km_anomaly"].values,
        "Autoencoder" : df["ae_anomaly"].values,
        "Hybrid"      : df["hybrid_anomaly"].values
    }

    records     = []
    full_report = {}

    for method_name, y_pred in methods.items():
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
        cm        = confusion_matrix(y_true, y_pred)

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        records.append({
            "Method"           : method_name,
            "Precision"        : round(precision, 4),
            "Recall"           : round(recall, 4),
            "F1"               : round(f1, 4),
            "TP"               : int(tp),
            "FP"               : int(fp),
            "FN"               : int(fn),
            "TN"               : int(tn),
            "Anomalies_Flagged": int(y_pred.sum())
        })

        full_report[method_name] = {
            "precision"       : precision,
            "recall"          : recall,
            "f1"              : f1,
            "confusion_matrix": cm
        }

        print(f"── {method_name} ──────────────────────")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
        print()

    summary_df = pd.DataFrame(records)

    # Which method is best and why
    _interpret_metrics(summary_df)

    # Save
    summary_df.to_csv(EVALUATION_CSV, index=False)
    print(f"\nEvaluation saved to: {EVALUATION_CSV}")

    return summary_df, full_report


# ─────────────────────────────────────────
# PER C-RATE EVALUATION
# ─────────────────────────────────────────

def evaluate_per_crate(df):
    """
    Evaluate hybrid method performance per C-rate.

    This is your main finding table.

    Expected pattern if pipeline works correctly:
        - Anomaly rate increases 2C → 3C → 4C
        - Recall may decrease at higher C-rates if the
          entire distribution shifts (harder to distinguish
          anomalies from elevated baseline)
        - Precision may increase at 4C if anomalies are
          more severe and thus more obvious

    Any deviation from expected pattern is worth
    discussing — it tells you something about how
    thermal stress manifests at different C-rates.

    Args:
        df: fully merged DataFrame

    Returns:
        crate_df: per C-rate metrics DataFrame
    """

    print("===== PER C-RATE EVALUATION =====\n")

    records = []

    for c_rate, group in df.groupby("c_rate"):
        y_true = group["gt_label"].values
        y_pred = group["hybrid_anomaly"].values

        n_total   = len(group)
        n_anomaly = int(y_pred.sum())
        n_gt      = int(y_true.sum())

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)

        # Feature means for this C-rate
        # These tell you WHY anomaly rates differ
        avg_temp_mean    = group["avg_temp"].mean()
        max_temp_mean    = group["max_temp"].mean()
        hotspot_area_mean= group["hotspot_area"].mean()
        entropy_mean     = group["entropy"].mean()
        ae_error_mean    = group["reconstruction_error"].mean()

        records.append({
            "C_Rate"              : c_rate,
            "Total_Images"        : n_total,
            "GT_Anomalies"        : n_gt,
            "Detected_Anomalies"  : n_anomaly,
            "Anomaly_Rate_%"      : round(100 * n_anomaly / n_total, 2),
            "Precision"           : round(precision, 4),
            "Recall"              : round(recall, 4),
            "F1"                  : round(f1, 4),
            "Mean_AvgTemp"        : round(avg_temp_mean, 4),
            "Mean_MaxTemp"        : round(max_temp_mean, 4),
            "Mean_HotspotArea"    : round(hotspot_area_mean, 4),
            "Mean_Entropy"        : round(entropy_mean, 4),
            "Mean_AE_Error"       : round(ae_error_mean, 6),
        })

        print(f"── {c_rate} ──────────────────────────")
        print(f"  Total images      : {n_total}")
        print(f"  GT anomalies      : {n_gt}")
        print(f"  Detected          : {n_anomaly} ({100*n_anomaly/n_total:.1f}%)")
        print(f"  Precision         : {precision:.4f}")
        print(f"  Recall            : {recall:.4f}")
        print(f"  F1                : {f1:.4f}")
        print(f"  Mean avg_temp     : {avg_temp_mean:.4f}")
        print(f"  Mean max_temp     : {max_temp_mean:.4f}")
        print(f"  Mean hotspot_area : {hotspot_area_mean:.4f}")
        print(f"  Mean AE error     : {ae_error_mean:.6f}")
        print()

    crate_df = pd.DataFrame(records)

    # Interpret the trend
    _interpret_crate_trend(crate_df)

    return crate_df


# ─────────────────────────────────────────
# FEATURE IMPORTANCE ANALYSIS
# ─────────────────────────────────────────

def analyze_feature_differences(df):
    """
    Compare mean feature values between anomalous
    and normal images — per C-rate.

    This answers: WHY is an image flagged as anomalous?
    Which features drive the anomaly score?

    If anomalous images have:
        higher max_temp     → peak thermal stress signal
        higher hotspot_area → larger stressed region
        higher entropy      → more chaotic heat distribution
        higher gradient     → sharper thermal boundaries
        extreme hotspot_cx/cy → stress concentrated at edges

    Each of these has a physical interpretation you
    can discuss in your paper.

    Args:
        df: fully merged DataFrame

    Returns:
        diff_df: DataFrame of feature differences
                 anomalous vs normal per C-rate
    """

    feature_cols = [
        "avg_temp", "max_temp", "gradient", "laplacian_var",
        "hotspot_area", "entropy", "hotspot_cx", "hotspot_cy",
        "reconstruction_error", "km_score"
    ]

    print("===== FEATURE DIFFERENCES: ANOMALOUS vs NORMAL =====\n")

    records = []

    for c_rate, group in df.groupby("c_rate"):
        anomalous = group[group["hybrid_anomaly"] == 1]
        normal    = group[group["hybrid_anomaly"] == 0]

        print(f"── {c_rate} ──────────────────────────")

        if len(anomalous) == 0:
            print("  No anomalies detected.")
            continue

        for feat in feature_cols:
            if feat not in group.columns:
                continue

            mean_anom   = anomalous[feat].mean()
            mean_norm   = normal[feat].mean()
            diff        = mean_anom - mean_norm
            pct_diff    = 100 * diff / (mean_norm + 1e-10)

            direction   = "↑" if diff > 0 else "↓"
            records.append({
                "C_Rate"        : c_rate,
                "Feature"       : feat,
                "Mean_Anomalous": round(mean_anom, 6),
                "Mean_Normal"   : round(mean_norm, 6),
                "Difference"    : round(diff, 6),
                "Pct_Change"    : round(pct_diff, 2),
                "Direction"     : direction
            })

            print(f"  {feat:<25}: "
                  f"anomalous={mean_anom:.4f} "
                  f"normal={mean_norm:.4f} "
                  f"{direction} {abs(pct_diff):.1f}%")
        print()

    diff_df = pd.DataFrame(records)
    return diff_df


# ─────────────────────────────────────────
# SAVE FINAL ANOMALY FILE
# ─────────────────────────────────────────

def save_final_anomalies(df):
    """
    Save the complete annotated DataFrame.
    This is the single source of truth for all results.

    Columns saved:
        filename, c_rate, all features,
        gt_label, km_anomaly, ae_anomaly,
        hybrid_anomaly, hybrid_score, agreement
    """

    cols_to_save = [
        "filename", "c_rate",
        "avg_temp", "max_temp", "gradient", "laplacian_var",
        "hotspot_area", "entropy", "hotspot_cx", "hotspot_cy",
        "gt_label", "gt_max_thresh", "gt_grad_thresh",
        "PCA1", "PCA2", "km_cluster", "km_score", "km_anomaly",
        "reconstruction_error", "ae_anomaly",
        "hybrid_score", "hybrid_anomaly", "agreement"
    ]

    # Only keep columns that exist
    cols_to_save = [c for c in cols_to_save if c in df.columns]
    df[cols_to_save].to_csv(ANOMALIES_CSV, index=False)
    print(f"Final anomalies saved to: {ANOMALIES_CSV}")


# ─────────────────────────────────────────
# INTERNAL INTERPRETATION HELPERS
# ─────────────────────────────────────────

def _interpret_metrics(summary_df):
    """
    Interpret which method performs best and why.
    Prints paper-ready observations.
    """

    print("===== INTERPRETATION =====\n")

    best_f1  = summary_df.loc[summary_df["F1"].idxmax()]
    best_rec = summary_df.loc[summary_df["Recall"].idxmax()]

    print(f"Best F1        : {best_f1['Method']} ({best_f1['F1']:.4f})")
    print(f"Best Recall    : {best_rec['Method']} ({best_rec['Recall']:.4f})")
    print()

    hybrid = summary_df[summary_df["Method"] == "Hybrid"].iloc[0]
    km     = summary_df[summary_df["Method"] == "KMeans"].iloc[0]
    ae     = summary_df[summary_df["Method"] == "Autoencoder"].iloc[0]

    if hybrid["F1"] > km["F1"] and hybrid["F1"] > ae["F1"]:
        print("Hybrid outperforms both individual methods on F1.")
        print("This validates the fusion approach.")
    elif hybrid["F1"] > km["F1"]:
        print("Hybrid outperforms KMeans but not Autoencoder on F1.")
        print("Autoencoder is the stronger individual method here.")
    elif hybrid["F1"] > ae["F1"]:
        print("Hybrid outperforms Autoencoder but not KMeans on F1.")
        print("KMeans is the stronger individual method here.")
    else:
        print("Individual methods outperform Hybrid.")
        print("Consider tuning HYBRID_W_RECON and HYBRID_W_KMEANS in config.py")

    print()
    print("Note: For battery safety applications, recall is critical.")
    print("A missed anomaly (FN) is more dangerous than a false alarm (FP).")
    print(f"  KMeans recall     : {km['Recall']:.4f}")
    print(f"  Autoencoder recall: {ae['Recall']:.4f}")
    print(f"  Hybrid recall     : {hybrid['Recall']:.4f}")


def _interpret_crate_trend(crate_df):
    """
    Check if anomaly rate increases with C-rate as expected.
    """

    print("===== C-RATE TREND INTERPRETATION =====\n")

    rates  = crate_df["C_Rate"].tolist()
    counts = crate_df["Anomaly_Rate_%"].tolist()

    print("Anomaly rate trend:")
    for r, c in zip(rates, counts):
        bar = "█" * int(c)
        print(f"  {r}: {bar} {c:.1f}%")

    if len(counts) == 3:
        if counts[0] < counts[1] < counts[2]:
            print("\n✓ Anomaly rate increases monotonically with C-rate.")
            print("  This confirms higher charge stress → more thermal anomalies.")
            print("  This is your primary finding.")
        elif counts[2] > counts[0]:
            print("\n~ Anomaly rate generally increases but not monotonically.")
            print("  Discuss why 3C may deviate from the trend.")
        else:
            print("\n✗ No clear increasing trend with C-rate.")
            print("  Investigate whether images loaded correctly per C-rate.")
            print("  Check if C-rate labels are assigned correctly in features.py")

    print()
    print("Mean feature trends across C-rates:")
    for feat in ["Mean_AvgTemp", "Mean_MaxTemp", "Mean_HotspotArea"]:
        if feat in crate_df.columns:
            vals = crate_df[feat].tolist()
            print(f"  {feat:<25}: {' → '.join([str(v) for v in vals])}")