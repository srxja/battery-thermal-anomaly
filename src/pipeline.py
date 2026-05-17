# src/pipeline.py

import sys
import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────

BASE_DIR = "/content/drive/MyDrive/0research"
SRC_DIR  = os.path.join(BASE_DIR, "battery-thermal-anomaly/src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ─────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────

from config import (
    C_RATES, FEATURES_CSV, ANOMALIES_CSV,
    HYBRID_THRESHOLD
)
from features import (
    load_all_crates,
    load_all_images_spatial
)
from ground_truth import (
    assign_ground_truth,
    check_ground_truth_quality
)
from autoencoder import (
    train_autoencoder,
    compute_reconstruction_scores
)
from clustering import (
    remove_multicollinear_features,
    apply_pca,
    find_optimal_k,
    apply_kmeans
)
from fusion import (
    fuse_scores,
    compute_gmm_separability,
    analyze_disagreements
)
from evaluate import (
    evaluate_all,
    evaluate_per_crate,
    analyze_feature_differences,
    save_final_anomalies
)
from visualize import generate_all_plots


# ─────────────────────────────────────────
# PIPELINE STAGES
# ─────────────────────────────────────────

def stage_1_features():
    """
    Load all images, extract features, assign ground truth.

    Output:
        df_features : DataFrame with all features + gt_label
    """
    print("\n" + "="*50)
    print("STAGE 1: FEATURE EXTRACTION")
    print("="*50)

    df_features = load_all_crates()
    df_features = assign_ground_truth(df_features)
    check_ground_truth_quality(df_features)

    return df_features


def stage_2_autoencoder(df_features):
    """
    Train Conv autoencoder on all images.
    Compute per-image reconstruction scores.

    Output:
        autoencoder  : trained model
        history      : training history
        df_ae_scores : DataFrame with reconstruction_error,
                       ae_anomaly per image
    """
    print("\n" + "="*50)
    print("STAGE 2: CONV AUTOENCODER")
    print("="*50)

    # Load spatial images (N, H, W, 1) for Conv AE
    X_spatial, filenames, c_rates = load_all_images_spatial()

    print(f"Loaded {len(X_spatial)} images")
    print(f"Image shape: {X_spatial.shape[1:]}")

    autoencoder, encoder, history = train_autoencoder(X_spatial)

    df_ae_scores, ae_threshold = compute_reconstruction_scores(
        autoencoder, X_spatial, filenames, c_rates
    )

    return autoencoder, encoder, history, df_ae_scores, ae_threshold


def stage_3_clustering(df_features):
    """
    VIF filtering → PCA → optimal K → KMeans.

    Output:
        df_clustered  : DataFrame with PCA coords,
                        cluster labels, km_score, km_anomaly
        kmeans        : trained KMeans model
        selected_cols : VIF-filtered feature columns
        vif_table     : for paper
        pca_meta      : explained variance + loadings
    """
    print("\n" + "="*50)
    print("STAGE 3: CLUSTERING")
    print("="*50)

    # VIF filtering
    selected_cols, vif_table = remove_multicollinear_features(
        df_features
    )

    # PCA
    df_pca, pca, scaler, explained = apply_pca(
        df_features, selected_cols
    )

    # Find best K
    best_k, k_results = find_optimal_k(df_pca)

    # KMeans
    df_clustered, kmeans = apply_kmeans(df_pca, best_k)

    pca_meta = {
        "explained"    : explained,
        "pca"          : pca,
        "scaler"       : scaler,
        "k_results"    : k_results,
        "best_k"       : best_k
    }

    return df_clustered, kmeans, selected_cols, vif_table, pca_meta


def stage_4_fusion(df_clustered, df_ae_scores):
    """
    Fuse KMeans and autoencoder scores into hybrid score.
    Compute GMM separability and disagreement analysis.

    Output:
        df_fused     : merged DataFrame with all scores
        separability : GMM separability score
        agreement_df : per C-rate agreement breakdown
    """
    print("\n" + "="*50)
    print("STAGE 4: SCORE FUSION")
    print("="*50)

    df_fused = fuse_scores(df_clustered, df_ae_scores)

    separability, gmm = compute_gmm_separability(df_fused)

    agreement_df = analyze_disagreements(df_fused)

    return df_fused, separability, agreement_df


def stage_5_evaluation(df_fused):
    """
    Full evaluation against ground truth.
    Per-method metrics, per-C-rate breakdown,
    feature difference analysis.

    Output:
        summary_df  : method comparison table
        crate_df    : per C-rate metrics
        diff_df     : feature differences anomalous vs normal
    """
    print("\n" + "="*50)
    print("STAGE 5: EVALUATION")
    print("="*50)

    # Merge ground truth back into fused df
    # gt_label comes from df_features, already in df_fused
    # via the clustering merge chain
    summary_df, full_report = evaluate_all(df_fused)

    crate_df = evaluate_per_crate(df_fused)

    diff_df = analyze_feature_differences(df_fused)

    # Add per-method rates to crate_df for plotting
    km_rates = df_fused.groupby("c_rate").apply(
        lambda g: 100 * g["km_anomaly"].sum() / len(g)
    ).reset_index()
    km_rates.columns = ["c_rate", "KMeans_Rate_%"]

    ae_rates = df_fused.groupby("c_rate").apply(
        lambda g: 100 * g["ae_anomaly"].sum() / len(g)
    ).reset_index()
    ae_rates.columns = ["c_rate", "AE_Rate_%"]

    crate_df = crate_df.merge(km_rates, on="c_rate", how="left")
    crate_df = crate_df.merge(ae_rates, on="c_rate", how="left")

    # Save final anomaly file
    save_final_anomalies(df_fused)

    return summary_df, crate_df, diff_df, full_report


def stage_6_visualize(
    df_fused, history, autoencoder,
    crate_df, diff_df
):
    """
    Generate all plots.
    """
    print("\n" + "="*50)
    print("STAGE 6: VISUALIZATION")
    print("="*50)

    generate_all_plots(
        df       = df_fused,
        history  = history,
        autoencoder = autoencoder,
        crate_df = crate_df,
        diff_df  = diff_df,
        threshold= HYBRID_THRESHOLD
    )


# ─────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────

def run():
    """
    Run the complete pipeline end to end.

    Stage 1 : Feature extraction + ground truth
    Stage 2 : Conv autoencoder training + scoring
    Stage 3 : VIF → PCA → KMeans clustering
    Stage 4 : Hybrid score fusion + disagreement analysis
    Stage 5 : Evaluation against ground truth
    Stage 6 : All visualizations

    All outputs saved to:
        /content/drive/MyDrive/0research/results/
    """

    print("\n" + "="*50)
    print("BATTERY THERMAL ANOMALY DETECTION PIPELINE")
    print("Li-Ion Batteries | 2C, 3C, 4C Charge Rates")
    print("="*50)

    # ── Stage 1 ───────────────────────────
    df_features = stage_1_features()

    # ── Stage 2 ───────────────────────────
    autoencoder, encoder, history, df_ae_scores, ae_threshold = (
        stage_2_autoencoder(df_features)
    )

    # ── Stage 3 ───────────────────────────
    df_clustered, kmeans, selected_cols, vif_table, pca_meta = (
        stage_3_clustering(df_features)
    )

    # ── Stage 4 ───────────────────────────
    df_fused, separability, agreement_df = stage_4_fusion(
        df_clustered, df_ae_scores
    )

    # Merge ground truth into fused df
    gt_cols = [
        "filename", "c_rate", "gt_label",
        "gt_max_thresh", "gt_grad_thresh"
    ]
    
    # Only merge if gt_label not already present
    if "gt_label" not in df_fused.columns:
        df_fused = df_fused.merge(
            df_features[gt_cols],
            on=["filename", "c_rate"],
            how="left"
        )
        print(f"gt_label merged. NaN count: {df_fused['gt_label'].isna().sum()}")
    else:
        print("gt_label already present in df_fused")
    
    print(f"df_fused columns: {df_fused.columns.tolist()}")

    # ── Stage 5 ───────────────────────────
    summary_df, crate_df, diff_df, full_report = (
        stage_5_evaluation(df_fused)
    )

    # ── Stage 6 ───────────────────────────
    stage_6_visualize(
        df_fused, history, autoencoder,
        crate_df, diff_df
    )

    # ── Final summary ─────────────────────
    _print_final_summary(
        df_fused, summary_df, crate_df, separability, pca_meta
    )

    return {
        "df_fused"      : df_fused,
        "summary_df"    : summary_df,
        "crate_df"      : crate_df,
        "diff_df"       : diff_df,
        "autoencoder"   : autoencoder,
        "encoder"       : encoder,
        "history"       : history,
        "separability"  : separability,
        "vif_table"     : vif_table,
        "pca_meta"      : pca_meta
    }


# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────

def _print_final_summary(
    df_fused, summary_df, crate_df,
    separability, pca_meta
):
    """
    Print a clean end-to-end summary of all findings.
    This is what you read first after a pipeline run.
    """

    print("\n" + "="*50)
    print("PIPELINE COMPLETE — RESULTS SUMMARY")
    print("="*50)

    total = len(df_fused)
    print(f"\nTotal images processed : {total}")
    for c_rate in C_RATES:
        n = (df_fused["c_rate"] == c_rate).sum()
        print(f"  {c_rate}: {n} images")

    print(f"\nGMM Separability       : {separability:.4f}")
    print(f"PCA explained variance : "
          f"{sum(pca_meta['explained']):.3f} "
          f"({pca_meta['explained'][0]:.3f} + "
          f"{pca_meta['explained'][1]:.3f})")
    print(f"Best K (KMeans)        : {pca_meta['best_k']}")

    print("\n── Method Comparison ──────────────────")
    print(summary_df[[
        "Method", "Precision", "Recall", "F1",
        "Anomalies_Flagged"
    ]].to_string(index=False))

    print("\n── Per C-Rate (Hybrid) ─────────────────")
    print(crate_df[[
        "C_Rate", "Total_Images", "GT_Anomalies",
        "Detected_Anomalies", "Anomaly_Rate_%",
        "Precision", "Recall", "F1"
    ]].to_string(index=False))

    print("\n── Key Finding ─────────────────────────")
    rates = crate_df["Anomaly_Rate_%"].tolist()
    crates = crate_df["C_Rate"].tolist()
    for c, r in zip(crates, rates):
        bar = "█" * int(r)
        print(f"  {c}: {bar} {r:.1f}%")

    print(f"\nAll results saved to:")
    print(f"  /content/drive/MyDrive/0research/results/")
    print(f"\nPlots saved to:")
    print(f"  /content/drive/MyDrive/0research/results/plots/")