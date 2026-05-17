# src/fusion.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from config import (
    HYBRID_W_RECON, HYBRID_W_KMEANS,
    HYBRID_THRESHOLD, ANOMALY_PERCENTILE
)

# ─────────────────────────────────────────
# SCORE FUSION
# ─────────────────────────────────────────

def fuse_scores(df_clustered, df_ae_scores):
    """
    Combine KMeans and autoencoder anomaly scores
    into a single hybrid score.

    Why fuse two methods?
        Each method catches different types of anomalies:

        KMeans catches GLOBAL outliers —
            images that don't belong to any cluster pattern.
            Like a student whose overall grade profile
            doesn't match any known performance group.

        Autoencoder catches LOCAL anomalies —
            images with unusual spatial patterns even if
            their overall statistics look normal.
            Like a student whose essay structure is bizarre
            even though their word count is average.

        Together they're more reliable than either alone.
        An image flagged by BOTH is very likely anomalous.
        An image flagged by only one deserves scrutiny.

    Fusion strategy:
        1. Normalize both scores to [0,1] (MinMaxScaler)
           so neither dominates due to scale differences
        2. Weighted sum: autoencoder weighted higher (0.6)
           because spatial reconstruction error is more
           directly tied to thermal anomaly patterns
        3. Flag as anomalous if hybrid score >= threshold

    Args:
        df_clustered  : DataFrame with km_score, km_anomaly
        df_ae_scores  : DataFrame with reconstruction_error,
                        ae_anomaly

    Returns:
        df_fused: merged DataFrame with hybrid scores
                  and agreement analysis
    """

    # ── Merge on filename + c_rate ────────────────
    df = pd.merge(
        df_clustered,
        df_ae_scores[["filename", "c_rate",
                      "reconstruction_error", "ae_anomaly"]],
        on=["filename", "c_rate"],
        how="inner"
    )

    print(f"Merged dataframe: {len(df)} images")

    # ── Normalize scores ──────────────────────────
    scaler  = MinMaxScaler()
    scores  = df[["km_score", "reconstruction_error"]].values
    normed  = scaler.fit_transform(scores)

    df["km_score_norm"]  = normed[:, 0]
    df["ae_score_norm"]  = normed[:, 1]

    # ── Weighted hybrid score ─────────────────────
    df["hybrid_score"] = (
        HYBRID_W_RECON   * df["ae_score_norm"] +
        HYBRID_W_KMEANS  * df["km_score_norm"]
    )

    # ── Hybrid anomaly label ──────────────────────
    df["hybrid_anomaly"] = (
        df["hybrid_score"] >= HYBRID_THRESHOLD
    ).astype(int)

    # ── Agreement analysis ────────────────────────
    # This is where it gets interesting
    # Disagreements between methods tell you something
    df["agreement"] = "both_normal"
    df.loc[
        (df["km_anomaly"] == 1) & (df["ae_anomaly"] == 1),
        "agreement"
    ] = "both_anomalous"
    df.loc[
        (df["km_anomaly"] == 1) & (df["ae_anomaly"] == 0),
        "agreement"
    ] = "kmeans_only"
    df.loc[
        (df["km_anomaly"] == 0) & (df["ae_anomaly"] == 1),
        "agreement"
    ] = "ae_only"

    # ── Summary ───────────────────────────────────
    _print_fusion_summary(df)

    return df


# ─────────────────────────────────────────
# GMM SEPARABILITY
# ─────────────────────────────────────────

def compute_gmm_separability(df):
    """
    Fit a 2-component GMM to hybrid scores and measure
    how well separated the normal/anomalous distributions are.

    Think of hybrid scores as heights of people in a room.
    If there are clearly two groups (short and tall),
    GMM separability is high — easy to draw a line between them.
    If everyone is roughly the same height, separability is low
    — hard to define who is "anomalous."

    High separability = your anomaly scores are meaningful.
    Low separability = scores are noisy, threshold is arbitrary.

    Formula:
        separability = |mu1 - mu2| / sqrt(0.5 * (var1 + var2))

        This is essentially a signal-to-noise ratio —
        how far apart are the two peaks relative to their spread.

    Args:
        df: DataFrame with hybrid_score column

    Returns:
        separability : float score
        gmm          : fitted GaussianMixture object
    """

    scores = df["hybrid_score"].values.reshape(-1, 1)

    gmm    = GaussianMixture(
        n_components=2,
        random_state=42
    ).fit(scores)

    mu     = np.sort(gmm.means_.ravel())
    var    = np.sort(gmm.covariances_.ravel())
    denom  = np.sqrt(0.5 * (var[0] + var[1])) if (var[0] + var[1]) > 0 else 1e-6
    sep    = float(abs(mu[1] - mu[0]) / denom)

    print(f"\n===== GMM SEPARABILITY =====")
    print(f"Component 1 mean : {mu[0]:.4f}")
    print(f"Component 2 mean : {mu[1]:.4f}")
    print(f"Separability     : {sep:.4f}")

    if sep > 2.0:
        print("  GOOD: Two clearly distinct score distributions.")
        print("  Your anomaly threshold is meaningful.")
    elif sep > 1.0:
        print("  MODERATE: Some separation between distributions.")
        print("  Threshold is reasonable but not sharp.")
    else:
        print("  LOW: Scores are not well separated.")
        print("  Consider tuning HYBRID_W_RECON/HYBRID_W_KMEANS in config.py")

    return sep, gmm


# ─────────────────────────────────────────
# DISAGREEMENT ANALYSIS
# ─────────────────────────────────────────

def analyze_disagreements(df):
    """
    Analyze cases where KMeans and autoencoder disagree.

    This is one of the most valuable analyses for your paper.

    kmeans_only anomalies:
        Image is a statistical outlier in PCA space but the
        autoencoder reconstructed it well.
        → Unusual feature combination but no spatial anomaly.
        → Could be a borderline case or a global shift in
          operating conditions at that C-rate.

    ae_only anomalies:
        Autoencoder struggled to reconstruct it but features
        look statistically normal.
        → Unusual spatial pattern that features didn't capture.
        → This is where Conv AE adds real value over KMeans.
        → Most likely a localized hotspot that avg_temp missed.

    both_anomalous:
        Both methods agree — highest confidence anomalies.
        → These are the ones you want to show visually in paper.

    Args:
        df: fused DataFrame with agreement column

    Returns:
        summary: DataFrame with counts per agreement type
                 per C-rate
    """

    print("\n===== DISAGREEMENT ANALYSIS =====")

    agreement_counts = df.groupby(
        ["c_rate", "agreement"]
    ).size().unstack(fill_value=0)

    print(agreement_counts.to_string())

    # What fraction of ae_only anomalies have high hotspot_area?
    # If ae catches spatial anomalies features miss, we expect
    # ae_only images to have unusual spatial patterns
    ae_only = df[df["agreement"] == "ae_only"]
    both    = df[df["agreement"] == "both_anomalous"]
    normal  = df[df["agreement"] == "both_normal"]

    if len(ae_only) > 0 and "hotspot_area" in df.columns:
        print(f"\nMean hotspot_area:")
        print(f"  both_normal     : {normal['hotspot_area'].mean():.4f}")
        print(f"  ae_only         : {ae_only['hotspot_area'].mean():.4f}")
        print(f"  both_anomalous  : {both['hotspot_area'].mean():.4f}")
        print()
        print("  If ae_only hotspot_area > both_normal:")
        print("  Conv AE is catching spatial anomalies that KMeans missed.")
        print("  This is a key finding for your paper.")

    return agreement_counts


# ─────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────

def _print_fusion_summary(df):
    """Print a clean summary of fusion results."""

    total = len(df)
    print("\n===== FUSION SUMMARY =====")
    print(f"Total images     : {total}")
    print(f"Weights          : AE={HYBRID_W_RECON}, KMeans={HYBRID_W_KMEANS}")
    print(f"Threshold        : {HYBRID_THRESHOLD}")
    print()

    # Per method counts
    print(f"KMeans anomalies    : {df['km_anomaly'].sum()}")
    print(f"AE anomalies        : {df['ae_anomaly'].sum()}")
    print(f"Hybrid anomalies    : {df['hybrid_anomaly'].sum()}")
    print()

    # Agreement breakdown
    agreement_total = df["agreement"].value_counts()
    print("Agreement breakdown:")
    for label, count in agreement_total.items():
        print(f"  {label:<20}: {count} ({100*count/total:.1f}%)")
    print()

    # Per C-rate hybrid anomaly rate
    print("Hybrid anomaly rate per C-rate:")
    for c_rate, group in df.groupby("c_rate"):
        n   = group["hybrid_anomaly"].sum()
        pct = 100 * n / len(group)
        print(f"  {c_rate}: {n} / {len(group)} ({pct:.1f}%)")