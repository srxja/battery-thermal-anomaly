# src/visualize.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
from config import (
    PLOTS_DIR, C_RATES, get_image_folder,
    RESIZE_DIM, BLUR_KERNEL, BLUR_SIGMA
)

# ─────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────

CRATE_COLORS = {
    "2C": "#4C72B0",
    "3C": "#DD8452",
    "4C": "#55A868"
}

def _save(fig, filename):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────
# 1. TRAINING HISTORY
# ─────────────────────────────────────────

def plot_training_history(history):
    """
    Plot autoencoder train vs validation loss.

    What to look for:
        - Both curves decreasing = model is learning
        - Val loss flattening before train loss = early stopping
          kicked in at the right time
        - Val loss increasing while train decreases = overfitting
          (bad — means model memorized training images)

    With only ~500 images this is worth checking carefully.
    """

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(history.history["loss"],
            label="Train Loss", color="#4C72B0", linewidth=2)
    ax.plot(history.history["val_loss"],
            label="Val Loss", color="#DD8452",
            linewidth=2, linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save(fig, "01_training_history.png")


# ─────────────────────────────────────────
# 2. FEATURE DISTRIBUTIONS PER C-RATE
# ─────────────────────────────────────────

def plot_feature_distributions(df):
    """
    KDE plots of each feature split by C-rate.

    What to look for:
        - Distributions shifting right at higher C-rates
          = higher charge stress = higher feature values
        - Overlapping distributions = features don't
          separate C-rates well (less useful for clustering)
        - Bimodal distributions = natural anomaly/normal split

    These plots explain WHY your clustering works
    (or doesn't) before you even run KMeans.
    """

    feature_cols = [
        "avg_temp", "max_temp", "gradient", "laplacian_var",
        "hotspot_area", "entropy", "hotspot_cx", "hotspot_cy"
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes      = axes.flatten()

    for i, feat in enumerate(feature_cols):
        ax = axes[i]
        for c_rate in C_RATES:
            subset = df[df["c_rate"] == c_rate][feat]
            subset.plot.kde(
                ax=ax,
                label=c_rate,
                color=CRATE_COLORS[c_rate],
                linewidth=2
            )
        ax.set_title(feat)
        ax.set_xlabel("")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Feature Distributions by C-Rate\n"
        "Rightward shift at higher C-rates indicates increasing thermal stress",
        fontsize=13
    )
    plt.tight_layout()
    _save(fig, "02_feature_distributions.png")


# ─────────────────────────────────────────
# 3. PCA SCATTER
# ─────────────────────────────────────────

def plot_pca_scatter(df):
    """
    PCA space colored by C-rate, then by anomaly status.

    Two subplots side by side:
        Left  : colored by C-rate
                Shows whether different charge rates
                occupy different regions of feature space.
                Separation = C-rate has distinct thermal signature.

        Right : colored by hybrid anomaly
                Shows where anomalies sit in PCA space.
                If anomalies cluster at the edges = they're
                genuine outliers. If scattered randomly =
                anomaly scores may be noisy.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: by C-rate
    ax = axes[0]
    for c_rate in C_RATES:
        subset = df[df["c_rate"] == c_rate]
        ax.scatter(
            subset["PCA1"], subset["PCA2"],
            c=CRATE_COLORS[c_rate],
            label=c_rate, alpha=0.6, s=30
        )
    ax.set_title("PCA Space — Colored by C-Rate")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: by anomaly
    ax = axes[1]
    colors = df["hybrid_anomaly"].map({0: "#AAAAAA", 1: "#E74C3C"})
    ax.scatter(
        df["PCA1"], df["PCA2"],
        c=colors, alpha=0.6, s=30
    )
    # Legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#AAAAAA', label='Normal', markersize=8),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#E74C3C', label='Anomalous', markersize=8)
    ]
    ax.legend(handles=legend_elements)
    ax.set_title("PCA Space — Colored by Hybrid Anomaly")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    fig.suptitle("PCA Projection of Thermal Features", fontsize=13)
    plt.tight_layout()
    _save(fig, "03_pca_scatter.png")


# ─────────────────────────────────────────
# 4. HYBRID SCORE DISTRIBUTION
# ─────────────────────────────────────────

def plot_hybrid_score_distribution(df, threshold):
    """
    Histogram of hybrid scores split by C-rate.

    What to look for:
        - Bimodal distribution = clear normal/anomalous split
          = your threshold is meaningful
        - Unimodal distribution = scores are not well separated
          = consider tuning weights in config.py
        - 4C distribution shifted right vs 2C
          = higher charge stress = higher anomaly scores overall
          = physically expected

    The vertical red line shows your threshold.
    Everything to the right is flagged anomalous.
    """

    fig, axes = plt.subplots(
        1, len(C_RATES),
        figsize=(15, 4),
        sharey=True
    )

    for i, c_rate in enumerate(C_RATES):
        ax     = axes[i]
        subset = df[df["c_rate"] == c_rate]["hybrid_score"]

        ax.hist(
            subset, bins=30,
            color=CRATE_COLORS[c_rate],
            alpha=0.7, edgecolor="white"
        )
        ax.axvline(
            threshold, color="red",
            linestyle="--", linewidth=1.5,
            label=f"Threshold={threshold:.2f}"
        )
        n_anom = (subset >= threshold).sum()
        ax.set_title(
            f"{c_rate}\n{n_anom}/{len(subset)} anomalous"
        )
        ax.set_xlabel("Hybrid Score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Count")
    fig.suptitle(
        "Hybrid Anomaly Score Distribution by C-Rate\n"
        "Right of threshold = flagged anomalous",
        fontsize=13
    )
    plt.tight_layout()
    _save(fig, "04_hybrid_score_distribution.png")


# ─────────────────────────────────────────
# 5. ANOMALY RATE BY C-RATE
# ─────────────────────────────────────────

def plot_anomaly_rate_by_crate(crate_df):
    """
    Bar chart of anomaly detection rate per C-rate
    for all three methods side by side.

    This is your main results figure for the paper.

    What to look for:
        - Bars increasing left to right (2C → 4C)
          = pipeline correctly captures increasing stress
        - Hybrid bar between KMeans and AE
          = fusion is working as expected
        - All three methods agreeing on the trend
          = finding is robust across methods
    """

    methods   = ["KMeans", "Autoencoder", "Hybrid"]
    col_map   = {
        "KMeans"      : "Anomaly_Rate_%",
        "Autoencoder" : "AE_Anomaly_Rate_%",
        "Hybrid"      : "Anomaly_Rate_%"
    }

    # Rebuild from crate_df which has hybrid rates
    # We need per-method rates — pass full df instead
    # This plot is called from pipeline.py with proper data
    fig, ax = plt.subplots(figsize=(8, 5))

    x      = np.arange(len(crate_df))
    width  = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    rate_cols = [
        "KMeans_Rate_%",
        "AE_Rate_%",
        "Anomaly_Rate_%"
    ]
    labels = ["KMeans", "Autoencoder", "Hybrid"]

    for i, (col, label, color) in enumerate(
        zip(rate_cols, labels, colors)
    ):
        if col in crate_df.columns:
            ax.bar(
                x + i * width,
                crate_df[col],
                width, label=label,
                color=color, alpha=0.8
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(crate_df["C_Rate"])
    ax.set_xlabel("Charge Rate")
    ax.set_ylabel("Anomaly Detection Rate (%)")
    ax.set_title(
        "Anomaly Detection Rate by C-Rate and Method\n"
        "Higher C-rate = higher charge stress = more anomalies expected"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    _save(fig, "05_anomaly_rate_by_crate.png")


# ─────────────────────────────────────────
# 6. FEATURE HEATMAP
# ─────────────────────────────────────────

def plot_feature_heatmap(diff_df):
    """
    Heatmap of % feature difference between
    anomalous and normal images per C-rate.

    Color:
        Red   = anomalous images have higher values
        Blue  = anomalous images have lower values

    What to look for:
        - max_temp and hotspot_area consistently red
          = these are the primary anomaly drivers
        - Pattern consistent across C-rates
          = anomaly signature is stable
        - Pattern changing across C-rates
          = different stress mechanisms at different rates
          = more interesting finding
    """

    if diff_df.empty:
        print("No data for feature heatmap.")
        return

    pivot = diff_df.pivot(
        index="Feature",
        columns="C_Rate",
        values="Pct_Change"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot,
        annot=True, fmt=".1f",
        cmap="RdBu_r", center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "% difference (anomalous vs normal)"}
    )
    ax.set_title(
        "Feature Difference: Anomalous vs Normal Images (%)\n"
        "Red = anomalous images higher | Blue = lower"
    )
    plt.tight_layout()
    _save(fig, "06_feature_heatmap.png")


# ─────────────────────────────────────────
# 7. TOP ANOMALY IMAGES
# ─────────────────────────────────────────

def plot_top_anomalies(df, top_n=5):
    """
    Show the top N anomalous images per C-rate
    side by side with their hybrid score and
    key feature values.

    This is essential for the paper —
    reviewers want to see what an anomaly
    actually looks like visually.

    Layout per C-rate:
        Row of top_n images, each with:
            - filename
            - hybrid score
            - max_temp, hotspot_area
            - whether KM and AE agreed
    """

    for c_rate in C_RATES:
        subset  = df[df["c_rate"] == c_rate]
        top     = subset.nlargest(top_n, "hybrid_score")
        folder  = get_image_folder(c_rate)

        if len(top) == 0:
            print(f"No anomalies to show for {c_rate}")
            continue

        fig, axes = plt.subplots(
            1, top_n,
            figsize=(4 * top_n, 4)
        )
        if top_n == 1:
            axes = [axes]

        for i, (_, row) in enumerate(top.iterrows()):
            ax       = axes[i]
            img_path = os.path.join(folder, row["filename"])

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, RESIZE_DIM)
                ax.imshow(img, cmap="hot")
            else:
                ax.text(
                    0.5, 0.5, "Image\nnot found",
                    ha="center", va="center",
                    transform=ax.transAxes
                )

            agreement = row.get("agreement", "unknown")
            ax.set_title(
                f"{row['filename']}\n"
                f"Score: {row['hybrid_score']:.3f}\n"
                f"MaxT: {row['max_temp']:.3f} "
                f"HS: {row['hotspot_area']:.3f}\n"
                f"{agreement}",
                fontsize=7
            )
            ax.axis("off")

        fig.suptitle(
            f"{c_rate} — Top {top_n} Anomalies by Hybrid Score",
            fontsize=12
        )
        plt.tight_layout()
        _save(fig, f"07_top_anomalies_{c_rate}.png")


# ─────────────────────────────────────────
# 8. RECONSTRUCTION COMPARISON
# ─────────────────────────────────────────

def plot_reconstruction_examples(df, autoencoder, top_n=3):
    """
    Show original vs reconstructed images for:
        - Top anomalies (high reconstruction error)
        - Normal images (low reconstruction error)

    This visually explains what the autoencoder learned.

    What to look for:
        Normal images   : reconstruction nearly identical
                          to original — model learned
                          normal thermal patterns well
        Anomalous images: reconstruction is blurry or
                          misses the hotspot — model was
                          "surprised" by the pattern

    This is the most intuitive explanation of how
    your autoencoder detects anomalies.
    """

    from features import preprocess_image

    for c_rate in C_RATES:
        subset = df[df["c_rate"] == c_rate].copy()
        folder = get_image_folder(c_rate)

        # Top anomalies and top normals
        top_anom   = subset.nlargest(top_n, "reconstruction_error")
        top_normal = subset.nsmallest(top_n, "reconstruction_error")
        examples   = pd.concat([top_anom, top_normal])

        n_rows = 2   # original / reconstructed
        n_cols = top_n * 2  # anomalous | normal

        fig = plt.figure(figsize=(4 * n_cols, 5))
        gs  = gridspec.GridSpec(
            n_rows, n_cols,
            figure=fig,
            hspace=0.4, wspace=0.2
        )

        col_titles = (
            [f"Anomalous {i+1}" for i in range(top_n)] +
            [f"Normal {i+1}"    for i in range(top_n)]
        )

        for col_i, (_, row) in enumerate(examples.iterrows()):
            img_path = os.path.join(folder, row["filename"])
            img      = preprocess_image(img_path)

            # Reconstruct
            img_input = img[np.newaxis, ..., np.newaxis]
            recon     = autoencoder.predict(
                img_input, verbose=0
            )[0, :, :, 0]

            error = np.mean((img - recon) ** 2)

            # Original
            ax_orig = fig.add_subplot(gs[0, col_i])
            ax_orig.imshow(img, cmap="hot", vmin=0, vmax=1)
            ax_orig.set_title(
                f"{col_titles[col_i]}\nOriginal\nErr={error:.5f}",
                fontsize=7
            )
            ax_orig.axis("off")

            # Reconstructed
            ax_recon = fig.add_subplot(gs[1, col_i])
            ax_recon.imshow(recon, cmap="hot", vmin=0, vmax=1)
            ax_recon.set_title("Reconstructed", fontsize=7)
            ax_recon.axis("off")

        fig.suptitle(
            f"{c_rate} — Original vs Reconstructed\n"
            f"Left: anomalous (high error) | Right: normal (low error)",
            fontsize=11
        )
        _save(fig, f"08_reconstruction_{c_rate}.png")


# ─────────────────────────────────────────
# 9. AGREEMENT BREAKDOWN
# ─────────────────────────────────────────

def plot_agreement_breakdown(df):
    """
    Stacked bar chart showing agreement between
    KMeans and Autoencoder per C-rate.

    Categories:
        both_normal     : neither flagged (grey)
        both_anomalous  : both flagged (red) — highest confidence
        kmeans_only     : KMeans flagged, AE did not (blue)
        ae_only         : AE flagged, KMeans did not (orange)

    What to look for:
        - Large both_anomalous bar = methods agree = robust detection
        - Large ae_only bar = AE catching spatial anomalies
          KMeans misses = Conv AE adds real value
        - Large kmeans_only bar = statistical outliers that
          don't have unusual spatial patterns
    """

    agreement_order = [
        "both_normal",
        "both_anomalous",
        "kmeans_only",
        "ae_only"
    ]
    agreement_colors = {
        "both_normal"    : "#CCCCCC",
        "both_anomalous" : "#E74C3C",
        "kmeans_only"    : "#4C72B0",
        "ae_only"        : "#DD8452"
    }

    counts = df.groupby(
        ["c_rate", "agreement"]
    ).size().unstack(fill_value=0)

    # Ensure all columns exist
    for cat in agreement_order:
        if cat not in counts.columns:
            counts[cat] = 0
    counts = counts[agreement_order]

    fig, ax = plt.subplots(figsize=(8, 5))

    bottom = np.zeros(len(counts))
    for cat in agreement_order:
        vals = counts[cat].values
        ax.bar(
            counts.index, vals,
            bottom=bottom,
            label=cat,
            color=agreement_colors[cat],
            alpha=0.85
        )
        bottom += vals

    ax.set_xlabel("Charge Rate")
    ax.set_ylabel("Number of Images")
    ax.set_title(
        "Detection Agreement Between KMeans and Autoencoder\n"
        "Red = both agree anomalous | Grey = both agree normal"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    _save(fig, "09_agreement_breakdown.png")


# ─────────────────────────────────────────
# RUN ALL PLOTS
# ─────────────────────────────────────────

def generate_all_plots(
    df, history, autoencoder,
    crate_df, diff_df, threshold
):
    """
    Generate all plots in order.
    Called from pipeline.py after all analysis is done.
    """

    print("\n===== GENERATING PLOTS =====\n")

    plot_training_history(history)
    plot_feature_distributions(df)
    plot_pca_scatter(df)
    plot_hybrid_score_distribution(df, threshold)
    plot_anomaly_rate_by_crate(crate_df)
    plot_feature_heatmap(diff_df)
    plot_top_anomalies(df)
    plot_reconstruction_examples(df, autoencoder)
    plot_agreement_breakdown(df)

    print(f"\nAll plots saved to: {PLOTS_DIR}")