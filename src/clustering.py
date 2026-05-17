# src/clustering.py

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from config import (
    PCA_COMPONENTS, KMEANS_K_RANGE, KMEANS_RANDOM_STATE,
    VIF_THRESHOLD, ANOMALY_PERCENTILE, KMEANS_PATH
)

# ─────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────

FEATURE_COLS = [
    "avg_temp",
    "max_temp",
    "gradient",
    "laplacian_var",
    "hotspot_area",
    "entropy",
    "hotspot_cx",
    "hotspot_cy"
]

# ─────────────────────────────────────────
# VIF FILTERING
# ─────────────────────────────────────────

def remove_multicollinear_features(df, feature_cols=FEATURE_COLS):
    """
    Iteratively remove features with VIF above threshold.

    Why VIF matters:
        VIF (Variance Inflation Factor) measures how much one
        feature can be predicted from the others.
        High VIF = redundant feature = feeding PCA correlated
        inputs which distorts the principal components.

        Think of it like a group project where two people
        do the exact same work — one of them is redundant.
        VIF finds and removes the redundant one.

        VIF < 5  : acceptable
        VIF 5-10 : moderate concern
        VIF > 10 : definitely remove

    Args:
        df          : features DataFrame
        feature_cols: list of column names to check

    Returns:
        selected_cols : list of columns that survived
        vif_table     : final VIF table (for paper)
    """

    X            = df[feature_cols].copy()
    removed      = []

    print("===== VIF FILTERING =====")

    while True:
        X_const  = add_constant(X)
        vif_vals = [
            variance_inflation_factor(X_const.values, i)
            for i in range(X_const.shape[1])
        ]
        vif_df   = pd.DataFrame({
            "feature": X_const.columns,
            "VIF"    : vif_vals
        }).query("feature != 'const'").reset_index(drop=True)

        max_vif  = vif_df["VIF"].max()

        if max_vif > VIF_THRESHOLD:
            worst = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]
            print(f"  Removing '{worst}' (VIF={max_vif:.2f})")
            removed.append(worst)
            X.drop(columns=[worst], inplace=True)
        else:
            break

    print(f"\nFeatures removed : {removed if removed else 'none'}")
    print(f"Features kept    : {list(X.columns)}")
    print(f"\nFinal VIF table:")
    print(vif_df.to_string(index=False))

    return list(X.columns), vif_df


# ─────────────────────────────────────────
# PCA
# ─────────────────────────────────────────

def apply_pca(df, selected_cols):
    """
    Standardize features and apply PCA.

    Why standardize before PCA?
        PCA finds directions of maximum variance.
        If avg_temp is in range [0.4, 0.5] and entropy
        is in range [0, 8], entropy dominates purely
        due to scale — not because it's more important.
        Standardizing puts everyone on equal footing.

        Like converting all currencies to USD before
        comparing prices across countries.

    Why PCA?
        We reduce 8 features → 2 principal components
        so we can visualize and cluster in 2D.
        The components capture the directions of maximum
        variance — the axes along which batteries differ most.

    Args:
        df            : features DataFrame
        selected_cols : VIF-filtered feature columns

    Returns:
        df_pca    : DataFrame with PCA1, PCA2 added
        pca       : fitted PCA object
        scaler    : fitted StandardScaler
        explained : explained variance ratios
    """

    X      = df[selected_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca    = PCA(n_components=PCA_COMPONENTS, random_state=KMEANS_RANDOM_STATE)
    X_pca  = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    print(f"\n===== PCA =====")
    print(f"Components      : {PCA_COMPONENTS}")
    print(f"Explained var   : PC1={explained[0]:.3f}, PC2={explained[1]:.3f}")
    print(f"Total explained : {sum(explained):.3f}")

    # What does each PC represent?
    # We look at which original features load most strongly
    print(f"\nPC loadings (what each component captures):")
    loadings = pd.DataFrame(
        pca.components_.T,
        index=selected_cols,
        columns=[f"PC{i+1}" for i in range(PCA_COMPONENTS)]
    )
    print(loadings.round(3).to_string())

    df_pca         = df.copy()
    df_pca["PCA1"] = X_pca[:, 0]
    df_pca["PCA2"] = X_pca[:, 1]

    return df_pca, pca, scaler, explained


# ─────────────────────────────────────────
# OPTIMAL K SELECTION
# ─────────────────────────────────────────

def find_optimal_k(df_pca):
    """
    Find best K for KMeans using silhouette score.

    Why silhouette score over elbow method?
        Elbow method requires visual inspection — subjective.
        Silhouette score is objective: it measures how similar
        each point is to its own cluster vs other clusters.

        Score ranges from -1 to 1:
            ~1  : point is well inside its cluster
            ~0  : point is on the boundary
            < 0 : point may be in the wrong cluster

        We pick K that maximizes average silhouette score.

        Think of it like measuring how well students are grouped
        by ability — a good grouping has students clearly
        stronger or weaker than those in other groups.

    Args:
        df_pca: DataFrame with PCA1, PCA2 columns

    Returns:
        best_k      : optimal number of clusters
        k_results   : DataFrame of k vs silhouette score
    """

    pts      = df_pca[["PCA1", "PCA2"]].values
    results  = []

    print("\n===== K SELECTION =====")

    for k in KMEANS_K_RANGE:
        km     = KMeans(
            n_clusters=k,
            random_state=KMEANS_RANDOM_STATE,
            n_init=10
        )
        labels = km.fit_predict(pts)
        sil    = silhouette_score(pts, labels)
        results.append({"k": k, "silhouette": sil})
        print(f"  k={k}: silhouette={sil:.4f}")

    k_results = pd.DataFrame(results)
    best_k    = int(k_results.loc[k_results["silhouette"].idxmax(), "k"])
    print(f"\nBest k: {best_k} (silhouette={k_results.loc[k_results['silhouette'].idxmax(), 'silhouette']:.4f})")

    return best_k, k_results


# ─────────────────────────────────────────
# KMEANS CLUSTERING
# ─────────────────────────────────────────

def apply_kmeans(df_pca, k):
    """
    Fit KMeans and compute distance-based anomaly scores.

    Distance to cluster center as anomaly score:
        Points far from any cluster center are outliers —
        they don't fit well into any learned grouping.

        Like a student whose test scores don't match any
        known performance pattern — that's the anomaly.

    Args:
        df_pca : DataFrame with PCA1, PCA2
        k      : number of clusters

    Returns:
        df_clustered : DataFrame with cluster labels
                       and kmeans anomaly scores
        kmeans       : fitted KMeans object
    """

    pts    = df_pca[["PCA1", "PCA2"]].values
    kmeans = KMeans(
        n_clusters=k,
        random_state=KMEANS_RANDOM_STATE,
        n_init=10
    )
    labels    = kmeans.fit_predict(pts)
    centers   = kmeans.cluster_centers_

    # Distance from each point to its assigned cluster center
    distances = np.linalg.norm(
        pts - centers[labels], axis=1
    )

    threshold = np.percentile(distances, ANOMALY_PERCENTILE)

    df_clustered                    = df_pca.copy()
    df_clustered["km_cluster"]      = labels
    df_clustered["km_score"]        = distances
    df_clustered["km_anomaly"]      = (distances >= threshold).astype(int)

    # Save model
    joblib.dump(kmeans, KMEANS_PATH)
    print(f"\nKMeans model saved to: {KMEANS_PATH}")

    # Summary
    sil = silhouette_score(pts, labels)
    print(f"\n===== KMEANS RESULTS =====")
    print(f"K                : {k}")
    print(f"Silhouette score : {sil:.4f}")
    print(f"Anomaly threshold: {threshold:.4f}")
    print(f"\nCluster sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for cl, ct in zip(unique, counts):
        print(f"  Cluster {cl}: {ct} images")

    print(f"\nKMeans anomaly counts per C-rate:")
    for c_rate, group in df_clustered.groupby("c_rate"):
        n = group["km_anomaly"].sum()
        print(f"  {c_rate}: {n} / {len(group)} ({100*n/len(group):.1f}%)")

    return df_clustered, kmeans