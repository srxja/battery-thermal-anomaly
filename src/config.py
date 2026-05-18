# src/config.py

import os

# ─────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────
BASE_DIR = "/content/drive/MyDrive/0research"
SRC_DIR  = os.path.join(BASE_DIR, "battery-thermal-anomaly/src")

# ─────────────────────────────────────────
# C-RATE CONFIGURATION
# ─────────────────────────────────────────
C_RATES = ["2C", "3C", "4C"]

def get_image_folder(c_rate):
    return os.path.join(BASE_DIR, c_rate, "images", c_rate)

# ─────────────────────────────────────────
# RESULTS (all shared, nothing per-crate)
# ─────────────────────────────────────────
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
FEATURES_DIR  = os.path.join(RESULTS_DIR, "features")
ANOMALIES_DIR = os.path.join(RESULTS_DIR, "anomalies")
METRICS_DIR   = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR     = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR    = os.path.join(RESULTS_DIR, "models")

# ─────────────────────────────────────────
# OUTPUT FILE PATHS
# ─────────────────────────────────────────
FEATURES_CSV        = os.path.join(FEATURES_DIR,  "all_crates_features.csv")
ANOMALIES_CSV       = os.path.join(ANOMALIES_DIR, "all_crates_anomalies.csv")
EVALUATION_CSV      = os.path.join(METRICS_DIR,   "evaluation_summary.csv")
AUTOENCODER_PATH    = os.path.join(MODELS_DIR,    "conv_autoencoder.keras")
KMEANS_PATH         = os.path.join(MODELS_DIR,    "kmeans.pkl")

# ─────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────
RESIZE_DIM      = (64, 64)
BLUR_KERNEL     = (5, 5)
BLUR_SIGMA      = 1

# ─────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────
HOTSPOT_PERCENTILE  = 95    # top 5% pixels = hotspot
ENTROPY_BINS        = 256   # histogram bins for entropy

# ─────────────────────────────────────────
# GROUND TRUTH (physics-based thresholds)
# ─────────────────────────────────────────
GT_MAX_TEMP_PERCENTILE  = 85   # top 5% max_temp values = anomalous
GT_GRADIENT_PERCENTILE  = 85   # top 10% gradient values = anomalous

# ─────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────
PCA_COMPONENTS      = 2
KMEANS_K_RANGE      = range(2, 8)
KMEANS_RANDOM_STATE = 42
VIF_THRESHOLD       = 5.0


# AUTOENCODER

ENCODING_DIM        = 64
EPOCHS              = 100
BATCH_SIZE          = 32
VALIDATION_SPLIT    = 0.1
EARLY_STOPPING_PAT  = 10

# ─────────────────────────────────────────
# ANOMALY SCORING
# ─────────────────────────────────────────
ANOMALY_PERCENTILE  = 95    # top 5% = anomalous
HYBRID_THRESHOLD    = 0.6
HYBRID_W_RECON      = 0.6   # weight for autoencoder score
HYBRID_W_KMEANS     = 0.4   # weight for kmeans score

# ─────────────────────────────────────────
# CREATE ALL DIRS ON IMPORT
# ─────────────────────────────────────────
for _dir in [FEATURES_DIR, ANOMALIES_DIR, METRICS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)