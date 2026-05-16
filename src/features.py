# src/features.py

import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from config import (
    C_RATES, get_image_folder,
    RESIZE_DIM, BLUR_KERNEL, BLUR_SIGMA,
    HOTSPOT_PERCENTILE, ENTROPY_BINS,
    FEATURES_CSV
)

# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────

def preprocess_image(image_path, resize_dim=RESIZE_DIM):
    """
    Load, resize, blur, and normalize a grayscale image.
    Returns float32 array in [0, 1].
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.resize(image, resize_dim)
    image = cv2.GaussianBlur(image, BLUR_KERNEL, sigmaX=BLUR_SIGMA)
    image = image.astype(np.float32) / 255.0
    return image


# ─────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────

def extract_features(image):
    """
    Extract thermal features from a preprocessed image.

    Features:
        avg_temp        : mean pixel intensity (overall temperature level)
        max_temp        : max pixel intensity (peak temperature)
        gradient        : mean gradient magnitude (thermal edge sharpness)
        laplacian_var   : variance of laplacian (texture complexity)
        hotspot_area    : fraction of pixels above 95th percentile (hotspot size)
        entropy         : pixel intensity entropy (thermal distribution disorder)
        hotspot_cx      : horizontal center of mass of hotspot region (0=left, 1=right)
        hotspot_cy      : vertical center of mass of hotspot region (0=top, 1=bottom)

    Why each feature matters for battery thermal analysis:
        avg_temp      → overall heat buildup at this C-rate
        max_temp      → peak stress point, early thermal runaway signal
        gradient      → sharp thermal boundaries = uneven heat distribution
        laplacian_var → surface texture changes under stress
        hotspot_area  → how much of the battery surface is under stress
        entropy       → uniform heat = low entropy, chaotic = high entropy
        hotspot_cx/cy → WHERE on the battery the stress is concentrated
    """

    # Basic statistics
    avg_temp = float(np.mean(image))
    max_temp = float(np.max(image))

    # Gradient magnitude (Sobel)
    grad_x   = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y   = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))

    # Laplacian variance
    laplacian     = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_var = float(np.var(laplacian))

    # Hotspot area (fraction of pixels above 95th percentile)
    threshold    = np.percentile(image, HOTSPOT_PERCENTILE)
    hotspot_mask = (image >= threshold).astype(np.uint8)
    hotspot_area = float(hotspot_mask.sum() / image.size)

    # Entropy of pixel intensity distribution
    hist, _ = np.histogram(image, bins=ENTROPY_BINS, range=(0, 1), density=True)
    hist    = hist + 1e-10  # avoid log(0)
    ent     = float(scipy_entropy(hist))

    # Spatial center of mass of hotspot region
    # Think of it like finding the "eye" of a storm on a weather map
    # cx=0.5, cy=0.5 means hotspot is centered; deviations tell you where stress concentrates
    moments = cv2.moments(hotspot_mask)
    if moments["m00"] > 0:
        hotspot_cx = float(moments["m10"] / moments["m00"] / image.shape[1])
        hotspot_cy = float(moments["m01"] / moments["m00"] / image.shape[0])
    else:
        hotspot_cx = 0.5
        hotspot_cy = 0.5

    return {
        "avg_temp"     : avg_temp,
        "max_temp"     : max_temp,
        "gradient"     : gradient,
        "laplacian_var": laplacian_var,
        "hotspot_area" : hotspot_area,
        "entropy"      : ent,
        "hotspot_cx"   : hotspot_cx,
        "hotspot_cy"   : hotspot_cy,
    }


# ─────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────

def process_crate(c_rate):
    """
    Process all images for a single C-rate.
    Returns a DataFrame with filename, c_rate, and all features.
    """
    folder   = get_image_folder(c_rate)
    records  = []
    errors   = []

    image_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    print(f"\nProcessing {c_rate}: {len(image_files)} images found")

    for filename in image_files:
        path = os.path.join(folder, filename)
        try:
            image    = preprocess_image(path)
            features = extract_features(image)
            records.append({"filename": filename, "c_rate": c_rate, **features})
        except Exception as e:
            errors.append(filename)
            print(f"  Skipped {filename}: {e}")

    if errors:
        print(f"  {len(errors)} images failed to process in {c_rate}")

    df = pd.DataFrame(records)
    print(f"  Done. {len(df)} images processed successfully.")
    return df


def load_all_crates():
    """
    Process all C-rates and return one combined DataFrame.
    Also saves to FEATURES_CSV.

    The c_rate column lets you slice by charge rate at any point downstream.
    """
    dfs = []
    for c_rate in C_RATES:
        df = process_crate(c_rate)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Save
    combined.to_csv(FEATURES_CSV, index=False)
    print(f"\nAll features saved to: {FEATURES_CSV}")
    print(f"Total images: {len(combined)}")
    print(f"\nPer C-rate counts:")
    print(combined["c_rate"].value_counts().sort_index())

    return combined


# ─────────────────────────────────────────
# LOAD FLATTENED IMAGES FOR AUTOENCODER
# ─────────────────────────────────────────

def load_images_flat(c_rate):
    """
    Load and flatten all images for a C-rate.
    Used by the autoencoder which needs raw pixel arrays.
    Returns (X array, filenames list)
    """
    folder      = get_image_folder(c_rate)
    images      = []
    filenames   = []

    image_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    for filename in image_files:
        path = os.path.join(folder, filename)
        try:
            img = preprocess_image(path)
            images.append(img.flatten())
            filenames.append(filename)
        except Exception as e:
            print(f"Skipped {filename}: {e}")

    return np.array(images, dtype=np.float32), filenames


def load_all_images_flat():
    """
    Load flattened images for all C-rates.
    Returns (X array, filenames list, c_rates list)
    """
    all_X         = []
    all_filenames = []
    all_crates    = []

    for c_rate in C_RATES:
        X, filenames = load_images_flat(c_rate)
        all_X.append(X)
        all_filenames.extend(filenames)
        all_crates.extend([c_rate] * len(filenames))

    return (
        np.vstack(all_X),
        all_filenames,
        all_crates
    )


def load_images_spatial(c_rate):
    """
    Load images as (N, H, W, 1) arrays for Conv autoencoder.
    """
    folder      = get_image_folder(c_rate)
    images      = []
    filenames   = []

    image_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    for filename in image_files:
        path = os.path.join(folder, filename)
        try:
            img = preprocess_image(path)
            images.append(img[..., np.newaxis])  # add channel dim
            filenames.append(filename)
        except Exception as e:
            print(f"Skipped {filename}: {e}")

    return np.array(images, dtype=np.float32), filenames


def load_all_images_spatial():
    """
    Load spatial images for all C-rates for Conv autoencoder.
    Returns (X array shape (N,H,W,1), filenames, c_rates)
    """
    all_X         = []
    all_filenames = []
    all_crates    = []

    for c_rate in C_RATES:
        X, filenames = load_images_spatial(c_rate)
        all_X.append(X)
        all_filenames.extend(filenames)
        all_crates.extend([c_rate] * len(filenames))

    return (
        np.vstack(all_X),
        all_filenames,
        all_crates
    )