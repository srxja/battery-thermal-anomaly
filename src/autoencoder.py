# src/autoencoder.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    BatchNormalization, Activation
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config import (
    RESIZE_DIM, ENCODING_DIM, EPOCHS, BATCH_SIZE,
    VALIDATION_SPLIT, EARLY_STOPPING_PAT,
    ANOMALY_PERCENTILE, AUTOENCODER_PATH
)

# ─────────────────────────────────────────
# ARCHITECTURE
# ─────────────────────────────────────────

def build_conv_autoencoder(input_shape):
    """
    Convolutional Autoencoder for thermal image reconstruction.

    Why Conv instead of FC (fully connected)?
        FC autoencoders treat every pixel independently.
        Conv autoencoders look at patches — they learn
        "this region runs hotter than its neighbors"
        which is exactly what a thermal anomaly is.

        Think of FC as reading a map by listing every
        coordinate's value. Conv reads it like a human —
        by recognizing regions and patterns.

    Architecture:
        Encoder: progressively compress spatial information
            64x64x1 → 32x32x32 → 16x16x64 → 8x8x128

        Decoder: reconstruct from compressed representation
            8x8x128 → 16x16x64 → 32x32x32 → 64x64x1

        BatchNormalization after each conv:
            Stabilizes training, especially important with
            small datasets like ours (~500 images total)

    Reconstruction error as anomaly score:
        Normal images → autoencoder reconstructs them well → low error
        Anomalous images → autoencoder struggles → high error

        Like a student who studied normal exam questions —
        they do fine on familiar problems but struggle on
        the unexpected ones.

    Args:
        input_shape: tuple (H, W, 1) — grayscale image shape

    Returns:
        autoencoder: full Model (encoder + decoder)
        encoder:     just the encoder half (for latent space analysis)
    """

    inputs = Input(shape=input_shape, name="input")

    # ── Encoder ──────────────────────────────
    x = Conv2D(32, (3, 3), padding="same", name="enc_conv1")(inputs)
    x = BatchNormalization(name="enc_bn1")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), name="enc_pool1")(x)          # 64→32

    x = Conv2D(64, (3, 3), padding="same", name="enc_conv2")(x)
    x = BatchNormalization(name="enc_bn2")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), name="enc_pool2")(x)          # 32→16

    x = Conv2D(128, (3, 3), padding="same", name="enc_conv3")(x)
    x = BatchNormalization(name="enc_bn3")(x)
    encoded = Activation("relu", name="encoded")(x)
    # shape here: (8, 8, 128) — the compressed representation
    # MaxPooling intentionally omitted at this stage to preserve
    # enough spatial detail for clean reconstruction

    # ── Decoder ──────────────────────────────
    x = Conv2D(128, (3, 3), padding="same", name="dec_conv1")(encoded)
    x = BatchNormalization(name="dec_bn1")(x)
    x = Activation("relu")(x)
    x = UpSampling2D((2, 2), name="dec_up1")(x)            # 16→32

    x = Conv2D(64, (3, 3), padding="same", name="dec_conv2")(x)
    x = BatchNormalization(name="dec_bn2")(x)
    x = Activation("relu")(x)
    x = UpSampling2D((2, 2), name="dec_up2")(x)            # 32→64

    x = Conv2D(32, (3, 3), padding="same", name="dec_conv3")(x)
    x = BatchNormalization(name="dec_bn3")(x)
    x = Activation("relu")(x)
    # no third upsample — already at 64×64

    decoded = Conv2D(
        1, (3, 3),
        padding="same",
        activation="sigmoid",
        name="output"
    )(x)
    # sigmoid keeps output in [0,1] matching our normalized input

    autoencoder = Model(inputs, decoded, name="conv_autoencoder")
    encoder     = Model(inputs, encoded, name="encoder")

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    return autoencoder, encoder


# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────

def train_autoencoder(X_train):
    """
    Train the Conv autoencoder on all images (all C-rates combined).

    Why train on all C-rates together?
        We want the autoencoder to learn what a "normal"
        thermal image looks like across all charge conditions.
        If we trained only on 2C, it would flag all 4C images
        as anomalous simply because they're hotter — not because
        they're actually anomalous within their own distribution.

    Callbacks:
        EarlyStopping     : stops if val_loss doesn't improve
        ReduceLROnPlateau : halves learning rate if training stalls
                           (like easing off the gas when you're
                            not making progress on a hill)

    Args:
        X_train: array of shape (N, H, W, 1)

    Returns:
        autoencoder : trained Model
        encoder     : trained encoder half
        history     : training history for plotting
    """

    input_shape            = X_train.shape[1:]  # (H, W, 1)
    autoencoder, encoder   = build_conv_autoencoder(input_shape)

    print("===== AUTOENCODER ARCHITECTURE =====")
    autoencoder.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PAT,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n===== TRAINING =====")
    history = autoencoder.fit(
        X_train, X_train,           # input = target (reconstruction)
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    # Save model
    autoencoder.save(AUTOENCODER_PATH)
    print(f"\nAutoencoder saved to: {AUTOENCODER_PATH}")

    return autoencoder, encoder, history


# ─────────────────────────────────────────
# ANOMALY SCORING
# ─────────────────────────────────────────

def compute_reconstruction_scores(autoencoder, X, filenames, c_rates):
    """
    Compute per-image reconstruction error.

    Reconstruction error = mean squared difference between
    input image and autoencoder's reconstruction of it.

    High error = the autoencoder was surprised by this image
               = likely anomalous thermal pattern

    Args:
        autoencoder : trained Model
        X           : array (N, H, W, 1)
        filenames   : list of filenames
        c_rates     : list of c_rate labels

    Returns:
        df_scores: DataFrame with filename, c_rate,
                   reconstruction_error, ae_anomaly
    """

    print("Computing reconstruction errors...")
    reconstructions = autoencoder.predict(X, verbose=0)

    # Per-image MSE
    errors = np.mean(
        (X - reconstructions) ** 2,
        axis=(1, 2, 3)          # mean over H, W, channels
    )

    threshold = np.percentile(errors, ANOMALY_PERCENTILE)
    print(f"Reconstruction error threshold (p{ANOMALY_PERCENTILE}): {threshold:.6f}")

    df_scores = pd.DataFrame({
        "filename"              : filenames,
        "c_rate"                : c_rates,
        "reconstruction_error"  : errors,
        "ae_anomaly"            : (errors >= threshold).astype(int)
    })

    # Summary
    print("\nAutoencoder anomaly counts per C-rate:")
    for c_rate, group in df_scores.groupby("c_rate"):
        n = group["ae_anomaly"].sum()
        print(f"  {c_rate}: {n} / {len(group)} ({100*n/len(group):.1f}%)")

    return df_scores, threshold


# ─────────────────────────────────────────
# LOAD SAVED MODEL
# ─────────────────────────────────────────

def load_autoencoder():
    """Load a previously saved autoencoder from disk."""
    print(f"Loading autoencoder from: {AUTOENCODER_PATH}")
    return tf.keras.models.load_model(AUTOENCODER_PATH)