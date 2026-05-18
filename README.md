# Li-Ion Battery Thermal Anomaly Detection

Unsupervised hybrid anomaly detection on infrared thermal images of
Li-ion batteries under increasing charge stress (2C, 3C, 4C).

## Research Question
How do thermal anomaly patterns in Li-ion batteries change under
increasing charge stress (2C → 3C → 4C), and can an unsupervised
hybrid system reliably detect them?

## Approach
A modular hybrid pipeline combining two complementary methods:

- **Convolutional Autoencoder** — detects spatially unusual thermal
  patterns via reconstruction error. When the model struggles to
  reconstruct an image, that image is likely anomalous.
- **KMeans Clustering** — detects global statistical outliers in
  PCA-reduced feature space. Images that don't fit any learned
  cluster pattern are flagged.
- **Hybrid Fusion** — normalized scores from both methods are
  combined via weighted sum (AE=0.7, KMeans=0.3), producing a
  single anomaly score per image.

Ground truth labels are derived from physics-based proxy thresholds
(max temperature OR gradient above 85th percentile globally across
all C-rates), enabling quantitative evaluation without manual
annotation.

## Dataset
- 367 infrared thermal images across three charge rates
  - 2C: 180 images
  - 3C: 97 images
  - 4C: 90 images
- Images captured at 320×240 resolution, processed at 64×64
- Grayscale normalized to [0, 1]

## Features Extracted
After VIF filtering (threshold=5.0), 7 features are retained:

| Feature | Description |
|---|---|
| avg_temp | Mean pixel intensity (overall heat level) |
| max_temp | Peak pixel intensity (maximum stress point) |
| laplacian_var | Surface texture complexity |
| hotspot_area | Fraction of pixels above 95th percentile |
| entropy | Thermal distribution disorder |
| hotspot_cx | Horizontal center of mass of hotspot |
| hotspot_cy | Vertical center of mass of hotspot |

`gradient` was removed by VIF filtering (VIF=34.99) due to high
multicollinearity.

## Results

### Method Comparison
| Method | Precision | Recall | F1 |
|---|---|---|---|
| KMeans | 0.158 | 0.029 | 0.050 |
| Autoencoder | 0.790 | 0.147 | 0.248 |
| **Hybrid** | **0.553** | **0.510** | **0.531** |

The hybrid system outperforms both individual methods on F1 and
recall, validating the fusion approach.

### Per C-Rate Performance (Hybrid)
| C-Rate | Images | GT Anomalies | Detected | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 2C | 180 | 39 | 39 | 0.564 | 0.564 | 0.564 |
| 3C | 97 | 24 | 30 | 0.500 | 0.625 | 0.556 |
| 4C | 90 | 39 | 25 | 0.600 | 0.385 | 0.469 |

### Key Findings
- Hybrid F1=0.53 outperforms standalone KMeans (0.05) and
  Autoencoder (0.25)
- Anomalous images consistently show elevated `hotspot_cy`
  across all C-rates — hotspot concentrates in the lower
  portion of the battery under stress
- `laplacian_var` and `gradient` are elevated 32–64% in
  anomalous images — sharper thermal boundaries signal stress
- Autoencoder precision=0.79 — when it raises an alarm,
  it is correct 79% of the time
- Conv AE catches spatial anomalies KMeans misses:
  `ae_only` hotspot_area (0.066) > `both_normal` (0.053)
- GMM separability=1.14 indicates moderate but meaningful
  separation between normal and anomalous score distributions

### Pipeline Quality Metrics
- PCA explained variance: 69.8% (PC1=51.6%, PC2=18.2%)
- KMeans silhouette score: 0.651 (k=4)
- GMM separability: 1.14

## Pipeline Structure
src/
├── config.py        — paths, constants, hyperparameters
├── features.py      — image preprocessing + feature extraction
├── ground_truth.py  — physics-based proxy labeling
├── autoencoder.py   — convolutional autoencoder
├── clustering.py    — VIF filtering, PCA, KMeans
├── fusion.py        — hybrid score fusion + GMM separability
├── evaluate.py      — metrics + interpretation
├── visualize.py     — all plots
└── pipeline.py      — end-to-end orchestration

## Running the Pipeline
```python
# In Google Colab
from google.colab import drive
drive.mount('/content/drive')

import importlib.util, sys

src_path = '/content/drive/MyDrive/0research/battery-thermal-anomaly/src'

def load_module(name):
    spec   = importlib.util.spec_from_file_location(
        name, f"{src_path}/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

for mod in ['config','features','ground_truth','autoencoder',
            'clustering','fusion','evaluate','visualize']:
    load_module(mod)
pipeline = load_module('pipeline')

# First run — trains autoencoder
results = pipeline.run()

# Subsequent runs — loads saved model
results = pipeline.run(skip_training=True)
```

## Requirements
See `requirements.txt`

## Target Venue
IEEE Region 10 Conference (TENCON) 2026
Theme: Intelligent Systems for a Resilient and Sustainable Society

## Citation
To be added after submission.
