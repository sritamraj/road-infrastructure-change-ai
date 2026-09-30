[![CI](https://github.com/sritamraj/road-infrastructure-change-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/sritamraj/road-infrastructure-change-ai/actions/workflows/ci.yml)

# Road Network Extraction & Infrastructure Change Detection Using Multi-Temporal Satellite Imagery

## Project Overview

This project is a research prototype for extracting road networks from satellite imagery and analyzing spatial changes between two image dates.

The pipeline combines deep-learning road segmentation, road-network extraction, temporal change detection, raster-to-vector GIS processing, and change-hotspot analysis.

### Research Question

> Can deep learning extract road networks from satellite imagery and identify spatial changes between two time periods?

## Setup & Testing

### Install dependencies

```bash
python -m pip install -r requirements-dev.txt
```

### Run automated tests

```bash
python -m pytest -q tests
```

The test suite validates model output shapes, model construction, the segmentation loss, and binary evaluation metrics. It does not require the research datasets or trained checkpoints.

### Compile source files

```bash
python -m compileall -q src scripts tests
```

GitHub Actions runs the compilation check and test suite automatically on pushes and pull requests to `main`.

## Methodology

```text
Satellite Imagery
       |
       v
Preprocessing
       |
       +-----------------------+
       |                       |
       v                       v
Road Segmentation       Change Detection
(U-Net baseline)        (Siamese CNN)
       |                       |
       v                       v
Road Masks T1/T2        Change Probability
       |                       |
       +-----------+-----------+
                   |
                   v
             GIS Analysis
                   |
          +--------+--------+
          |        |        |
          v        v        v
       Vector   Statistics Hotspots
       Network
          |
          v
    Visualization / Map
```

## Datasets

### DeepGlobe Road Extraction

Used for road segmentation.

* 5,000 image/mask pairs
* 1,024 × 1,024 source imagery
* Binary road masks
* 4,000 training samples
* 1,000 validation samples
* Additional medium experiment: 500 training / 100 validation

The current split is random and is not region-aware.

### LEVIR-CD

Used for the paired temporal change-detection component.

* 637 total image pairs
* 445 training pairs
* 64 validation pairs
* 128 test pairs
* 1,024 × 1,024 images

**Important:** LEVIR-CD is a building-change dataset, not a road-change dataset. Therefore, its results validate the temporal change-detection component rather than proving road-construction detection.

## Dataset Setup

The research datasets are not included in this repository because of dataset distribution and storage constraints. Download them from their official dataset sources and place the extracted files in the following local structure.

### DeepGlobe Road Extraction

Expected layout:

```text
data/raw/deepglobe/train/
+-- images/
|   +-- <sample_id>_sat.jpg
+-- masks/
    +-- <sample_id>_mask.png
```

The repository already includes reproducible split files under `data/splits/deepglobe/`:

* `train.txt` / `val.txt` - full 4,000 / 1,000 split
* `train_500.txt` / `val_100.txt` - medium experiment
* `train_small.txt` / `val_small.txt` - small smoke-test split

The segmentation configuration uses the medium split by default.

### LEVIR-CD

Expected layout:

```text
data/raw/levir_cd/
+-- train/
|   +-- A/
|   +-- B/
|   +-- label/
+-- val/
|   +-- A/
|   +-- B/
|   +-- label/
+-- test/
    +-- A/
    +-- B/
    +-- label/
```

The change-detection configuration expects this directory as `data/raw/levir_cd`.

### Dataset verification

Before running training, verify that the expected dataset directories exist locally. The automated test suite does not require these research datasets.

```bash
python -m pytest -q tests
```

Training should only be started after the corresponding dataset files and split lists are available.

## Road Segmentation

A baseline U-Net was implemented using:

* RGB satellite imagery
* Encoder-decoder architecture
* Binary segmentation
* BCE + Dice loss
* AdamW optimizer
* Horizontal and vertical augmentation

Medium experiment:

* 500 training images
* 100 validation images
* 256 × 256 input
* Batch size: 2
* 2 epochs
* Learning rate: 1e-4

The best validation IoU from this baseline experiment was **0.2007**.

Implemented metrics:

* Precision
* Recall
* Dice / F1
* Intersection over Union (IoU)

## Road Network Extraction

The predicted road mask is processed using:

1. Binary thresholding
2. Morphological cleanup
3. Skeletonization
4. Connected-component extraction
5. Line vectorization
6. GeoJSON export

Example for sample `142436`:

* Skeleton pixels: 664
* Line features: 4
* Total network length: 799.99 pixels

### GIS Limitation

The current network uses raster pixel coordinates rather than real geographic coordinates.

Therefore:

* Length is reported in pixels.
* Physical distance in meters/kilometers is not claimed.
* Geographic road density is not claimed.
* The GeoJSON is currently a pixel-space research output.

A future GIS version should use georeferenced imagery, CRS information, and raster geotransforms.

## Temporal Change Detection

A lightweight Siamese CNN was implemented.

```text
Image T1 --> Shared Encoder --+
                              |
                              v
                     Feature Difference
                              |
                              v
                       Change Head
                              ^
                              |
Image T2 --> Shared Encoder --+
```

The model predicts a binary change-probability map.

The improved experiment used:

* Shared Siamese encoder
* Weighted BCE + Dice loss
* Positive-class weight: 4.0
* 256 × 256 input
* Batch size: 2
* AdamW
* Learning rate: 1e-4
* 6 epochs

Validation threshold analysis selected **0.45** for the final held-out evaluation.

## Held-Out Change-Detection Results

The final evaluation used the 128-image LEVIR-CD test split.

| Metric    | Result |
| --------- | -----: |
| Precision | 0.3057 |
| Recall    | 0.4526 |
| Dice / F1 | 0.3649 |
| IoU       | 0.2232 |

Confusion counts:

| Quantity       |    Pixels |
| -------------- | --------: |
| True Positive  |   193,352 |
| True Negative  | 7,522,202 |
| False Positive |   439,174 |
| False Negative |   233,880 |

These results describe change detection on LEVIR-CD and should not be interpreted as road-construction accuracy.

## Change Visualization

Five held-out test examples were visualized with:

* T1 image
* T2 image
* Ground-truth change mask
* Predicted change mask
* Probability map

Outputs are stored in:

```text
outputs/change_visualizations/
```

## Road-Change GIS Analysis

A separate GIS module compares two aligned binary road masks.

It calculates:

* Added road pixels
* Removed road pixels
* Total changed pixels
* Filtered connected change components

### Controlled Synthetic Validation

Because the DeepGlobe road masks are not temporal image pairs, a synthetic T2 road mask was created from sample `142436` to validate the GIS change-analysis pipeline.

Results:

* T1 road pixels: 9,216
* T2 road pixels: 15,407
* Added road pixels: 6,191
* Removed road pixels: 0
* Changed pixels: 6,191

This is a controlled software-pipeline validation, **not evidence of real-world road construction or demolition**.

## Change Hotspots

Connected components in the change mask are converted into hotspot records.

Example:

* Hotspots detected: 1
* Area: 6,191 pixels
* Centroid: `(775.00, 170.00)`
* Size: 151 × 41 pixels

Output:

```text
outputs/metrics/142436_change_hotspots.csv
```

## Interactive GIS Visualization

The generated network map is:

```text
outputs/predictions/142436_road_network_map.html
```

The current map uses pixel-space geometry and is not a geographically positioned operational map.

## Important Outputs

```text
outputs/
├── change_visualizations/
├── metrics/
│   ├── segmentation_history.json
│   ├── change_detection_history.json
│   ├── change_detection_weighted_history.json
│   ├── change_detection_test_results.json
│   └── 142436_change_hotspots.csv
└── predictions/
    ├── 142436_pred.png
    ├── 142436_segmentation_result.png
    ├── 142436_skeleton.png
    ├── 142436_road_network.geojson
    ├── 142436_road_network_map.html
    ├── 142436_road_change.png
    ├── 142436_road_change_result.png
    ├── 142436_change_hotspots.png
    └── 142436_t2_synthetic.png
```

## Model Checkpoints

Stored in:

```text
outputs/checkpoints/
```

Current checkpoints:

```text
unet_best.pt
siamese_change_best.pt
siamese_change_weighted.pt
```

## Visual Results

### Road Segmentation

Example output from the U-Net road-segmentation pipeline:

![Road segmentation result](docs/images/142436_segmentation_result.png)

### Road-Change Analysis

Controlled comparison of the T1 road mask, synthetic T2 road mask, and detected difference:

![Road change result](docs/images/142436_road_change_result.png)

> Note: The T2 mask in this example is synthetically modified to validate the GIS change-analysis pipeline. It is not a real temporal observation.

### Change Hotspot Analysis

Connected-component analysis identifies spatial hotspots in the detected road-change mask:

![Road change hotspots](docs/images/142436_change_hotspots.png)

## Technology Stack

* Python
* PyTorch
* Torchvision
* Albumentations
* NumPy
* Pillow
* scikit-image
* SciPy
* scikit-learn
* Rasterio
* GeoPandas
* Shapely
* PyProj
* Folium
* Matplotlib
* PyYAML

## Limitations

1. The road-segmentation model is a short baseline experiment.
2. The DeepGlobe split is random rather than region-aware.
3. Road vectors currently use pixel coordinates.
4. Geographic road length cannot yet be reported.
5. LEVIR-CD contains building changes rather than road changes.
6. The synthetic road-change example is only a pipeline validation.
7. Real road-change analysis requires genuinely paired, aligned, georeferenced imagery.
8. Shadows, seasons, illumination, clouds, registration errors, and resolution differences can produce false changes.
9. This prototype is not an operational monitoring system.

## Future Work

* Obtain genuinely paired road/infrastructure change imagery.
* Use georeferenced multi-temporal satellite imagery.
* Implement region-aware dataset splitting.
* Compare U-Net with SegFormer or DeepLabV3+.
* Improve Siamese change detection.
* Add image registration and temporal normalization.
* Build a proper road graph from skeletons.
* Transform pixel coordinates into geographic coordinates.
* Calculate road length and density in physical units.
* Perform intersection and connectivity analysis.
* Expand hotspot analysis.
* Perform more extensive error analysis.

## Research Positioning

This project should be presented as:

> **A research prototype for road-network extraction and infrastructure change analysis using remote sensing and geospatial AI.**

It should not be presented as an operational system for illegal-construction detection, real-time highway monitoring, or government infrastructure surveillance.

## Project Status

### Completed

* Dataset acquisition and verification
* Dataset splitting
* U-Net road segmentation baseline
* Road prediction
* Skeletonization
* Road-network vectorization
* Network statistics
* Siamese change detection
* Weighted change-detection experiment
* Threshold analysis
* Held-out test evaluation
* Change visualization
* Road-mask difference analysis
* Synthetic temporal validation
* Change-hotspot extraction
* Interactive network-map generation
* Python syntax verification

### Main Remaining Research Gap

The main next research step is connecting the road-extraction model to **genuinely paired, georeferenced multi-temporal road/infrastructure imagery** so that detected road changes represent real spatial infrastructure changes rather than synthetic perturbations or building-change labels.
