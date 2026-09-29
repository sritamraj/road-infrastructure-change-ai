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
