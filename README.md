# T-Maze Video Undistortion Pipeline

A Python package for undistorting and rectifying T-maze videos using camera calibration with known physical dimensions.

## Features

- **Automatic Camera Calibration**: Uses labeled ROI regions and known physical maze dimensions to calibrate camera intrinsics
- **Lens Distortion Correction**: Removes radial and tangential distortion from fisheye/wide-angle lenses
- **Perspective Rectification**: Transforms video to top-down view aligned with physical maze coordinates
- **Flexible Maze Models**: Supports custom T-maze dimensions and orientations
- **Calibration Reuse**: Save and load calibration parameters for processing multiple videos
- **Masking & Cropping**: Optionally mask background and crop to maze bounds

## Installation

```bash
# Clone the repository
git clone https://github.com/LeoMeow123/tmaze-undistort.git
cd tmaze-undistort

# Install in editable mode
pip install -e .
```

## Quick Start

### 0. Test if Undistortion is Needed (Optional)

If your videos contain a charuco board, you can test whether undistortion is necessary:

```bash
# Test a single video
python 02.test_distortion_need.py video.mp4

# Test all videos in a directory (automatically saves CSV, JSON, and log files)
python 02.test_distortion_need.py /path/to/videos/ --batch

# Customize output file names
python 02.test_distortion_need.py /path/to/videos/ --batch \
    --output-csv results.csv \
    --output-json results.json \
    --output-log summary.txt

# Customize charuco board parameters (defaults: 14x9, 20mm squares, 15mm markers, DICT_5X5)
python 02.test_distortion_need.py video.mp4 \
    --squares-x 14 --squares-y 9 \
    --square-length 0.02 --marker-length 0.015 \
    --dictionary DICT_5X5_250
```

The tool analyzes the charuco board geometry and provides three metrics:
- **Line Straightness**: Measures edge curvature (< 2.0 px is good)
- **Spacing Uniformity**: Measures corner spacing consistency (< 0.05 is good)
- **Reprojection Error**: Measures fit to pinhole model (< 5.0 px is good)

**Batch mode automatically generates:**
- `distortion_test_results_<timestamp>.csv` - Spreadsheet with all metrics
- `distortion_test_results_<timestamp>.json` - Machine-readable JSON format
- `distortion_test_log_<timestamp>.txt` - Human-readable summary log

Output example:
```
============================================================
Distortion Analysis for: video_001.mp4
============================================================
Frames analyzed: 10/10

Metrics:
  Line straightness:      1.234 px (threshold: 2.0 px)
  Spacing uniformity:     0.0321 (threshold: 0.05)
  Reprojection error:     2.456 px (threshold: 5.0 px)

============================================================
Recommendation: NO UNDISTORTION NEEDED
============================================================
```

### 1. Label ROI Regions

Use a ROI labeling tool (e.g., [labelroi](https://github.com/talmolab/labelroi)) to mark the maze segments in your video. The tool will generate a `.rois.yml` file.

Required ROI labels:
- `segment1`, `segment2`, `segment3`, `segment4` (stem segments)
- `junction` (center T junction)
- `arm_left`, `arm_right` (T-maze arms)

Example ROI file structure:
```yaml
rois:
  - name: segment2
    coordinates: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  - name: junction
    coordinates: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  # ... more segments
```

### 2. Undistort Video

```bash
# Basic usage (auto-detects ROI file)
tmaze-undistort video.mp4 -o undistorted.mp4

# Specify ROI file explicitly
tmaze-undistort video.mp4 --roi video.rois.yml -o undistorted.mp4

# With background masking
tmaze-undistort video.mp4 -o undistorted.mp4 --mask

# With masking and cropping (changes output resolution)
tmaze-undistort video.mp4 -o undistorted.mp4 --mask --crop

# Save calibration for reuse
tmaze-undistort video.mp4 -o undistorted.mp4 --save-calibration calib.yml

# Reuse existing calibration on another video (same camera/setup)
tmaze-undistort video2.mp4 -o undistorted2.mp4 --calibration calib.yml
```

## Python API

```python
from tmaze_undistort import TMazeUndistortionPipeline, create_standard_tmaze

# Initialize pipeline
pipeline = TMazeUndistortionPipeline(
    video_path="video.mp4",
    roi_path="video.rois.yml",  # Optional, auto-detected if None
)

# Calibrate camera
pipeline.calibrate(save_calibration_path="calibration.yml")

# Undistort video
pipeline.undistort_video("undistorted.mp4")
```

### Custom Maze Dimensions

```python
from tmaze_undistort import TMazeUndistortionPipeline, create_custom_tmaze

# Create custom maze model
maze_model = create_custom_tmaze(
    maze_width=0.12,        # 12 cm corridor width
    segment_length=0.25,    # 25 cm per segment
    arm_length=0.30,        # 30 cm arms
    wall_width=0.005        # 0.5 cm walls
)

# Use custom model in pipeline
pipeline = TMazeUndistortionPipeline(
    video_path="video.mp4",
    maze_model=maze_model
)
```

## Command-Line Options

### Basic Options
- `video`: Input video file path
- `-o, --output`: Output video file path (required)
- `--roi`: ROI YAML file (auto-detected if not provided)
- `--calibration`: Load existing calibration file
- `--save-calibration`: Save calibration for reuse

### Maze Model Options
- `--maze-width`: Internal corridor width in meters (default: 0.10)
- `--segment-length`: Length of each stem segment in meters (default: 0.20)
- `--arm-length`: Length of each arm in meters (default: 0.275)
- `--wall-width`: Wall thickness in meters (default: 0.005)

### Calibration Options
- `--free-principal-point`: Allow principal point to vary
- `--free-aspect-ratio`: Allow aspect ratio to vary
- `--no-tangential`: Disable tangential distortion estimation

### Output Options
- `--mask`: Apply mask to remove background outside maze (default: disabled)
- `--crop`: Crop output to maze bounds, changes resolution (default: disabled)
- `--interpolation`: Interpolation method (linear, cubic, lanczos4, default: linear)
- `--no-progress`: Disable progress bar

## How It Works

1. **ROI Loading**: Loads labeled ROI regions from YAML file
2. **Model Creation**: Generates physical maze model with known dimensions
3. **Point Correspondence**: Matches ROI centroids between video and model
4. **Camera Calibration**: Uses OpenCV to estimate camera intrinsics (K, distortion)
5. **Undistortion**: Removes lens distortion using calibrated parameters
6. **Homography**: Computes transformation to rectified top-down view
7. **Video Processing**: Applies undistortion and rectification to all frames

## Physical Coordinate System

- **Origin**: Center of segment 1 (bottom segment)
- **X-axis**: Horizontal (left arm = negative X, right arm = positive X)
- **Y-axis**: Vertical pointing up (toward junction and arms)
- **Units**: Meters

## Requirements

- Python ≥ 3.9
- OpenCV ≥ 4.5.0
- NumPy ≥ 1.20.0
- Shapely ≥ 2.0.0
- sleap-io ≥ 0.5.0
- PyYAML ≥ 5.4.0

See `pyproject.toml` for full dependency list.

## Examples

### Different Maze Orientations

The pipeline automatically handles different maze orientations as long as the ROI labels are consistent with the physical model.

```bash
# Maze rotated 90 degrees
tmaze-undistort rotated_video.mp4 -o undistorted.mp4

# Maze at different scale
tmaze-undistort scaled_video.mp4 -o undistorted.mp4 --maze-width 0.15
```

### Batch Processing

```bash
# Test which videos need undistortion
python 02.test_distortion_need.py /path/to/videos/ --batch

# Process multiple videos with same calibration
tmaze-undistort video1.mp4 -o out1.mp4 --save-calibration calib.yml
tmaze-undistort video2.mp4 -o out2.mp4 --calibration calib.yml
tmaze-undistort video3.mp4 -o out3.mp4 --calibration calib.yml
```

## Troubleshooting

### "ROI file not found"
- Ensure ROI file exists and matches video filename (e.g., `video.rois.yml`)
- Or specify explicitly with `--roi` flag

### "Missing required ROIs"
- Make sure you labeled at least: `junction`, `segment2`, `segment3`
- ROI names must match expected names (case-insensitive)

### High RMS calibration error
- Check that ROI labels accurately outline maze segments
- Ensure physical dimensions match your actual maze
- Try adding more ROI segments for better calibration

### Distorted output
- Verify maze dimensions are correct
- Check that all ROI segments are properly labeled
- Consider adjusting calibration flags (e.g., `--free-principal-point`)

## License

MIT License

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{tmaze_undistort,
  author = {Leo},
  title = {T-Maze Video Undistortion Pipeline},
  year = {2025},
  url = {https://github.com/LeoMeow123/tmaze-undistort}
}
```

## Related Projects

- [spacecage-undistort](https://github.com/talmolab/spacecage-undistort): Video undistortion for NASA SpaceCage experiments
- [labelroi](https://github.com/talmolab/labelroi): Interactive ROI labeling tool

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For questions or issues, please open an issue on GitHub:
https://github.com/LeoMeow123/tmaze-undistort/issues