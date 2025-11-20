"""ROI file loading and processing."""

import yaml
from pathlib import Path
from shapely.geometry import Polygon
from typing import Dict, Optional


# ROI name mapping (flexible naming)
ROI_NAME_MAP = {
    # Standard lowercase
    "segment1": "segment1",
    "segment2": "segment2",
    "segment3": "segment3",
    "segment4": "segment4",
    "junction": "junction",
    "arm_right": "arm_right",
    "arm_left": "arm_left",
    # Capitalized variants
    "Segment1": "segment1",
    "Segment2": "segment2",
    "Segment3": "segment3",
    "Segment4": "segment4",
    "Junction": "junction",
    "ArmRight": "arm_right",
    "ArmLeft": "arm_left",
    # Alternative names
    "stem1": "segment1",
    "stem2": "segment2",
    "stem3": "segment3",
    "stem4": "segment4",
    "right_arm": "arm_right",
    "left_arm": "arm_left",
}


def load_rois(roi_path: Path, name_map: Optional[Dict[str, str]] = None) -> Dict[str, Polygon]:
    """
    Load ROI polygons from a YAML file.

    Expected YAML format:
        rois:
          - name: segment1
            coordinates: [[x1, y1], [x2, y2], ...]
          - name: segment2
            coordinates: [[x1, y1], [x2, y2], ...]

    Args:
        roi_path: Path to ROI YAML file
        name_map: Optional custom name mapping dict (default uses ROI_NAME_MAP)

    Returns:
        Dictionary mapping standardized ROI names to Shapely Polygon objects
    """
    if name_map is None:
        name_map = ROI_NAME_MAP

    with open(roi_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'rois' not in data:
        raise ValueError(f"ROI file {roi_path} missing 'rois' key")

    polygons = {}

    for roi in data['rois']:
        name = roi.get('name')
        coords = roi.get('coordinates')

        if name is None or coords is None:
            continue

        # Map to standardized name
        std_name = name_map.get(name, name)

        # Create polygon
        try:
            poly = Polygon(coords)
            polygons[std_name] = poly
        except Exception as e:
            print(f"Warning: Could not create polygon for ROI '{name}': {e}")
            continue

    return polygons


def validate_roi_coverage(
    video_polygons: Dict[str, Polygon],
    required_rois: Optional[tuple] = None
) -> bool:
    """
    Validate that required ROIs are present in the video ROI file.

    Args:
        video_polygons: Dictionary of loaded video ROI polygons
        required_rois: Tuple of required ROI names (default: at least junction + 2 segments)

    Returns:
        True if all required ROIs are present, False otherwise

    Raises:
        ValueError if validation fails with details about missing ROIs
    """
    if required_rois is None:
        # Minimum requirement: junction and at least 2 segments for calibration
        required_rois = ('junction', 'segment2', 'segment3')

    missing = [roi for roi in required_rois if roi not in video_polygons]

    if missing:
        raise ValueError(
            f"Missing required ROIs: {missing}\n"
            f"Found ROIs: {list(video_polygons.keys())}\n"
            f"Required ROIs: {required_rois}"
        )

    return True


def auto_detect_roi_file(video_path: Path) -> Optional[Path]:
    """
    Auto-detect ROI file based on video filename.

    Tries multiple naming conventions:
    - video.rois.yml
    - video.rois.yaml
    - video_rois.yml
    - video_rois.yaml

    Args:
        video_path: Path to video file

    Returns:
        Path to ROI file if found, None otherwise
    """
    candidates = [
        video_path.with_suffix('.rois.yml'),
        video_path.with_suffix('.rois.yaml'),
        video_path.with_name(f"{video_path.stem}_rois.yml"),
        video_path.with_name(f"{video_path.stem}_rois.yaml"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None
