"""T-Maze Video Undistortion Package.

This package provides tools for undistorting and rectifying T-maze videos
using camera calibration with known physical dimensions.
"""

from .undistort import TMazeUndistortionPipeline
from .models import TMazeModel, create_standard_tmaze, create_custom_tmaze
from .calibration import calibrate_camera, save_calibration, load_calibration
from .roi import load_rois

__version__ = "0.1.0"

__all__ = [
    "TMazeUndistortionPipeline",
    "TMazeModel",
    "create_standard_tmaze",
    "create_custom_tmaze",
    "calibrate_camera",
    "save_calibration",
    "load_calibration",
    "load_rois",
]
