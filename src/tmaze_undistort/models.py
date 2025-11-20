"""Physical maze models and configurations."""

import numpy as np
from shapely.geometry import Polygon
from typing import Dict, Tuple, Optional


class TMazeModel:
    """
    Physical model of a T-maze with known dimensions.

    This class defines the ground-truth geometry of the maze in meters,
    which will be matched to video ROI labels for calibration.
    """

    def __init__(
        self,
        wall_width: float = 0.005,  # 0.5 cm
        maze_width: float = 0.10,   # 10 cm
        segment_length: float = 0.20,  # 20 cm
        arm_length: float = 0.275,  # 27.5 cm (calculated from total arms)
        origin: str = "segment1_center"
    ):
        """
        Initialize T-maze physical model.

        Args:
            wall_width: Width of maze walls in meters
            maze_width: Internal width of maze corridor in meters
            segment_length: Length of each stem segment in meters
            arm_length: Length of each arm (from junction edge to end) in meters
            origin: Reference point for coordinate system:
                - "segment1_center": Origin at center of segment 1 (bottom segment)
                - "junction_center": Origin at center of junction
        """
        self.wall_width = wall_width
        self.maze_width = maze_width
        self.segment_length = segment_length
        self.arm_length = arm_length
        self.origin = origin

    def get_roi_polygons(self, segments: Tuple[str, ...] = None) -> Dict[str, Polygon]:
        """
        Generate model ROI polygons in physical coordinates (meters).

        Args:
            segments: Tuple of segment names to include. If None, includes all:
                ('segment1', 'segment2', 'segment3', 'segment4', 'junction', 'arm_right', 'arm_left')

        Returns:
            Dictionary mapping ROI names to Shapely Polygon objects (in meters, Y-up coordinates)
        """
        if segments is None:
            segments = ('segment1', 'segment2', 'segment3', 'segment4',
                       'junction', 'arm_right', 'arm_left')

        w = self.maze_width
        sl = self.segment_length
        al = self.arm_length

        polygons = {}

        # Define all segments (origin at segment1 center, Y-up)
        # Each segment is a rectangle with 4 corners in CCW order

        # Segment 1 (bottom segment) - centered on origin
        if 'segment1' in segments:
            polygons['segment1'] = Polygon([
                (w/2, -sl/2),
                (-w/2, -sl/2),
                (-w/2, sl/2),
                (w/2, sl/2)
            ])

        # Segment 2 (second from bottom)
        if 'segment2' in segments:
            polygons['segment2'] = Polygon([
                (w/2, sl/2),
                (-w/2, sl/2),
                (-w/2, 3*sl/2),
                (w/2, 3*sl/2)
            ])

        # Segment 3 (third from bottom)
        if 'segment3' in segments:
            polygons['segment3'] = Polygon([
                (w/2, 3*sl/2),
                (-w/2, 3*sl/2),
                (-w/2, 5*sl/2),
                (w/2, 5*sl/2)
            ])

        # Segment 4 (fourth from bottom)
        if 'segment4' in segments:
            polygons['segment4'] = Polygon([
                (w/2, 5*sl/2),
                (-w/2, 5*sl/2),
                (-w/2, 7*sl/2),
                (w/2, 7*sl/2)
            ])

        # Junction (center of T)
        if 'junction' in segments:
            polygons['junction'] = Polygon([
                (w/2, 7*sl/2),
                (-w/2, 7*sl/2),
                (-w/2, 7*sl/2 + w),
                (w/2, 7*sl/2 + w)
            ])

        # Arm Right (extends in negative X direction from junction)
        if 'arm_right' in segments:
            polygons['arm_right'] = Polygon([
                (-w/2, 7*sl/2),
                (-w/2 - al, 7*sl/2),
                (-w/2 - al, 7*sl/2 + w),
                (-w/2, 7*sl/2 + w)
            ])

        # Arm Left (extends in positive X direction from junction)
        if 'arm_left' in segments:
            polygons['arm_left'] = Polygon([
                (w/2 + al, 7*sl/2),
                (w/2, 7*sl/2),
                (w/2, 7*sl/2 + w),
                (w/2 + al, 7*sl/2 + w)
            ])

        return polygons

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box of the entire maze.

        Returns:
            (xmin, xmax, ymin, ymax) in meters
        """
        polygons = self.get_roi_polygons()
        all_coords = np.concatenate([
            np.asarray(p.exterior.coords) for p in polygons.values()
        ], axis=0)

        xmin, ymin = all_coords.min(axis=0)
        xmax, ymax = all_coords.max(axis=0)

        return xmin, xmax, ymin, ymax


def create_standard_tmaze() -> TMazeModel:
    """
    Create a standard T-maze model with typical dimensions.

    Dimensions based on common T-maze specifications:
    - Wall width: 0.5 cm
    - Maze width: 10 cm
    - Segment length: 20 cm
    - Total arms length: 65 cm (-> 27.5 cm per arm after junction)

    Returns:
        TMazeModel instance with standard dimensions
    """
    return TMazeModel(
        wall_width=0.005,
        maze_width=0.10,
        segment_length=0.20,
        arm_length=0.275
    )


def create_custom_tmaze(
    maze_width: float,
    segment_length: float,
    arm_length: float,
    wall_width: float = 0.005
) -> TMazeModel:
    """
    Create a custom T-maze model with specified dimensions.

    Args:
        maze_width: Internal width of maze corridor in meters
        segment_length: Length of each stem segment in meters
        arm_length: Length of each arm in meters
        wall_width: Width of maze walls in meters

    Returns:
        TMazeModel instance with custom dimensions
    """
    return TMazeModel(
        wall_width=wall_width,
        maze_width=maze_width,
        segment_length=segment_length,
        arm_length=arm_length
    )
