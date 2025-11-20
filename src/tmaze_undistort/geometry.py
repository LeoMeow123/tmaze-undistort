"""Geometric utilities for polygon processing."""

import numpy as np
from shapely.geometry import Polygon
from typing import Dict


def _quad_from_poly(poly: Polygon) -> np.ndarray:
    """
    Extract 4 corner vertices from a polygon.

    - If polygon already has 4 unique vertices -> use them
    - Else -> use minimum rotated rectangle (always 4 corners)

    Args:
        poly: Input Shapely Polygon

    Returns:
        Array of shape (4, 2) containing corner coordinates
    """
    coords = np.asarray(poly.exterior.coords)[:-1]  # drop closing duplicate

    # Remove near-duplicates (sometimes ROI tools repeat vertices)
    if len(coords) > 1:
        uniq = [coords[0]]
        for p in coords[1:]:
            if np.linalg.norm(p - uniq[-1]) > 1e-6:
                uniq.append(p)
        coords = np.array(uniq)

    # Already a quad?
    if coords.shape[0] == 4:
        return coords.astype(float)

    # Fallback: use minimum rotated rectangle
    mrr = poly.minimum_rotated_rectangle
    rect = np.asarray(mrr.exterior.coords)[:-1]
    assert rect.shape[0] == 4, "Minimum rotated rectangle did not produce 4 vertices"
    return rect.astype(float)


def _canonical_order_ccw(quads: np.ndarray, y_up: bool) -> np.ndarray:
    """
    Reorder 4 vertices to be in canonical CCW order starting at top-left.

    Args:
        quads: Array of shape (4, 2) containing quad vertices
        y_up: If True, Y-axis points up (model coords). If False, Y points down (image coords)

    Returns:
        Reordered vertices in canonical CCW order
    """
    q = quads.copy()
    c = q.mean(axis=0)  # centroid

    # Calculate angles for CCW ordering
    # For y_down, invert Y so angles behave like y_up
    dy = (q[:, 1] - c[1]) * (1.0 if y_up else -1.0)
    dx = (q[:, 0] - c[0])
    ang = np.arctan2(dy, dx)
    order = np.argsort(ang)  # CCW order
    q = q[order]

    # Start at top-left per convention
    if y_up:
        # Y-up: top = max Y, then min X
        start = np.lexsort((q[:, 0], -q[:, 1]))[0]
    else:
        # Y-down: top = min Y, then min X
        start = np.lexsort((q[:, 0], q[:, 1]))[0]

    q = np.roll(q, -start, axis=0)
    return q


def corners_canonical(poly: Polygon, y_up: bool) -> np.ndarray:
    """
    Extract quad corners in canonical CCW order starting at top-left.

    Args:
        poly: Input Shapely Polygon
        y_up: Coordinate system convention (True for Y-up, False for Y-down)

    Returns:
        Array of shape (4, 2) with canonically ordered corners
    """
    q = _quad_from_poly(poly)
    return _canonical_order_ccw(q, y_up=y_up)


def extract_canonical_corners(
    model_polygons: Dict[str, Polygon],
    video_polygons: Dict[str, Polygon]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract matching corner correspondences from model and video ROI polygons.

    Args:
        model_polygons: Dict of model ROI polygons (meters, Y-up)
        video_polygons: Dict of video ROI polygons (pixels, Y-down)

    Returns:
        Tuple of (model_points, image_points):
            - model_points: (N×4, 2) array of model corners in meters
            - image_points: (N×4, 2) array of image corners in pixels
    """
    # Get keys present in both
    roi_keys = [k for k in model_polygons.keys() if k in video_polygons]

    model_quads = {}
    video_quads = {}

    for k in roi_keys:
        m_poly = model_polygons[k]
        v_poly = video_polygons[k]

        m_quad = corners_canonical(m_poly, y_up=True)   # model is Y-up
        v_quad = corners_canonical(v_poly, y_up=False)  # video pixels are Y-down

        model_quads[k] = m_quad
        video_quads[k] = v_quad

    # Stack all corners
    model_pts = np.vstack([model_quads[k] for k in roi_keys]).astype(np.float32)
    img_pts = np.vstack([video_quads[k] for k in roi_keys]).astype(np.float32)

    return model_pts, img_pts


def extract_centroid_correspondences(
    model_polygons: Dict[str, Polygon],
    video_polygons: Dict[str, Polygon]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract matching centroid correspondences from model and video ROI polygons.

    Args:
        model_polygons: Dict of model ROI polygons (meters, Y-up)
        video_polygons: Dict of video ROI polygons (pixels, Y-down)

    Returns:
        Tuple of (model_centroids, image_centroids):
            - model_centroids: (N, 2) array of model centroids in meters
            - image_centroids: (N, 2) array of image centroids in pixels
    """
    # Get keys present in both
    roi_keys = [k for k in model_polygons.keys() if k in video_polygons]

    def centroid_xy(poly: Polygon) -> np.ndarray:
        c = poly.centroid
        return np.array([c.x, c.y], dtype=np.float32)

    model_centroids = np.vstack([
        centroid_xy(model_polygons[k]) for k in roi_keys
    ]).astype(np.float32)

    img_centroids = np.vstack([
        centroid_xy(video_polygons[k]) for k in roi_keys
    ]).astype(np.float32)

    return model_centroids, img_centroids


def estimate_pixels_per_meter(
    model_polygons: Dict[str, Polygon],
    video_polygons: Dict[str, Polygon],
    reference_segment: str = "segment2"
) -> float:
    """
    Estimate pixels-per-meter scale by comparing a reference segment width.

    Args:
        model_polygons: Dict of model ROI polygons (meters)
        video_polygons: Dict of video ROI polygons (pixels)
        reference_segment: Name of ROI segment to use for scale estimation

    Returns:
        Estimated pixels per meter
    """
    if reference_segment not in model_polygons or reference_segment not in video_polygons:
        raise ValueError(f"Reference segment '{reference_segment}' not found in both polygon sets")

    # Get segment width in meters
    model_poly = model_polygons[reference_segment]
    model_coords = np.asarray(model_poly.exterior.coords)[:-1]
    model_width = np.linalg.norm(model_coords[1] - model_coords[0])

    # Get segment width in pixels
    video_poly = video_polygons[reference_segment]
    video_coords = np.asarray(video_poly.exterior.coords)[:-1]

    # Calculate width as mean of horizontal edges
    def top_bottom_indices(q_ydown):
        ys = q_ydown[:, 1]
        top_two = np.argsort(ys)[:2]
        bot_two = np.argsort(ys)[-2:]
        return np.sort(top_two), np.sort(bot_two)

    top2, bot2 = top_bottom_indices(video_coords)
    w_top = np.linalg.norm(video_coords[top2[1]] - video_coords[top2[0]])
    w_bot = np.linalg.norm(video_coords[bot2[1]] - video_coords[bot2[0]])
    video_width = (w_top + w_bot) * 0.5

    pixels_per_meter = video_width / model_width

    return pixels_per_meter
