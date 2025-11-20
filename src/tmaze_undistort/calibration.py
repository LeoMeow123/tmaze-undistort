"""Camera calibration functions using OpenCV."""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict


def calibrate_camera(
    model_points: np.ndarray,
    image_points: np.ndarray,
    image_size: Tuple[int, int],
    fix_principal_point: bool = True,
    fix_aspect_ratio: bool = True,
    estimate_tangential_distortion: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calibrate camera using matched point correspondences.

    Args:
        model_points: (N, 2) array of model points in meters (X, Y)
        image_points: (N, 2) array of corresponding image points in pixels
        image_size: (width, height) of image in pixels
        fix_principal_point: If True, fix principal point at image center
        fix_aspect_ratio: If True, constrain fx ≈ fy
        estimate_tangential_distortion: If True, estimate p1, p2 (helps with lens tilt)

    Returns:
        Tuple of (rms_error, K, dist):
            - rms_error: RMS reprojection error in pixels
            - K: (3, 3) camera intrinsic matrix
            - dist: (5,) distortion coefficients [k1, k2, p1, p2, k3]
    """
    w, h = image_size

    # Convert 2D model points to 3D (Z=0 plane)
    objp = np.column_stack([model_points, np.zeros(len(model_points))]).astype(np.float32)
    objp = objp.reshape(-1, 1, 3)

    imgp = image_points.reshape(-1, 1, 2).astype(np.float32)

    # Initial intrinsics guess
    K0 = np.array([
        [w, 0, w/2],
        [0, w, h/2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist0 = np.zeros(5, dtype=np.float64)

    # Build calibration flags
    flags = 0
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_SKEW  # Assume skew = 0

    if fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

    if fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO

    if not estimate_tangential_distortion:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST

    # Fix higher-order radial distortion terms (only estimate k1, k2)
    flags |= cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6

    # Calibration criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 400, 1e-10)

    # Run calibration
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [imgp], (w, h), K0, dist0,
        flags=flags, criteria=criteria
    )

    return rms, K, dist


def create_undistort_maps(
    K: np.ndarray,
    dist: np.ndarray,
    image_size: Tuple[int, int],
    alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create undistortion maps for efficient remapping.

    Args:
        K: (3, 3) camera intrinsic matrix
        dist: (5,) distortion coefficients
        image_size: (width, height) of image
        alpha: Free scaling parameter (0=crop all invalid pixels, 1=keep all pixels)

    Returns:
        Tuple of (map1, map2, newK):
            - map1: X-coordinate remap
            - map2: Y-coordinate remap
            - newK: New camera matrix for undistorted image
    """
    w, h = image_size

    # Get optimal new camera matrix
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=alpha, newImgSize=(w, h))

    # Create undistortion maps
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, newK, (w, h), cv2.CV_32FC1
    )

    return map1, map2, newK


def undistort_image(
    image: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Undistort an image using precomputed maps.

    Args:
        image: Input image
        map1: X-coordinate remap
        map2: Y-coordinate remap
        interpolation: Interpolation method (INTER_LINEAR, INTER_LANCZOS4, etc.)

    Returns:
        Undistorted image
    """
    return cv2.remap(image, map1, map2, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)


def save_calibration(
    save_path: Path,
    K: np.ndarray,
    dist: np.ndarray,
    newK: np.ndarray,
    rms_error: float,
    image_size: Tuple[int, int],
    metadata: Optional[Dict] = None
) -> None:
    """
    Save calibration parameters to YAML file.

    Args:
        save_path: Path to save calibration file
        K: Camera intrinsic matrix
        dist: Distortion coefficients
        newK: New camera matrix for undistorted image
        rms_error: RMS reprojection error
        image_size: Image dimensions (width, height)
        metadata: Optional additional metadata
    """
    data = {
        'camera_matrix': K.tolist(),
        'distortion_coefficients': dist.ravel().tolist(),
        'new_camera_matrix': newK.tolist(),
        'rms_error': float(rms_error),
        'image_size': list(image_size),
    }

    if metadata:
        data['metadata'] = metadata

    with open(save_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Calibration saved to: {save_path}")


def load_calibration(calib_path: Path) -> Dict:
    """
    Load calibration parameters from YAML file.

    Args:
        calib_path: Path to calibration YAML file

    Returns:
        Dictionary with calibration parameters:
            - 'K': Camera intrinsic matrix
            - 'dist': Distortion coefficients
            - 'newK': New camera matrix
            - 'rms_error': RMS reprojection error
            - 'image_size': Image dimensions
    """
    with open(calib_path, 'r') as f:
        data = yaml.safe_load(f)

    return {
        'K': np.array(data['camera_matrix'], dtype=np.float64),
        'dist': np.array(data['distortion_coefficients'], dtype=np.float64),
        'newK': np.array(data['new_camera_matrix'], dtype=np.float64),
        'rms_error': data['rms_error'],
        'image_size': tuple(data['image_size']),
    }
