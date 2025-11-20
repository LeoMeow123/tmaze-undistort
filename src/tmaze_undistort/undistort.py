"""Main T-maze undistortion pipeline."""

import cv2
import numpy as np
import sleap_io as sio
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict
from shapely.geometry import Polygon

from .models import TMazeModel, create_standard_tmaze
from .roi import load_rois, validate_roi_coverage, auto_detect_roi_file
from .geometry import extract_centroid_correspondences, estimate_pixels_per_meter
from .calibration import (
    calibrate_camera,
    create_undistort_maps,
    undistort_image,
    save_calibration,
    load_calibration
)


class TMazeUndistortionPipeline:
    """
    Complete pipeline for T-maze video undistortion.

    This class handles:
    1. Loading ROI labels from video
    2. Creating physical maze model
    3. Camera calibration from model-to-video correspondences
    4. Video undistortion and rectification
    5. Optional masking to maze regions only
    """

    def __init__(
        self,
        video_path: str,
        roi_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        maze_model: Optional[TMazeModel] = None
    ):
        """
        Initialize T-maze undistortion pipeline.

        Args:
            video_path: Path to input video file
            roi_path: Path to ROI YAML file (auto-detected if None)
            calibration_path: Path to saved calibration file (will calibrate if None)
            maze_model: Physical maze model (uses standard T-maze if None)
        """
        self.video_path = Path(video_path)

        # Auto-detect ROI file if not provided
        if roi_path is None:
            roi_path = auto_detect_roi_file(self.video_path)
            if roi_path is None:
                raise FileNotFoundError(
                    f"Could not auto-detect ROI file for {self.video_path}\n"
                    f"Expected: {self.video_path.with_suffix('.rois.yml')}\n"
                    f"Please provide roi_path or create ROI labels."
                )

        self.roi_path = Path(roi_path)
        self.calibration_path = Path(calibration_path) if calibration_path else None

        # Use standard T-maze model if not provided
        self.maze_model = maze_model if maze_model else create_standard_tmaze()

        # Will be populated during calibration
        self.K = None
        self.dist = None
        self.newK = None
        self.map1 = None
        self.map2 = None
        self.H_u2c = None  # Homography: undistorted -> canvas
        self.rms_error = None
        self.image_size = None
        self.canvas_size = None
        self.meters_per_pixel = None

        # ROI polygons
        self.model_polygons = None
        self.video_polygons = None

    def calibrate(
        self,
        save_calibration_path: Optional[str] = None,
        fix_principal_point: bool = True,
        fix_aspect_ratio: bool = True,
        estimate_tangential_distortion: bool = True
    ) -> Dict:
        """
        Perform camera calibration.

        Args:
            save_calibration_path: If provided, save calibration to this path
            fix_principal_point: Fix principal point at image center
            fix_aspect_ratio: Constrain fx ≈ fy
            estimate_tangential_distortion: Estimate tangential distortion (p1, p2)

        Returns:
            Dictionary with calibration results and diagnostics
        """
        print(f"Loading video: {self.video_path}")
        video = sio.load_video(str(self.video_path))
        frame0 = video[0]
        h, w = frame0.shape[:2]
        self.image_size = (w, h)

        print(f"Loading ROIs: {self.roi_path}")
        if not self.roi_path.exists():
            raise FileNotFoundError(
                f"ROI file not found: {self.roi_path}\n"
                f"Please create ROI labels for the video."
            )

        self.video_polygons = load_rois(self.roi_path)
        print(f"Loaded {len(self.video_polygons)} ROI segments: {list(self.video_polygons.keys())}")

        # Validate ROI coverage
        validate_roi_coverage(self.video_polygons)

        # Get model polygons (only for ROIs present in video)
        available_segments = tuple(self.video_polygons.keys())
        self.model_polygons = self.maze_model.get_roi_polygons(segments=available_segments)
        print(f"Created {len(self.model_polygons)} model segments")

        # Extract correspondences (using centroids for calibration)
        model_pts, img_pts = extract_centroid_correspondences(
            self.model_polygons, self.video_polygons
        )
        print(f"Extracted {len(model_pts)} point correspondences")

        # Calibrate camera
        print("Calibrating camera...")
        self.rms_error, self.K, self.dist = calibrate_camera(
            model_pts, img_pts, self.image_size,
            fix_principal_point=fix_principal_point,
            fix_aspect_ratio=fix_aspect_ratio,
            estimate_tangential_distortion=estimate_tangential_distortion
        )

        print(f"RMS reprojection error: {self.rms_error:.4f} pixels")
        print(f"Camera matrix:\n{self.K}")
        print(f"Distortion coefficients: {self.dist.ravel()}")

        # Create undistortion maps
        print("Creating undistortion maps...")
        self.map1, self.map2, self.newK = create_undistort_maps(
            self.K, self.dist, self.image_size, alpha=1.0
        )

        # Compute homography (undistorted image -> rectified canvas)
        print("Computing homography for rectification...")
        self._compute_homography(model_pts, img_pts)

        # Save calibration if requested
        if save_calibration_path:
            save_calibration(
                Path(save_calibration_path),
                self.K, self.dist, self.newK,
                self.rms_error, self.image_size,
                metadata={
                    'video': str(self.video_path),
                    'roi_file': str(self.roi_path),
                    'maze_width_m': self.maze_model.maze_width,
                    'segment_length_m': self.maze_model.segment_length,
                }
            )

        return {
            'rms_error': self.rms_error,
            'K': self.K,
            'dist': self.dist,
            'newK': self.newK,
            'image_size': self.image_size,
        }

    def _compute_homography(self, model_pts: np.ndarray, img_pts: np.ndarray) -> None:
        """Compute homography from undistorted image to rectified canvas."""
        # Undistort image points
        imgp = img_pts.reshape(-1, 1, 2).astype(np.float32)
        und_img_pts = cv2.undistortPoints(imgp, self.K, self.dist, P=self.newK).reshape(-1, 2)

        # Set up canvas that matches input image size
        h, w = self.image_size[1], self.image_size[0]

        # Get model bounds
        all_m = np.concatenate([
            np.asarray(p.exterior.coords) for p in self.model_polygons.values()
        ], axis=0)
        xmin, xmax = all_m[:, 0].min(), all_m[:, 0].max()
        ymin, ymax = all_m[:, 1].min(), all_m[:, 1].max()
        width_m = xmax - xmin
        height_m = ymax - ymin

        # Choose meters-per-pixel to fit maze in canvas
        self.meters_per_pixel = max(width_m / float(w), height_m / float(h))

        # Size of fitted maze in pixels
        W_fit = int(np.ceil(width_m / self.meters_per_pixel))
        H_fit = int(np.ceil(height_m / self.meters_per_pixel))

        # Center maze on canvas
        ox = (w - W_fit) // 2
        oy = (h - H_fit) // 2

        self.canvas_size = (w, h)

        def meters_to_canvas_xy(mxy: np.ndarray) -> np.ndarray:
            """Model (meters, Y-up) -> canvas pixels (Y-down), centered."""
            x_px = (mxy[:, 0] - xmin) / self.meters_per_pixel + ox
            y_px = (ymax - mxy[:, 1]) / self.meters_per_pixel + oy  # Y-flip
            return np.column_stack([x_px, y_px]).astype(np.float32)

        # Convert model points to canvas coordinates
        dst_canvas_pts = meters_to_canvas_xy(model_pts)

        # Find homography: undistorted image -> canvas
        self.H_u2c, inl = cv2.findHomography(
            und_img_pts.astype(np.float32),
            dst_canvas_pts.astype(np.float32),
            cv2.RANSAC, 2.0
        )

        inliers = inl.ravel().astype(bool)
        proj_canvas = cv2.perspectiveTransform(und_img_pts.reshape(-1, 1, 2), self.H_u2c).reshape(-1, 2)
        rms_h = np.sqrt(np.mean(np.sum((proj_canvas[inliers] - dst_canvas_pts[inliers])**2, axis=1)))

        print(f"Homography inliers: {int(inliers.sum())}/{len(inliers)}")
        print(f"Homography RMS: {rms_h:.2f} canvas pixels")
        print(f"Canvas size: {self.canvas_size[0]}×{self.canvas_size[1]} px")
        print(f"Meters per pixel: {self.meters_per_pixel:.6f} m/px ({1000*self.meters_per_pixel:.3f} mm/px)")

    def load_existing_calibration(self) -> Dict:
        """
        Load calibration from saved file.

        Returns:
            Dictionary with calibration parameters
        """
        if self.calibration_path is None or not self.calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {self.calibration_path}")

        print(f"Loading calibration: {self.calibration_path}")
        calib = load_calibration(self.calibration_path)

        self.K = calib['K']
        self.dist = calib['dist']
        self.newK = calib['newK']
        self.rms_error = calib['rms_error']
        self.image_size = calib['image_size']

        # Create undistortion maps
        self.map1, self.map2, _ = create_undistort_maps(
            self.K, self.dist, self.image_size, alpha=1.0
        )

        print(f"Loaded calibration with RMS error: {self.rms_error:.4f} pixels")

        return calib

    def undistort_video(
        self,
        output_path: str,
        apply_mask: bool = True,
        crop_to_maze: bool = True,
        interpolation: str = "linear",
        show_progress: bool = True
    ) -> None:
        """
        Undistort and rectify entire video.

        Args:
            output_path: Path to save output video
            apply_mask: If True, mask background outside maze regions
            crop_to_maze: If True, crop output to maze bounding box
            interpolation: Interpolation method ("linear", "cubic", "lanczos4")
            show_progress: Show progress bar
        """
        if self.K is None or self.H_u2c is None:
            raise RuntimeError("Must call calibrate() or load_existing_calibration() first")

        # Map interpolation string to OpenCV constant
        interp_map = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
        }
        interp = interp_map.get(interpolation, cv2.INTER_LINEAR)

        print(f"Loading video: {self.video_path}")
        video = sio.load_video(str(self.video_path))

        # Get FPS
        meta = iio.immeta(str(self.video_path), exclude_applied=False)
        fps = meta.get("fps") or (meta.get("video") or {}).get("fps") or 30.0
        print(f"Video FPS: {fps}")

        # Create mask if requested
        mask = None
        crop_box = None
        if apply_mask:
            mask = self._create_canvas_mask()
            if crop_to_maze:
                crop_box = self._get_mask_crop_box(mask)
                print(f"Crop box: x={crop_box[0]}:{crop_box[1]}, y={crop_box[2]}:{crop_box[3]}")

        # Process video
        print(f"Writing undistorted video: {output_path}")
        W_out, H_out = self.canvas_size

        with sio.VideoWriter(output_path, fps=fps) as vw:
            iterator = tqdm(video, desc="Undistorting") if show_progress else video

            for frame in iterator:
                # Undistort
                und = cv2.remap(frame, self.map1, self.map2, interpolation=interp,
                               borderMode=cv2.BORDER_CONSTANT)

                # Rectify
                rect = cv2.warpPerspective(und, self.H_u2c, (W_out, H_out),
                                          flags=interp, borderMode=cv2.BORDER_CONSTANT)

                # Apply mask
                if mask is not None:
                    if rect.ndim == 2:
                        rect = cv2.bitwise_and(rect, rect, mask=mask)
                    else:
                        rect = cv2.bitwise_and(rect, rect, mask=cv2.merge([mask, mask, mask]))

                # Crop
                if crop_box is not None:
                    x0, x1, y0, y1 = crop_box
                    rect = rect[y0:y1, x0:x1]

                vw(rect)

        print(f"Done! Output saved to: {output_path}")

    def _create_canvas_mask(self) -> np.ndarray:
        """Create binary mask for maze regions on canvas."""
        W_out, H_out = self.canvas_size
        mask = np.zeros((H_out, W_out), dtype=np.uint8)

        # Get model bounds
        all_m = np.concatenate([
            np.asarray(p.exterior.coords) for p in self.model_polygons.values()
        ], axis=0)
        xmin = all_m[:, 0].min()
        ymax = all_m[:, 1].max()

        def meters_to_canvas_xy(mxy: np.ndarray) -> np.ndarray:
            """Model (meters, Y-up) -> canvas pixels (Y-down)."""
            W_out, H_out = self.canvas_size
            w, h = self.image_size
            width_m = all_m[:, 0].max() - xmin
            height_m = ymax - all_m[:, 1].min()
            W_fit = int(np.ceil(width_m / self.meters_per_pixel))
            H_fit = int(np.ceil(height_m / self.meters_per_pixel))
            ox = (w - W_fit) // 2
            oy = (h - H_fit) // 2

            x_px = (mxy[:, 0] - xmin) / self.meters_per_pixel + ox
            y_px = (ymax - mxy[:, 1]) / self.meters_per_pixel + oy
            return np.column_stack([x_px, y_px]).astype(np.int32)

        # Draw each ROI on mask
        for name, poly in self.model_polygons.items():
            pts_m = np.asarray(poly.exterior.coords)[:-1]
            pts_c = meters_to_canvas_xy(pts_m)
            cv2.fillPoly(mask, [pts_c], 255)

        return mask

    def _get_mask_crop_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of mask for cropping."""
        ys, xs = np.where(mask > 0)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        return x0, x1, y0, y1
