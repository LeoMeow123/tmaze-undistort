"""
Statistical test to determine if a video needs undistortion based on charuco board analysis.

This script analyzes charuco boards in video frames to quantify lens distortion
and determine whether undistortion is necessary.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import argparse
import json
import csv
from datetime import datetime


def detect_charuco_board(
    image: np.ndarray,
    squares_x: int = 14,
    squares_y: int = 9,
    square_length: float = 0.02,
    marker_length: float = 0.015,
    dictionary: int = cv2.aruco.DICT_5X5_250
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect charuco board in an image.

    Args:
        image: Input image (grayscale or BGR)
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length: Length of square side (meters)
        marker_length: Length of marker side (meters)
        dictionary: ArUco dictionary type

    Returns:
        Tuple of (corners, ids, rejected):
            - corners: Detected charuco corner positions
            - ids: Corner IDs
            - rejected: Rejected marker candidates
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Create charuco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dict
    )

    # Create CharucoDetector
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    # Detect charuco corners
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    # Return corners if detected
    if charuco_corners is not None and len(charuco_corners) > 0:
        return charuco_corners, charuco_ids, marker_corners

    return None, None, None


def compute_line_straightness_score(corners: np.ndarray, ids: np.ndarray,
                                    squares_x: int, squares_y: int) -> float:
    """
    Measure how straight the board edges are. Distortion causes edge bowing.

    Args:
        corners: (N, 1, 2) array of detected corner positions
        ids: (N, 1) array of corner IDs
        squares_x: Number of squares in X
        squares_y: Number of squares in Y

    Returns:
        RMS deviation from straight lines (pixels) - lower is better
    """
    if corners is None or len(corners) < 4:
        return np.inf

    corners = corners.reshape(-1, 2)
    ids = ids.flatten()

    # Organize corners by row and column
    # CharucoBoard corner indexing: row * (squares_x - 1) + col
    corner_dict = {}
    for corner, corner_id in zip(corners, ids):
        row = corner_id // (squares_x - 1)
        col = corner_id % (squares_x - 1)
        corner_dict[(row, col)] = corner

    deviations = []

    # Check horizontal lines (rows)
    for row in range(squares_y - 1):
        row_corners = []
        for col in range(squares_x - 1):
            if (row, col) in corner_dict:
                row_corners.append(corner_dict[(row, col)])

        if len(row_corners) >= 3:
            row_corners = np.array(row_corners)
            # Fit line and measure deviations
            vx, vy, x0, y0 = cv2.fitLine(row_corners.astype(np.float32),
                                          cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

            # Calculate perpendicular distance from each point to line
            for pt in row_corners:
                # Distance from point to line: |ax + by + c| / sqrt(a^2 + b^2)
                # Line equation: vy*x - vx*y + (vx*y0 - vy*x0) = 0
                dist = abs(vy * pt[0] - vx * pt[1] + (vx * y0 - vy * x0))
                deviations.append(dist)

    # Check vertical lines (columns)
    for col in range(squares_x - 1):
        col_corners = []
        for row in range(squares_y - 1):
            if (row, col) in corner_dict:
                col_corners.append(corner_dict[(row, col)])

        if len(col_corners) >= 3:
            col_corners = np.array(col_corners)
            vx, vy, x0, y0 = cv2.fitLine(col_corners.astype(np.float32),
                                          cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

            for pt in col_corners:
                dist = abs(vy * pt[0] - vx * pt[1] + (vx * y0 - vy * x0))
                deviations.append(dist)

    if len(deviations) == 0:
        return np.inf

    # Return RMS deviation
    return np.sqrt(np.mean(np.array(deviations) ** 2))


def compute_corner_spacing_uniformity(corners: np.ndarray, ids: np.ndarray,
                                      squares_x: int) -> float:
    """
    Measure uniformity of corner spacing. Distortion causes non-uniform spacing.

    Args:
        corners: (N, 1, 2) array of corner positions
        ids: (N, 1) array of corner IDs
        squares_x: Number of squares in X

    Returns:
        Coefficient of variation of distances - lower is better (< 0.05 is good)
    """
    if corners is None or len(corners) < 2:
        return np.inf

    corners = corners.reshape(-1, 2)
    ids = ids.flatten()

    # Calculate distances between adjacent corners
    distances = []
    corner_dict = {}
    for corner, corner_id in zip(corners, ids):
        row = corner_id // (squares_x - 1)
        col = corner_id % (squares_x - 1)
        corner_dict[(row, col)] = corner

    # Horizontal distances
    for row, col in corner_dict.keys():
        if (row, col + 1) in corner_dict:
            dist = np.linalg.norm(corner_dict[(row, col)] - corner_dict[(row, col + 1)])
            distances.append(dist)

        # Vertical distances
        if (row + 1, col) in corner_dict:
            dist = np.linalg.norm(corner_dict[(row, col)] - corner_dict[(row + 1, col)])
            distances.append(dist)

    if len(distances) == 0:
        return np.inf

    distances = np.array(distances)
    # Coefficient of variation: std / mean
    return np.std(distances) / np.mean(distances)


def compute_reprojection_error(
    corners: np.ndarray,
    ids: np.ndarray,
    image_size: Tuple[int, int],
    squares_x: int = 14,
    squares_y: int = 9,
    square_length: float = 0.02,
    marker_length: float = 0.015,
    dictionary: int = cv2.aruco.DICT_5X5_250
) -> float:
    """
    Compute reprojection error after initial calibration.
    High error indicates significant distortion.

    Args:
        corners: Detected charuco corners
        ids: Corner IDs
        image_size: (width, height) of image
        squares_x: Number of squares in X
        squares_y: Number of squares in Y
        square_length: Square side length in meters
        marker_length: Marker side length in meters

    Returns:
        RMS reprojection error in pixels
    """
    if corners is None or len(corners) < 4:
        return np.inf

    # Create board
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dict
    )

    # Get 3D object points for detected corners
    obj_points = board.getChessboardCorners()[ids.flatten()]

    # Simple calibration with no distortion assumed
    w, h = image_size
    K_init = np.array([
        [w, 0, w/2],
        [0, w, h/2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_init = np.zeros(5, dtype=np.float64)

    # Estimate pose
    ret, rvec, tvec = cv2.solvePnP(
        obj_points.astype(np.float32),
        corners.astype(np.float32),
        K_init,
        dist_init,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Reproject points
    reprojected, _ = cv2.projectPoints(obj_points, rvec, tvec, K_init, dist_init)
    reprojected = reprojected.reshape(-1, 2)
    corners_2d = corners.reshape(-1, 2)

    # Calculate RMS error
    errors = np.linalg.norm(corners_2d - reprojected, axis=1)
    rms_error = np.sqrt(np.mean(errors ** 2))

    return rms_error


def test_video_distortion(
    video_path: Path,
    num_frames: int = 10,
    squares_x: int = 14,
    squares_y: int = 9,
    square_length: float = 0.02,
    marker_length: float = 0.015,
    dictionary: int = cv2.aruco.DICT_5X5_250,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Test if a video needs undistortion by analyzing charuco board.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length: Square side length in meters
        marker_length: Marker side length in meters
        verbose: Print detailed results

    Returns:
        Dictionary with distortion metrics:
            - 'line_straightness': RMS deviation from straight lines (pixels)
            - 'spacing_uniformity': Coefficient of variation of corner distances
            - 'reprojection_error': RMS reprojection error (pixels)
            - 'needs_undistortion': Boolean recommendation
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    line_straightness_scores = []
    spacing_uniformity_scores = []
    reprojection_errors = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Detect charuco board
        corners, ids, _ = detect_charuco_board(
            frame, squares_x, squares_y, square_length, marker_length, dictionary
        )

        if corners is not None:
            # Compute metrics
            line_score = compute_line_straightness_score(corners, ids, squares_x, squares_y)
            spacing_score = compute_corner_spacing_uniformity(corners, ids, squares_x)
            reproj_error = compute_reprojection_error(
                corners, ids, (width, height),
                squares_x, squares_y, square_length, marker_length, dictionary
            )

            line_straightness_scores.append(line_score)
            spacing_uniformity_scores.append(spacing_score)
            reprojection_errors.append(reproj_error)

    cap.release()

    if len(line_straightness_scores) == 0:
        if verbose:
            print("WARNING: No charuco board detected in any frame!")
        return {
            'line_straightness': np.inf,
            'spacing_uniformity': np.inf,
            'reprojection_error': np.inf,
            'needs_undistortion': None,
            'frames_detected': 0
        }

    # Average metrics across frames
    avg_line_straightness = np.mean(line_straightness_scores)
    avg_spacing_uniformity = np.mean(spacing_uniformity_scores)
    avg_reprojection_error = np.mean(reprojection_errors)

    # Decision thresholds (empirically determined)
    # These can be tuned based on your specific setup
    LINE_THRESHOLD = 2.0  # pixels
    SPACING_THRESHOLD = 0.05  # coefficient of variation
    REPROJ_THRESHOLD = 5.0  # pixels

    # Determine if undistortion is needed
    needs_undistortion = (
        avg_line_straightness > LINE_THRESHOLD or
        avg_spacing_uniformity > SPACING_THRESHOLD or
        avg_reprojection_error > REPROJ_THRESHOLD
    )

    results = {
        'line_straightness': avg_line_straightness,
        'spacing_uniformity': avg_spacing_uniformity,
        'reprojection_error': avg_reprojection_error,
        'needs_undistortion': needs_undistortion,
        'frames_detected': len(line_straightness_scores)
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Distortion Analysis for: {video_path.name}")
        print(f"{'='*60}")
        print(f"Frames analyzed: {len(line_straightness_scores)}/{num_frames}")
        print(f"\nMetrics:")
        print(f"  Line straightness:      {avg_line_straightness:.3f} px (threshold: {LINE_THRESHOLD:.1f} px)")
        print(f"  Spacing uniformity:     {avg_spacing_uniformity:.4f} (threshold: {SPACING_THRESHOLD:.2f})")
        print(f"  Reprojection error:     {avg_reprojection_error:.3f} px (threshold: {REPROJ_THRESHOLD:.1f} px)")
        print(f"\n{'='*60}")
        print(f"Recommendation: {'UNDISTORTION NEEDED' if needs_undistortion else 'NO UNDISTORTION NEEDED'}")
        print(f"{'='*60}\n")

    return results


def save_results_csv(results: List[Dict], output_path: str) -> None:
    """
    Save batch results to CSV file.

    Args:
        results: List of result dictionaries
        output_path: Path to CSV file
    """
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'video', 'video_path', 'frames_detected',
            'line_straightness', 'spacing_uniformity', 'reprojection_error',
            'needs_undistortion', 'recommendation'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            if r['needs_undistortion'] is None:
                recommendation = "NO_BOARD_DETECTED"
            elif r['needs_undistortion']:
                recommendation = "UNDISTORTION_NEEDED"
            else:
                recommendation = "OK"

            writer.writerow({
                'video': r['video'],
                'video_path': r['video_path'],
                'frames_detected': r['frames_detected'],
                'line_straightness': f"{r['line_straightness']:.4f}" if np.isfinite(r['line_straightness']) else 'N/A',
                'spacing_uniformity': f"{r['spacing_uniformity']:.4f}" if np.isfinite(r['spacing_uniformity']) else 'N/A',
                'reprojection_error': f"{r['reprojection_error']:.4f}" if np.isfinite(r['reprojection_error']) else 'N/A',
                'needs_undistortion': r['needs_undistortion'],
                'recommendation': recommendation
            })


def save_results_json(results: List[Dict], output_path: str) -> None:
    """
    Save batch results to JSON file.

    Args:
        results: List of result dictionaries
        output_path: Path to JSON file
    """
    # Convert numpy types to native Python types for JSON serialization
    json_results = []
    for r in results.copy():
        json_r = {
            'video': r['video'],
            'video_path': r['video_path'],
            'frames_detected': int(r['frames_detected']),
            'line_straightness': float(r['line_straightness']) if np.isfinite(r['line_straightness']) else None,
            'spacing_uniformity': float(r['spacing_uniformity']) if np.isfinite(r['spacing_uniformity']) else None,
            'reprojection_error': float(r['reprojection_error']) if np.isfinite(r['reprojection_error']) else None,
            'needs_undistortion': bool(r['needs_undistortion']) if r['needs_undistortion'] is not None else None
        }
        json_results.append(json_r)

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': json_results
        }, f, indent=2)


def save_log_file(
    results: List[Dict],
    output_path: str,
    total_videos: int,
    videos_ok: int,
    videos_needing_undistortion: int,
    videos_no_board: int,
    args
) -> None:
    """
    Save summary log file.

    Args:
        results: List of result dictionaries
        output_path: Path to log file
        total_videos: Total number of videos processed
        videos_ok: Number of videos not needing undistortion
        videos_needing_undistortion: Number of videos needing undistortion
        videos_no_board: Number of videos with no board detected
        args: Command-line arguments
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DISTORTION TEST LOG\n")
        f.write("="*80 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directory: {args.video}\n\n")

        f.write("Test Parameters:\n")
        f.write(f"  Charuco board: {args.squares_x}x{args.squares_y}\n")
        f.write(f"  Square length: {args.square_length}m\n")
        f.write(f"  Marker length: {args.marker_length}m\n")
        f.write(f"  Dictionary: {args.dictionary}\n")
        f.write(f"  Frames per video: {args.num_frames}\n\n")

        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Total videos processed: {total_videos}\n")
        f.write(f"Videos OK (no undistortion needed): {videos_ok} ({videos_ok/total_videos*100:.1f}%)\n")
        f.write(f"Videos needing undistortion: {videos_needing_undistortion} ({videos_needing_undistortion/total_videos*100:.1f}%)\n")
        f.write(f"Videos with no board detected: {videos_no_board} ({videos_no_board/total_videos*100:.1f}%)\n\n")

        # List videos needing undistortion
        if videos_needing_undistortion > 0:
            f.write("="*80 + "\n")
            f.write("VIDEOS REQUIRING UNDISTORTION:\n")
            f.write("="*80 + "\n")
            for r in results:
                if r['needs_undistortion']:
                    f.write(f"\n{r['video']}:\n")
                    f.write(f"  Path: {r['video_path']}\n")
                    f.write(f"  Line straightness: {r['line_straightness']:.4f} px\n")
                    f.write(f"  Spacing uniformity: {r['spacing_uniformity']:.4f}\n")
                    f.write(f"  Reprojection error: {r['reprojection_error']:.4f} px\n")
            f.write("\n")

        # List videos with no board detected
        if videos_no_board > 0:
            f.write("="*80 + "\n")
            f.write("VIDEOS WITH NO CHARUCO BOARD DETECTED:\n")
            f.write("="*80 + "\n")
            for r in results:
                if r['needs_undistortion'] is None:
                    f.write(f"  - {r['video']}\n")
            f.write("\n")

        f.write("="*80 + "\n")
        f.write("ALL RESULTS\n")
        f.write("="*80 + "\n\n")

        for r in results:
            status = "NO_BOARD" if r['needs_undistortion'] is None else ("NEEDS_UNDIST" if r['needs_undistortion'] else "OK")
            f.write(f"{r['video']:50s} [{status:12s}] ")
            if r['needs_undistortion'] is not None:
                f.write(f"LS:{r['line_straightness']:6.3f} SU:{r['spacing_uniformity']:6.4f} RE:{r['reprojection_error']:6.3f}")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test if a video needs undistortion based on charuco board analysis"
    )
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--num-frames", type=int, default=10,
                       help="Number of frames to sample (default: 10)")
    parser.add_argument("--squares-x", type=int, default=14,
                       help="Number of squares in X direction (default: 14)")
    parser.add_argument("--squares-y", type=int, default=9,
                       help="Number of squares in Y direction (default: 9)")
    parser.add_argument("--square-length", type=float, default=0.02,
                       help="Square side length in meters (default: 0.02)")
    parser.add_argument("--marker-length", type=float, default=0.015,
                       help="Marker side length in meters (default: 0.015)")
    parser.add_argument("--dictionary", type=str, default="DICT_5X5_250",
                       choices=["DICT_4X4_50", "DICT_5X5_250", "DICT_6X6_250",
                               "DICT_7X7_250", "DICT_ARUCO_ORIGINAL"],
                       help="ArUco dictionary type (default: DICT_5X5_250)")
    parser.add_argument("--batch", action="store_true",
                       help="Process all videos in directory")
    parser.add_argument("--output-csv", type=str, default=None,
                       help="Save results to CSV file (default: distortion_test_results.csv in batch mode)")
    parser.add_argument("--output-json", type=str, default=None,
                       help="Save results to JSON file (default: distortion_test_results.json in batch mode)")
    parser.add_argument("--output-log", type=str, default=None,
                       help="Save summary log file (default: distortion_test_log.txt in batch mode)")

    args = parser.parse_args()

    # Convert dictionary string to constant
    dict_map = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    dictionary = dict_map[args.dictionary]

    if args.batch:
        # Process all videos in directory
        if not args.video.is_dir():
            print(f"Error: {args.video} is not a directory")
            return

        video_files = list(args.video.glob("*.mp4")) + list(args.video.glob("*.avi"))

        if len(video_files) == 0:
            print(f"No video files found in {args.video}")
            return

        print(f"Found {len(video_files)} videos to analyze\n")

        results_summary = []
        for video_path in video_files:
            results = test_video_distortion(
                video_path,
                num_frames=args.num_frames,
                squares_x=args.squares_x,
                squares_y=args.squares_y,
                square_length=args.square_length,
                marker_length=args.marker_length,
                dictionary=dictionary,
                verbose=True
            )
            results['video'] = video_path.name
            results['video_path'] = str(video_path)
            results_summary.append(results)

        # Print summary
        print("\n" + "="*80)
        print("BATCH SUMMARY")
        print("="*80)

        total_videos = len(results_summary)
        videos_needing_undistortion = [r for r in results_summary if r['needs_undistortion']]
        videos_ok = [r for r in results_summary if not r['needs_undistortion']]
        videos_no_board = [r for r in results_summary if r['needs_undistortion'] is None]

        for r in results_summary:
            if r['needs_undistortion'] is None:
                status = "NO BOARD DETECTED"
            elif r['needs_undistortion']:
                status = "NEEDS UNDISTORTION"
            else:
                status = "OK"
            print(f"{r['video']:50s} -> {status}")

        print("="*80)
        print(f"Total videos processed: {total_videos}")
        print(f"Videos OK (no undistortion needed): {len(videos_ok)}")
        print(f"Videos needing undistortion: {len(videos_needing_undistortion)}")
        print(f"Videos with no board detected: {len(videos_no_board)}")
        print("="*80)

        # Save results to files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV output
        csv_path = args.output_csv if args.output_csv else f"distortion_test_results_{timestamp}.csv"
        save_results_csv(results_summary, csv_path)
        print(f"\n✓ Results saved to: {csv_path}")

        # JSON output
        json_path = args.output_json if args.output_json else f"distortion_test_results_{timestamp}.json"
        save_results_json(results_summary, json_path)
        print(f"✓ Results saved to: {json_path}")

        # Log file
        log_path = args.output_log if args.output_log else f"distortion_test_log_{timestamp}.txt"
        save_log_file(
            results_summary,
            log_path,
            total_videos,
            len(videos_ok),
            len(videos_needing_undistortion),
            len(videos_no_board),
            args
        )
        print(f"✓ Log saved to: {log_path}")
        print()

    else:
        # Process single video
        if not args.video.exists():
            print(f"Error: Video file not found: {args.video}")
            return

        test_video_distortion(
            args.video,
            num_frames=args.num_frames,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length=args.square_length,
            marker_length=args.marker_length,
            dictionary=dictionary,
            verbose=True
        )


if __name__ == "__main__":
    main()
