"""Command-line interface for T-maze undistortion."""

import argparse
import sys
from pathlib import Path

from .undistort import TMazeUndistortionPipeline
from .models import create_standard_tmaze, create_custom_tmaze


def main():
    """Main CLI entry point for tmaze-undistort command."""
    parser = argparse.ArgumentParser(
        description="Undistort T-maze videos using camera calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detects ROI file)
  tmaze-undistort video.mp4 -o undistorted.mp4

  # Specify ROI file explicitly
  tmaze-undistort video.mp4 --roi video.rois.yml -o undistorted.mp4

  # Save calibration for reuse
  tmaze-undistort video.mp4 -o undistorted.mp4 --save-calibration calib.yml

  # Reuse existing calibration
  tmaze-undistort video2.mp4 -o undistorted2.mp4 --calibration calib.yml

  # Custom maze dimensions
  tmaze-undistort video.mp4 -o out.mp4 --maze-width 0.12 --segment-length 0.25

  # Skip masking/cropping
  tmaze-undistort video.mp4 -o out.mp4 --no-mask --no-crop
        """
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to output undistorted video"
    )

    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Path to ROI YAML file (auto-detected if not provided)"
    )

    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to existing calibration file (will calibrate if not provided)"
    )

    parser.add_argument(
        "--save-calibration",
        type=str,
        default=None,
        help="Path to save calibration for reuse"
    )

    # Maze model options
    maze_group = parser.add_argument_group("maze model options")
    maze_group.add_argument(
        "--maze-width",
        type=float,
        default=0.10,
        help="Internal width of maze corridor in meters (default: 0.10)"
    )

    maze_group.add_argument(
        "--segment-length",
        type=float,
        default=0.20,
        help="Length of each stem segment in meters (default: 0.20)"
    )

    maze_group.add_argument(
        "--arm-length",
        type=float,
        default=0.275,
        help="Length of each arm in meters (default: 0.275)"
    )

    maze_group.add_argument(
        "--wall-width",
        type=float,
        default=0.005,
        help="Width of maze walls in meters (default: 0.005)"
    )

    # Calibration options
    calib_group = parser.add_argument_group("calibration options")
    calib_group.add_argument(
        "--free-principal-point",
        action="store_true",
        help="Allow principal point to vary (default: fixed at image center)"
    )

    calib_group.add_argument(
        "--free-aspect-ratio",
        action="store_true",
        help="Allow aspect ratio to vary (default: fixed fx≈fy)"
    )

    calib_group.add_argument(
        "--no-tangential",
        action="store_true",
        help="Disable tangential distortion estimation (default: enabled)"
    )

    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable masking (keep background)"
    )

    output_group.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable cropping to maze bounds"
    )

    output_group.add_argument(
        "--interpolation",
        choices=["linear", "cubic", "lanczos4"],
        default="linear",
        help="Interpolation method (default: linear)"
    )

    output_group.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )

    args = parser.parse_args()

    # Validate inputs
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1

    # Create maze model
    if args.maze_width == 0.10 and args.segment_length == 0.20 and args.arm_length == 0.275:
        # Use standard model
        maze_model = create_standard_tmaze()
    else:
        # Use custom dimensions
        maze_model = create_custom_tmaze(
            maze_width=args.maze_width,
            segment_length=args.segment_length,
            arm_length=args.arm_length,
            wall_width=args.wall_width
        )

    try:
        # Initialize pipeline
        print("=" * 60)
        print("T-Maze Video Undistortion Pipeline")
        print("=" * 60)

        pipeline = TMazeUndistortionPipeline(
            video_path=str(video_path),
            roi_path=args.roi,
            calibration_path=args.calibration,
            maze_model=maze_model
        )

        # Load or perform calibration
        if args.calibration:
            pipeline.load_existing_calibration()
        else:
            pipeline.calibrate(
                save_calibration_path=args.save_calibration,
                fix_principal_point=not args.free_principal_point,
                fix_aspect_ratio=not args.free_aspect_ratio,
                estimate_tangential_distortion=not args.no_tangential
            )

        # Undistort video
        print("\n" + "=" * 60)
        print("Undistorting Video")
        print("=" * 60)

        pipeline.undistort_video(
            output_path=args.output,
            apply_mask=not args.no_mask,
            crop_to_maze=not args.no_crop,
            interpolation=args.interpolation,
            show_progress=not args.no_progress
        )

        print("\n" + "=" * 60)
        print("Success!")
        print("=" * 60)
        print(f"Output video: {args.output}")
        if args.save_calibration:
            print(f"Calibration saved: {args.save_calibration}")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
