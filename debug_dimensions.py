"""Debug script to understand dimension differences between notebook and pipeline."""

import numpy as np
from shapely.geometry import Polygon

# Physical measurements (from notebook)
wall_width = 0.5 / 100  # m
maze_width = 10 / 100  # m
segment_length = 20 / 100  # m
maze_length = 110 / 100  # m
arms_length = 65 / 100  # m
arm_length = (arms_length - maze_width) / 2

print("Physical dimensions:")
print(f"  maze_width: {maze_width:.3f} m")
print(f"  segment_length: {segment_length:.3f} m")
print(f"  arm_length: {arm_length:.3f} m")
print()

# Define model polygons (same as notebook)
model_roi_polygons = {}

# Segment 2
model_roi_polygons['segment2'] = Polygon([
    (maze_width/2, segment_length/2),
    (-maze_width/2, segment_length/2),
    (-maze_width/2, 3*segment_length/2),
    (maze_width/2, 3*segment_length/2)
])

# Segment 3
model_roi_polygons['segment3'] = Polygon([
    (maze_width/2, 3*segment_length/2),
    (-maze_width/2, 3*segment_length/2),
    (-maze_width/2, 5*segment_length/2),
    (maze_width/2, 5*segment_length/2)
])

# Segment 4
model_roi_polygons['segment4'] = Polygon([
    (maze_width/2, 5*segment_length/2),
    (-maze_width/2, 5*segment_length/2),
    (-maze_width/2, 7*segment_length/2),
    (maze_width/2, 7*segment_length/2)
])

# Junction
model_roi_polygons['junction'] = Polygon([
    (maze_width/2, 7*segment_length/2),
    (-maze_width/2, 7*segment_length/2),
    (-maze_width/2, 7*segment_length/2 + maze_width),
    (maze_width/2, 7*segment_length/2 + maze_width)
])

# Arm Right
model_roi_polygons['arm_right'] = Polygon([
    (-maze_width/2, 7*segment_length/2),
    (-maze_width/2 - arm_length, 7*segment_length/2),
    (-maze_width/2 - arm_length, 7*segment_length/2 + maze_width),
    (-maze_width/2, 7*segment_length/2 + maze_width)
])

# Arm Left
model_roi_polygons['arm_left'] = Polygon([
    (maze_width/2 + arm_length, 7*segment_length/2),
    (maze_width/2, 7*segment_length/2),
    (maze_width/2, 7*segment_length/2 + maze_width),
    (maze_width/2 + arm_length, 7*segment_length/2 + maze_width)
])

print("Model ROI bounds:")
all_m = np.concatenate([np.asarray(p.exterior.coords) for p in model_roi_polygons.values()], axis=0)
xmin, xmax = all_m[:, 0].min(), all_m[:, 0].max()
ymin, ymax = all_m[:, 1].min(), all_m[:, 1].max()
width_m = xmax - xmin
height_m = ymax - ymin

print(f"  X range: [{xmin:.3f}, {xmax:.3f}] = {width_m:.3f} m")
print(f"  Y range: [{ymin:.3f}, {ymax:.3f}] = {height_m:.3f} m")
print()

# Canvas calculation (like notebook cell 2be8b4b9)
h, w = 1024, 1024
m_per_px = max(width_m / float(w), height_m / float(h))
W_fit = int(np.ceil(width_m / m_per_px))
H_fit = int(np.ceil(height_m / m_per_px))

ox = (w - W_fit) // 2
oy = (h - H_fit) // 2

print("Canvas calculation (notebook approach):")
print(f"  Input size: {w}x{h}")
print(f"  m/px = max({width_m}/{w}, {height_m}/{h}) = max({width_m/w:.6f}, {height_m/h:.6f}) = {m_per_px:.6f}")
print(f"  Fitted maze: {W_fit}x{H_fit}")
print(f"  Offset: ox={ox}, oy={oy}")
print(f"  Crop would be: x=[{ox}:{ox+W_fit}], y=[{oy}:{oy+H_fit}]")
print(f"  Cropped size: {W_fit}x{H_fit}")
