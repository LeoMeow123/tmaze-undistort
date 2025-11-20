import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.lines import Line2D
from skimage.transform import estimate_transform
from scipy.interpolate import Rbf
from scipy.linalg import norm
from scipy.ndimage import map_coordinates
import sleap_io as sio
from tqdm import tqdm

