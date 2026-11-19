"""Fixed, versioned configuration for the deterministic sample and pipeline."""
from __future__ import annotations

SEED = 20261119
TOTAL_POINTS = 120_000

# Cube room: interior spans [0, ROOM] on each axis.  Floor is z=0.
ROOM = 10.0
FLOOR_Z = 0.0
POINT_JITTER = 0.01          # positional Gaussian noise sigma (metres)
DEPTH_NOISE = 0.008          # along-normal Gaussian noise sigma
CONFIDENCE_K = 8             # neighbourhood size baked into generator normals

# Reconstruction defaults
VOXEL_SIZE = 1.0             # patch grid edge length
SUPPORT_RADIUS = 0.85        # point support radius around a patch cell
MESH_GRID_STEP = 0.25        # mesh vertex grid spacing inside a patch
SURFACE_BAND = 0.6           # max |signed distance| for a support point
DEFAULT_NEIGHBOR_RADIUS = 0.35

FRAME_BUDGET_MS = 33.3       # 30 FPS interaction budget
FRAME_BUDGET_SOFT_MS = 25.0

# File names inside the data workspace
POINTS_FILE = "points.pcrc"
NORMALS_FILE = "normals.pcrc"
MANIFEST_FILE = "manifest.json"
CAMERA_FILE = "camera_trajectory.json"
EDITS_FILE = "edits.json"
