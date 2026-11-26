"""Stable constants shared by the reconstruction pipeline."""

ENGINE_VERSION = "mv3d-1.0.0"
SCHEMA_VERSION = 1
PROJECT_RECIPE_VERSION = "mv3d-demo-1"

STAGE_ORDER = (
    "load",
    "features",
    "matches",
    "triangulation",
    "depth",
    "voxels",
)

ALLOWED_FIXED_BITS = (8, 12, 16, 20)
ALLOWED_QUANT_LEVELS = (1, 2, 4, 8)

EXIT_OK = 0
EXIT_VALIDATION_FAILURE = 2
EXIT_POINT_BUDGET_PENDING = 3
EXIT_WORKSPACE_LOCKED = 4
EXIT_INTERNAL_ERROR = 5
EXIT_KILLED = 137

SNAPSHOT_KEEP = 5

STRATEGY_TABLE = (
    {
        "name": "focused",
        "max_views": 3,
        "max_pairs": 4,
        "max_estimated_subvoxels": 384,
        "feature_block": 10,
        "descriptor_chroma_floor": 120,
        "match_radius": 60,
        "max_matches": 32,
        "subdivision_order": "depth_first",
    },
    {
        "name": "balanced",
        "max_views": 6,
        "max_pairs": 10,
        "max_estimated_subvoxels": 1536,
        "feature_block": 12,
        "descriptor_chroma_floor": 100,
        "match_radius": 60,
        "max_matches": 24,
        "subdivision_order": "breadth",
    },
    {
        "name": "wide",
        "max_views": 12,
        "max_pairs": 24,
        "max_estimated_subvoxels": 8192,
        "feature_block": 16,
        "descriptor_chroma_floor": 80,
        "match_radius": 56,
        "max_matches": 16,
        "subdivision_order": "lexicographic",
    },
    {
        "name": "constrained",
        "max_views": 1_000_000,
        "max_pairs": 1_000_000,
        "max_estimated_subvoxels": 1_000_000_000,
        "feature_block": 20,
        "descriptor_chroma_floor": 60,
        "match_radius": 48,
        "max_matches": 12,
        "subdivision_order": "lexicographic",
    },
)

BUSINESS_CODES = (
    "E_PROJECT_SCHEMA_INVALID",
    "E_IMAGE_RESOLUTION_INVALID",
    "E_IMAGE_PIXEL_MODE_INVALID",
    "E_PNG_HEADER_INVALID",
    "E_VIEW_ID_DUPLICATE",
    "E_VIEW_IMAGE_UNIQUE",
    "E_VIEW_IMAGE_MISSING",
    "E_INTRINSIC_NONINVERTIBLE",
    "E_EXTRINSIC_NOT_ORTHOGONAL",
    "E_PAIR_DECLARATION_INVALID",
    "E_PAIR_UNKNOWN_VIEW",
    "E_MIN_DISPARITY_INVALID",
    "E_VOXEL_SIZE_INVALID",
    "E_FIXED_BITS_INVALID",
    "E_QUANT_LEVEL_INVALID",
    "E_RECONSTRUCTION_PARAM_INVALID",
    "E_COMMAND_INVALID",
    "E_TRIANGULATION_DEGENERATE",
    "E_VIEW_QUOTA_EXCEEDED",
    "E_POINT_BUDGET_PENDING",
    "E_WORKSPACE_NOT_WRITABLE",
    "E_WORKSPACE_LOCKED",
    "E_SNAPSHOT_INCOMPATIBLE",
)
