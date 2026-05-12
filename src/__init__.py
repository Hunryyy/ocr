"""
src — Core document parsing modules.

Re-exports the primary public API for convenience.
"""

from src.constants import (
    SCHEMA_VERSION,
    BLOCK_FEAT_DIM,
    PAIR_FEAT_DIM,
    NUM_CLASSES,
    LABEL_MAP,
    TEXT_TYPES_NESTED_SUPPRESS,
    TEXT_TYPES_PROJECTION,
    IGNORED_TYPES,
    FLOAT_TYPES,
)

from src.layout_postprocess import (
    suppress_nested_detections,
    refine_title_paragraph_blocks,
)

from src.order_features import compute_advanced_pair_features

from src.reading_order_utils import (
    detect_columns_by_projection,
    assign_block_columns,
    compute_page_median_gap,
)

from src.reading_order_pipeline import (
    xycut_graph_sort,
    build_chain_order_edges,
)
