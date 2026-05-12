"""
Shared constants for the document parsing system.

Single source of truth for feature dimensions, label maps, and schema version.
Import from here to prevent the schema-mismatch class of bugs.
"""

SCHEMA_VERSION = "2.3"

BLOCK_FEAT_DIM = 33
PAIR_FEAT_DIM = 51

NUM_CLASSES = 11

LABEL_MAP = [
    "title",
    "paragraph",
    "list_item",
    "caption",
    "table",
    "figure",
    "formula",
    "header",
    "footer",
    "chart",
    "unknown",
]

# Text-like types used for nested box suppression (broad: includes headers/footers/page_numbers)
TEXT_TYPES_NESTED_SUPPRESS = {
    "paragraph", "title", "list_item", "caption", "header", "footer", "page_number",
}

# Text-like types used for column projection (narrower: formula replaces header/footer/page_number)
TEXT_TYPES_PROJECTION = {
    "paragraph", "title", "list_item", "caption", "formula",
}

# Types ignored in reading-order column assignment
IGNORED_TYPES = {"header", "footer", "page_number"}

# Floating/embedded element types
FLOAT_TYPES = {"figure", "table", "chart"}

# Header / footer types for reading-order separation
HEADER_TYPES = {"header"}
FOOTER_TYPES = {"footer", "page_number"}

# Types that can be used as caption targets
CAPTION_TARGET_TYPES = FLOAT_TYPES
