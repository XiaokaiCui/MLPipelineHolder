"""Shared marker base for pipeline holders.

Internal modules recognise holder instances through this type instead of importing the concrete
PipelineHolder class, which would create import cycles.

execution/block.py and execution/gate_block.py keep a TYPE_CHECKING import of PipelineHolder because they read a
large holder surface (about 28 members); routing them through this marker would require declaring that whole
surface here, so their two type-checker import cycles are accepted as pre-existing.
"""

from __future__ import annotations


class PipelineBase:
    """Marker base implemented by PipelineHolder."""
