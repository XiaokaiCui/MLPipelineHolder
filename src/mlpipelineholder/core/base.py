"""Shared marker base for pipeline holders.

Internal modules recognise holder instances through this type instead of importing the concrete
PipelineHolder class, which would create import cycles.
"""

from __future__ import annotations


class PipelineBase:
    """Marker base implemented by PipelineHolder."""
