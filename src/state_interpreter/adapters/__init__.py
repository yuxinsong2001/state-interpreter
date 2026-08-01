"""Adapter contracts for external encoders and data sources."""

from .interfaces import DatasetAdapter, GearboxAdapter, VibFMAdapter
from .xjtu_sy import XJTUSYDatasetAdapter

__all__ = [
    "DatasetAdapter",
    "GearboxAdapter",
    "VibFMAdapter",
    "XJTUSYDatasetAdapter",
]
