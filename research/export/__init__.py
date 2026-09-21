"""Exporting investigations into other tools.

A presentation layer, like ``research.ui``: nothing below it imports it, and
removing it changes nothing about how research runs.
"""

from research.export.obsidian import ExportResult, ObsidianExporter

__all__ = ["ExportResult", "ObsidianExporter"]
