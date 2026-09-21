"""A read-only view over stored investigation state.

Kept entirely outside the research core: nothing in ``research.ui`` is
imported by anything below it, and nothing it exposes can write, search or
spend. Delete this package and the research system is unchanged.
"""

from research.ui.api import InvestigationView, investigations
from research.ui.server import UiServer, serve

__all__ = ["InvestigationView", "UiServer", "investigations", "serve"]
