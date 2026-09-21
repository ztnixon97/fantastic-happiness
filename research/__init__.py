"""A persistent, evidence-driven research environment.

The package is organised around the research domain rather than around a
generic agent loop:

``research.models``
    Domain objects (Investigation, ResearchTask, EvidenceDocument, Claim,
    Entity, Event, Provenance).
``research.normalize``
    Deterministic normalisation: URLs, DOIs, text, HTML, fingerprints.
``research.sources``
    Provider adapters hidden behind a provider-independent interface.
``research.storage``
    SQLite-backed persistence; investigation state lives here, not in a
    model context window.
``research.acquisition``
    Turning search hits into normalised, deduplicated evidence.
``research.graph``
    Citation / evidence graph traversal.
``research.orchestration``
    Budgets and (later) planning and stopping rules.
``research.operations``
    Research-native operations composed from the layers above.
``research.cli``
    Inspection and execution entry points.
"""

__version__ = "0.1.0"
