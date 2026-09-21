"""Research roles.

Specialised workers over shared infrastructure. Each one gets the same
storage, budgets and provenance; what differs is the method it is asked to
follow and the slice of the action vocabulary it may use.
"""

from research.agents.actions import ACTIONS, ActionSpec, actions_for, render_catalogue
from research.agents.planner import Plan, Planner
from research.agents.runtime import ActionRequest, ActionRuntime, Observation
from research.agents.worker import ResearchWorker, WorkerOutcome

__all__ = [
    "ACTIONS",
    "ActionRequest",
    "ActionRuntime",
    "ActionSpec",
    "Observation",
    "Plan",
    "Planner",
    "ResearchWorker",
    "WorkerOutcome",
    "actions_for",
    "render_catalogue",
]
