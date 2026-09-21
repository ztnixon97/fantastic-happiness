"""Coordinating a run: what happens next, and when to stop.

Budgets live in :mod:`research.budgets`, a layer below, because everything
charges against them. They are re-exported here for callers that think of
limits as part of orchestration.
"""

from research.budgets import BudgetLedger, BudgetState, Resource
from research.orchestration.scheduler import InvestigationRun, Scheduler
from research.orchestration.stopping import StopDecision, StoppingRules

__all__ = [
    "BudgetLedger",
    "BudgetState",
    "InvestigationRun",
    "Resource",
    "Scheduler",
    "StopDecision",
    "StoppingRules",
]
