"""Fixed personas and Problem model for ai_intake_bot.

Personas are intentionally limited and opinionated. Problems are a typed object
used only with `expert_eval` during persona role-play.
"""
from typing import List, Optional
from pydantic import BaseModel


PERSONAS = {
    "support_agent": {
        "tone": "empathetic",
        "emotional_baseline": "calm",
        "behavior": "ask clarifying questions and offer concrete help",
    },
    "sales_rep": {
        "tone": "upbeat",
        "emotional_baseline": "motivated",
        "behavior": "highlight benefits and close politely",
    },
    "expert_reviewer": {
        "tone": "analytical",
        "emotional_baseline": "neutral",
        "behavior": "evaluate thoroughly and justify suggestions",
    },
}


class Problem(BaseModel):
    """Typed problem scenario for expert evaluations."""

    description: str
    emotional_state: Optional[str] = "neutral"
    goals: Optional[List[str]] = []
    constraints: Optional[List[str]] = []

    class Config:
        extra = "forbid"
