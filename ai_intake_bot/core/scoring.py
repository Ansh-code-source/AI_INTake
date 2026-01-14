"""Scoring utilities.

Converts signal values (0-1) into a candidate score (0-1) using a simple
unweighted mean of normalized signals. Aggression is treated as negative
influence (inverted). This is deterministic and avoids training.
"""
from typing import Dict, Iterable


EXPECTED_SIGNALS = ["empathy", "clarity", "aggression", "urgency"]


def _normalize_signal(name: str, value: float) -> float:
    """Normalize a single signal into a 0-1 positive contribution.

    - For 'aggression', higher values are worse, so we invert: contribution = 1 - value.
    - For other signals, assume higher is better and clamp to [0,1].
    """
    v = max(0.0, min(1.0, float(value)))
    if name == "aggression":
        return 1.0 - v
    return v


def compute_score(signals: Dict[str, float]) -> float:
    """Compute a candidate score between 0 and 1 from signals.

    Missing signals are ignored. If no valid signals present, return 0.0.
    """
    contributions = []
    for name in EXPECTED_SIGNALS:
        if name in signals:
            try:
                contributions.append(_normalize_signal(name, signals[name]))
            except Exception:
                continue
    if not contributions:
        return 0.0
    return sum(contributions) / len(contributions)

def apply_selection(candidate_score: float, selection_probability: float) -> bool:
    """Simple deterministic selection rule: recommend selection if candidate_score >= selection_probability."""
    if selection_probability is None:
        return None
    return float(candidate_score) >= float(selection_probability)
