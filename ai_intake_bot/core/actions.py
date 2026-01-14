"""Agentic recommendation detection.

This module inspects parsed structured output and signals and returns a list of
recommended actions. The logic is rule-based and intentionally simple.
"""
from typing import Dict, List


ACTION_TYPES = ["ALERT", "ESCALATE", "API_CALL", "FOLLOW_UP"]


def detect_actions(parsed_output: Dict) -> List[Dict]:
    """Detect actions based on signals and structured data.

    Rules (v1):
    - If signals.empathy < 0.3 and signals.urgency >= 0.7 -> ALERT HIGH
    - If signals.aggression >= 0.7 -> ESCALATE HIGH
    - If structured_data contains a key 'api_action' -> API_CALL MEDIUM
    - If structured_data missing required contact fields -> FOLLOW_UP LOW
    """
    actions = []
    signals = parsed_output.get("signals", {}) or {}
    sd = parsed_output.get("structured_data", {}) or {}

    def add_action(type_, priority, reason):
        actions.append({"type": type_, "priority": priority, "reason": reason})

    empathy = float(signals.get("empathy", 1.0))
    urgency = float(signals.get("urgency", 0.0))
    aggression = float(signals.get("aggression", 0.0))

    if empathy < 0.3 and urgency >= 0.7:
        add_action("ALERT", "HIGH", "Low empathy and high urgency detected")

    if aggression >= 0.7:
        add_action("ESCALATE", "HIGH", "High aggression detected")

    if sd.get("api_action"):
        add_action("API_CALL", "MEDIUM", "API action requested in structured data")

    # Follow-up if contact info missing from common fields
    contact_fields = ["email", "phone", "contact_info"]
    if not any(k in sd for k in contact_fields):
        add_action("FOLLOW_UP", "LOW", "Missing contact information; follow up required")

    return actions
