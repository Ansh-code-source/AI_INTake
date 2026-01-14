"""Opinionated templates (survey, contact, complaint, expert_eval).

Templates are static in v1. Each template describes required fields, extraction
goals, and whether alerts/scoring are enabled.
"""

TEMPLATES = {
    "survey": {
        "required_fields": ["name", "responses"],
        "extraction_goals": ["insights"],
        "alerts": False,
        "scoring": False,
    },
    "contact": {
        "required_fields": ["name", "email"],
        "extraction_goals": ["contact_info"],
        "alerts": False,
        "scoring": False,
    },
    "complaint": {
        "required_fields": ["order_id", "issue"],
        "extraction_goals": ["severity", "actions"],
        "alerts": True,
        "scoring": True,
    },
    "expert_eval": {
        "required_fields": ["scenario", "role_play"],
        "extraction_goals": ["evaluation"],
        "alerts": True,
        "scoring": True,
    },
}


def get_template(name: str) -> dict:
    """Return template dict or raise KeyError if unknown."""
    if name not in TEMPLATES:
        raise KeyError(f"Unknown template: {name}")
    return TEMPLATES[name]


def compose_template_prompt(template_name: str) -> str:
    """Return a short prompt instructing the model about template goals and required fields."""
    t = get_template(template_name)
    reqs = ", ".join(t["required_fields"])
    goals = ", ".join(t["extraction_goals"])
    return (
        f"Extract the following fields: {reqs}. The extraction goals are: {goals}. "
        f"If a required field is missing, return it with an empty value."
    )
