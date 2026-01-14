from ai_intake_bot.core.personas import PERSONAS, Problem
from ai_intake_bot.core import templates


def test_personas_defined():
    assert "support_agent" in PERSONAS
    assert "sales_rep" in PERSONAS


def test_problem_model_validation():
    p = Problem(description="Example")
    assert p.description == "Example"


def test_templates_get():
    t = templates.get_template("complaint")
    assert t["alerts"] is True
    assert t["scoring"] is True
