from ai_intake_bot.core import prompts
from ai_intake_bot.core.personas import PERSONAS, Problem


def test_compose_base_with_extra():
    base = prompts.compose_base_system_prompt("Extra guidance")
    assert "Extra guidance" in base


def test_persona_prompt_separation():
    # Persona prompt must not include evaluation keywords like 'evaluate', 'score', or 'assess'
    persona = PERSONAS["support_agent"]
    p = prompts.compose_persona_role_prompt("support_agent", persona)
    lower = p.lower()
    for forbidden in ("evaluate", "evaluator", "score", "assess"):
        assert forbidden not in lower
    assert persona["behavior"].split()[0] in p


def test_problem_prompt_inclusion():
    prob = Problem(description="User is upset", emotional_state="distressed", goals=["refund"], constraints=["must verify identity"])
    pp = prompts.compose_problem_scenario_prompt(prob.dict())
    assert "User is upset" in pp
    assert "refund" in pp


def test_evaluation_prompt_separation():
    ev = prompts.compose_evaluation_prompt("Assess empathy")
    assert "evaluator" in ev.lower()
    # Must not instruct role-play
    assert "role-playing" not in ev.lower()
