from ai_intake_bot.core.engine import IntakeBot
from ai_intake_bot.core.llm import FakeLLM


def test_scoring_inline_for_complaint():
    # FakeLLM will detect scoring prompt by substring and return signals
    signals_resp = {"signals": {"empathy": 0.2, "clarity": 0.5, "aggression": 0.8, "urgency": 0.9}}
    fake = FakeLLM(responses={"for each of the following signals": signals_resp})

    bot = IntakeBot(
        mode="persona",
        template="complaint",
        persona="support_agent",
        problem=None,
        api_key="sk-test",
        qdrant_url=None,
        qdrant_api_key=None,
        files=None,
        selection_probability=0.6,
        enable_alerts=True,
        extra_system_prompt=None,
    )
    bot.set_llm(fake)
    out = bot.handle("Customer angry about billing")
    assert isinstance(out.get("signals"), dict)
    assert out["candidate_score"] is not None
    # With aggression high and urgency high, score should be lower due to aggression inversion
    assert 0 <= out["candidate_score"] <= 1


def test_expert_eval_two_stage():
    # First call (role-play) returns candidate reply only; second call (evaluation) returns signals
    role_resp = {"reply": "Candidate response: resolve issue by asking for details", "structured_data": {}}
    eval_resp = {"signals": {"empathy": 0.9, "clarity": 0.8}, "candidate_score": 0.85}

    fake = FakeLLM(responses={
        "respond with valid json matching the output contract": role_resp,
        "you are an expert evaluator": eval_resp,
    })

    bot = IntakeBot(
        mode="persona",
        template="expert_eval",
        persona="expert_reviewer",
        problem={"description": "Cannot login", "emotional_state": "frustrated", "goals": ["restore access"]},
        api_key="sk-test",
        qdrant_url=None,
        qdrant_api_key=None,
        files=None,
        selection_probability=0.5,
        enable_alerts=True,
        extra_system_prompt=None,
    )
    bot.set_llm(fake)
    out = bot.handle("Please evaluate")
    assert out["reply"].startswith("Candidate response")
    assert isinstance(out.get("signals"), dict)
    assert out["candidate_score"] == 0.85
