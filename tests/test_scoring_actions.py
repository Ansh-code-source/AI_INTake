from ai_intake_bot.core.scoring import compute_score, apply_selection
from ai_intake_bot.core.actions import detect_actions


def test_compute_score_basic():
    signals = {"empathy": 0.8, "clarity": 0.9, "aggression": 0.2}
    score = compute_score(signals)
    assert 0 <= score <= 1
    # empathy and clarity good, aggression low -> high score
    assert score > 0.7


def test_selection_logic():
    assert apply_selection(0.8, 0.75) is True
    assert apply_selection(0.6, 0.7) is False


def test_detect_actions_rules():
    parsed = {"signals": {"empathy": 0.1, "urgency": 0.8}, "structured_data": {}}
    actions = detect_actions(parsed)
    assert any(a["type"] == "ALERT" for a in actions)

    parsed2 = {"signals": {"aggression": 0.9}, "structured_data": {"api_action": "do_something"}}
    actions2 = detect_actions(parsed2)
    assert any(a["type"] == "ESCALATE" for a in actions2)
    assert any(a["type"] == "API_CALL" for a in actions2)
