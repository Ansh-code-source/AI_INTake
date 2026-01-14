import shutil
import subprocess
import pytest

from ai_intake_bot.core.voice import NoOpVoice, MacOSSayVoice
from ai_intake_bot.core.engine import IntakeBot
from ai_intake_bot.core.llm import FakeLLM


def test_noop_voice():
    v = NoOpVoice()
    assert v.speak("hello") is None


@pytest.mark.skipif(shutil.which("say") is None, reason="macOS 'say' not available")
def test_mac_say_voice_runs(monkeypatch, tmp_path):
    v = MacOSSayVoice()
    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = v.speak("hi there")
    assert out is None
    assert any("say" in c[0] for c in calls)

    # Test file output
    fname = str(tmp_path / "out.aiff")
    calls.clear()
    outfile = v.speak("save me", filename=fname)
    assert outfile == fname
    assert any("-o" in c for c in calls)


def test_intakebot_set_voice_and_speak(monkeypatch):
    bot = IntakeBot(
        mode="persona",
        template="survey",
        persona="support_agent",
        problem=None,
        api_key="sk-test",
        qdrant_url=None,
        qdrant_api_key=None,
        files=None,
        selection_probability=None,
        enable_alerts=False,
        extra_system_prompt=None,
    )
    class FakeVoice:
        def __init__(self):
            self.last = None

        def speak(self, text, filename=None):
            self.last = text
            return None

    v = FakeVoice()
    bot.set_voice(v)

    # use FakeLLM to produce a reply
    bot.set_llm(FakeLLM())
    out = bot.handle("Hello")
    assert v.last == out["reply"]
