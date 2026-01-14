"""Optional voice (TTS) adapters for ai_intake_bot.

Design principles:
- Opt-in: user must call `IntakeBot.set_voice(voice)` to enable speaking.
- Minimal, local-first: prefer macOS `say` command; fallback to `pyttsx3` if available.
- No external network calls by default.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Optional


class BaseVoice:
    """Abstract voice interface."""

    def speak(self, text: str, filename: Optional[str] = None) -> Optional[str]:
        """Speak `text` or write to `filename` if provided. Return path to file if written."""
        raise NotImplementedError


class NoOpVoice(BaseVoice):
    def speak(self, text: str, filename: Optional[str] = None) -> Optional[str]:
        # Do nothing
        return None


class MacOSSayVoice(BaseVoice):
    """Uses macOS `say` command. If `filename` is provided, writes out an AIFF file.

    Example: `say -o out.aiff "Hello"` writes an AIFF file. When no filename is
    provided, it plays audio via the system `say` command.
    """

    def __init__(self):
        if shutil.which("say") is None:
            raise EnvironmentError("macOS 'say' command not found on PATH")

    def speak(self, text: str, filename: Optional[str] = None) -> Optional[str]:
        if filename:
            # say -o supports AIFF output
            subprocess.run(["say", "-o", filename, text], check=True)
            return filename
        else:
            subprocess.run(["say", text], check=True)
            return None


class PyTTSX3Voice(BaseVoice):
    """Uses pyttsx3 package to speak or write to file if supported.

    Falls back to raising ImportError if pyttsx3 is not available.
    """

    def __init__(self):
        try:
            import pyttsx3  # type: ignore
        except Exception as e:
            raise ImportError("pyttsx3 is required for PyTTSX3Voice") from e
        self.engine = pyttsx3.init()

    def speak(self, text: str, filename: Optional[str] = None) -> Optional[str]:
        if filename:
            # Some pyttsx3 drivers support save_to_file
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
            return filename
        else:
            self.engine.say(text)
            self.engine.runAndWait()
            return None
