"""Credentials and runtime config validation using pydantic.

This module intentionally keeps secrets in memory only and does not persist them.
"""
from typing import Optional
from pydantic import BaseModel, Field


class CredentialsConfig(BaseModel):
    """Simple credentials container for runtime use.

    - Keys are optional during initial phases.
    - Use `redacted()` to get a safe dict for logging.
    """

    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")

    def redacted(self) -> dict:
        """Return a redacted dict for safe logging (do NOT include secrets)."""
        return {
            "openai_api_key": "REDACTED" if self.openai_api_key else None,
            "qdrant_url": self.qdrant_url,
            "qdrant_api_key": "REDACTED" if self.qdrant_api_key else None,
        }

