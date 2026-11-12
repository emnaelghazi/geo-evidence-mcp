"""Typed runtime configuration (R14). All values come from the environment or a local .env."""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import quote

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings.

    Names match .env.example exactly (no prefix). Compose-only variables in .env are ignored.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = "INFO"

    # LLM
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5-5"

    # PostGIS + pgvector
    postgres_user: str = "gem"
    postgres_password: SecretStr | None = None
    postgres_db: str = "gem"
    db_host: str = "localhost"
    db_port: int = 5432

    # Data access.
    # TODO(M1): verify the Earth Search endpoint and collection names against the live API.
    stac_endpoint: str = "https://earth-search.aws.element84.com/v1"

    # Langfuse tracing
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None

    @property
    def database_url(self) -> str:
        """libpq-style URL. Contains the password: never log it."""
        if self.postgres_password is None:
            raise ValueError("POSTGRES_PASSWORD is not set; run `make init`.")
        password = quote(self.postgres_password.get_secret_value(), safe="")
        return (
            f"postgresql://{quote(self.postgres_user, safe='')}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
