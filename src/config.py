from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_secret_file() -> dict:
    """Load settings from the local Streamlit secrets file when running outside Streamlit."""
    secret_path = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    if not secret_path.exists():
        return {}

    with secret_path.open("rb") as handle:
        return tomllib.load(handle)


def _load_streamlit_secrets() -> dict:
    """Read configuration from st.secrets when the app is running inside Streamlit."""
    try:
        import streamlit as st
    except Exception:
        return {}

    if not hasattr(st, "secrets"):
        return {}

    try:
        return dict(st.secrets)
    except Exception:
        return {}


class Settings(BaseSettings):
    openai_api_key: str = ""
    elasticsearch_host: str = "http://localhost:9200"
    elasticsearch_username: str = ""
    elasticsearch_password: str = ""
    index_name: str = "utrains-qa"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    vector_dimension: int = 1536
    streamlit_port: int = 8501

    model_config = SettingsConfigDict(extra="ignore")

    @classmethod
    def load(cls) -> "Settings":
        """Merge Streamlit secrets and local secret-file values. Streamlit wins when both exist."""
        values = _load_secret_file()
        streamlit_values = _load_streamlit_secrets()
        values.update(streamlit_values)
        return cls(**values)


settings = Settings.load()
