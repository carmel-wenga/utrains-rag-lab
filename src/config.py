from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str = "set-your-api-key"
    ELASTICSEARCH_HOST: str = "http://localhost:9200"
    INDEX_NAME: str = "utrains-qa"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    VECTOR_DIMENSION: int = 1536
    STREAMLIT_PORT: int = 8501

    model_config = SettingsConfigDict(env_file="../.env")

settings = Settings()
