"""
Centralized settings. Everything is pulled from environment variables (.env).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Avoid pydantic protected namespace conflicts for fields like `model_fast`
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=("settings_",))

    # OpenRouter
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_fast: str = "meta-llama/llama-3.1-8b-instruct"
    model_balanced: str = "openai/gpt-4o-mini"
    model_powerful: str = "anthropic/claude-3.5-sonnet"

    # NeonDB
    database_url: str
    database_url_unpooled: str

    # Tools
    tavily_api_key: str | None = None

    app_env: str = "development"


settings = Settings()
