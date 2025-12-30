from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "review-writing-helper"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    db_host: Optional[str] = None
    db_port: Optional[int] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None


DEFAULT_CONFIG = AppConfig()


def load_config() -> AppConfig:
    return DEFAULT_CONFIG


@dataclass(frozen=True)
class ChatVLMSettings:
    base_url: str = os.environ.get("CHATVLM_BASE_URL", "")
    api_key: str = os.environ.get("CHATVLM_API_KEY", "")
    model: str = os.environ.get("CHATVLM_MODEL", "")


@dataclass(frozen=True)
class ChatLLMSettings:
    base_url: str = os.environ.get("CHATLLM_BASE_URL", "")
    api_key: str = os.environ.get("CHATLLM_API_KEY", "")
    model: str = os.environ.get("CHATLLM_MODEL", "")


@dataclass(frozen=True)
class Settings:
    chatvlm: ChatVLMSettings = ChatVLMSettings()
    chatllm: ChatLLMSettings = ChatLLMSettings()


settings = Settings()
