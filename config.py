from dataclasses import dataclass
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
