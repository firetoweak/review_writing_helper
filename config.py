from dataclasses import dataclass
import os
from pathlib import Path
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


def _get_env_or(value: str | None, env_key: str) -> str:
    env_val = os.environ.get(env_key)
    if env_val is not None and env_val != "":
        return env_val
    return value or ""


def _load_yaml_simple(path: Path) -> dict:
    if not path.exists():
        return {}
    data: dict = {}
    current_section: Optional[str] = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and ":" not in line[:-1]:
            current_section = line[:-1].strip()
            data[current_section] = {}
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip() or ""
            if current_section:
                data.setdefault(current_section, {})[key] = value
            else:
                data[key] = value
    return data


@dataclass(frozen=True)
class ChatVLMSettings:
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass(frozen=True)
class ChatLLMSettings:
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass(frozen=True)
class Settings:
    chatvlm: ChatVLMSettings = ChatVLMSettings()
    chatllm: ChatLLMSettings = ChatLLMSettings()


def load_settings() -> Settings:
    config_path = Path(__file__).with_name("config.yaml")
    data = _load_yaml_simple(config_path)
    chatvlm = data.get("chatvlm", {})
    chatllm = data.get("chatllm", {})
    return Settings(
        chatvlm=ChatVLMSettings(
            base_url=_get_env_or(chatvlm.get("base_url"), "CHATVLM_BASE_URL"),
            api_key=_get_env_or(chatvlm.get("api_key"), "CHATVLM_API_KEY"),
            model=_get_env_or(chatvlm.get("model"), "CHATVLM_MODEL"),
        ),
        chatllm=ChatLLMSettings(
            base_url=_get_env_or(chatllm.get("base_url"), "CHATLLM_BASE_URL"),
            api_key=_get_env_or(chatllm.get("api_key"), "CHATLLM_API_KEY"),
            model=_get_env_or(chatllm.get("model"), "CHATLLM_MODEL"),
        ),
    )


settings = load_settings()
