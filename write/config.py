# config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict
import os

# 尝试导入 PyYAML
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# ====== 路径：固定从项目根目录找 config.yaml ======
# 这里假设 config.py 在项目根目录（你 tree 里就是这样）
ROOT = Path(__file__).resolve().parent
DEFAULT_YAML_PATH = ROOT / "config.yaml"


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML 未安装，无法读取 config.yaml。请 pip install pyyaml")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _get(d: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    """安全取嵌套字段：_get(cfg, 'chatllm','model')"""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _env_or(value: Any, env_key: str, default: Any = "") -> Any:
    """允许环境变量覆盖 yaml（可选）"""
    env_v = os.environ.get(env_key)
    if env_v is not None and str(env_v).strip() != "":
        return env_v
    return value if value not in (None, "") else default


# ====== 你原来的 AppConfig 先保留（如果你还要用）======
@dataclass(frozen=True)
class AppConfig:
    app_name: str = "review-writing-helper"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    checkpoint_dsn: Optional[str] = None
    db_name: Optional[str] = None
    db_host: Optional[str] = None
    db_port: Optional[int] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None


DEFAULT_CONFIG = AppConfig()


def load_config(yaml_path: Optional[str] = None) -> AppConfig:
    """如果你还在用 AppConfig，就让它从 yaml 读，而不是永远 DEFAULT_CONFIG"""
    cfg = _read_yaml(Path(yaml_path) if yaml_path else DEFAULT_YAML_PATH)
    return AppConfig(
        app_name=str(cfg.get("app_name") or DEFAULT_CONFIG.app_name),
        llm_api_key=cfg.get("llm_api_key"),
        llm_base_url=cfg.get("llm_base_url"),
        checkpoint_dsn=cfg.get("checkpoint_dsn"),
        db_name=cfg.get("db_name"),
        db_host=cfg.get("db_host"),
        db_port=cfg.get("db_port"),
        db_user=cfg.get("db_user"),
        db_password=cfg.get("db_password"),
    )


# ====== 你的 ChatLLM / ChatVLM Settings：从 YAML 读取，再允许 env 覆盖 ======
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
class ModelEndpoint:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_tokens: int = 0


@dataclass(frozen=True)
class Settings:
    chatvlm: ChatVLMSettings
    chatllm: ChatLLMSettings
    models: Dict[str, ModelEndpoint] = field(default_factory=dict)  # ✅ 默认空



def load_settings(yaml_path: Optional[str] = None) -> Settings:
    cfg = _read_yaml(Path(yaml_path) if yaml_path else DEFAULT_YAML_PATH)

    chatvlm = ChatVLMSettings(
        base_url=str(_env_or(_get(cfg, "chatvlm", "base_url", default=""), "CHATVLM_BASE_URL", "")),
        api_key=str(_env_or(_get(cfg, "chatvlm", "api_key", default=""), "CHATVLM_API_KEY", "")),
        model=str(_env_or(_get(cfg, "chatvlm", "model", default=""), "CHATVLM_MODEL", "")),
    )

    chatllm = ChatLLMSettings(
        base_url=str(_env_or(_get(cfg, "chatllm", "base_url", default=""), "CHATLLM_BASE_URL", "")),
        api_key=str(_env_or(_get(cfg, "chatllm", "api_key", default=""), "CHATLLM_API_KEY", "")),
        model=str(_env_or(_get(cfg, "chatllm", "model", default=""), "CHATLLM_MODEL", "")),
    )
    # ✅ 新增：读取 models 路由表
    models_raw = cfg.get("models") or {}
    models: Dict[str, ModelEndpoint] = {}
    if isinstance(models_raw, dict):
        for k, v in models_raw.items():
            if not isinstance(v, dict):
                continue
            models[str(k)] = ModelEndpoint(
                base_url=str(v.get("base_url") or ""),
                api_key=str(v.get("api_key") or ""),
                model=str(v.get("model") or ""),
                max_tokens=int(v.get("max_tokens") or 0),
            )


    return Settings(chatvlm=chatvlm, chatllm=chatllm, models=models)


# 全局 settings：真正从 config.yaml 来
settings = load_settings()
