from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import settings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: int = 300
    temperature: float = 0.2
    max_tokens: int = 2000


def get_llm_config() -> LLMConfig:
    return LLMConfig(
        base_url=settings.chatllm.base_url,
        api_key=settings.chatllm.api_key or "EMPTY",
        model=settings.chatllm.model,
    )


def get_vlm_config() -> LLMConfig:
    return LLMConfig(
        base_url=settings.chatvlm.base_url,
        api_key=settings.chatvlm.api_key or "EMPTY",
        model=settings.chatvlm.model,
    )


def is_llm_configured() -> bool:
    return bool(get_llm_config().base_url)


def is_vlm_configured() -> bool:
    return bool(get_vlm_config().base_url)


def build_chat_model(*, streaming: bool, multimodal: bool = False) -> ChatOpenAI:
    cfg = get_vlm_config() if multimodal else get_llm_config()
    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        timeout=cfg.timeout_s,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        streaming=streaming,
        model_kwargs={
            "extra_body": {
                "extra_parameters": {
                    "enable_thinking": True,
                }
            }
        },
    )


def build_messages(
    *,
    system_prompt: Optional[str],
    user_text: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    image_urls: Optional[List[str]] = None,
) -> List:
    out = []
    if system_prompt:
        out.append(SystemMessage(content=system_prompt))
    for msg in messages or []:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content", "")
        if role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    if image_urls:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for url in image_urls:
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})
        out.append(HumanMessage(content=content))
    else:
        out.append(HumanMessage(content=user_text))
    return out


def extract_image_urls_from_text_list(text_list: List[Dict[str, Any]]) -> List[str]:
    urls: List[str] = []
    for item in text_list or []:
        attachments = item.get("attachments_url") or {}
        if isinstance(attachments, dict):
            for url in attachments.values():
                if url:
                    urls.append(str(url))
    return urls


def extract_json(text: str) -> Dict[str, Any] | None:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return None
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
