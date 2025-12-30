from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


class LLMInterfaceAsync:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "")

    def is_configured(self) -> bool:
        return bool(self.base_url)

    async def chat(self, messages: List[dict], max_tokens: int = 2000) -> str:
        if not self.base_url:
            return ""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


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
