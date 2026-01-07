from __future__ import annotations

import json
from typing import Dict

from models.llm_interface_async import build_chat_model, build_messages, is_llm_configured


class MergeAgent:
    def merge_texts(self, payload: Dict) -> Dict:
        node_id = payload.get("nodeId", "")
        title = payload.get("title", "")
        texts = payload.get("text", [])
        session_list = payload.get("sessionList", [])
        merge_prompt = payload.get("mergePrompt") or ""
        history_text = payload.get("historyText", [])

        if is_llm_configured():
            merged_texts = self._llm_merge(title, texts, session_list, merge_prompt, history_text)
        else:
            merged_texts = self._fallback_merge(texts, session_list, merge_prompt)
        return {
            "nodeId": node_id,
            "title": title,
            "task": payload.get("task", "merge"),
            "texts": merged_texts,
        }

    def _llm_merge(
        self,
        title: str,
        texts: list,
        session_list: list,
        merge_prompt: str,
        history_text: list,
    ) -> list:
        model = build_chat_model(streaming=False)
        system_prompt = "你是合入重写助手，请严格输出 JSON。"
        user_payload = {
            "title": title,
            "texts": texts,
            "sessionList": session_list,
            "historyText": history_text,
            "mergePrompt": merge_prompt,
            "output_format": [
                {"nodeId": "string", "level": "number", "title": "string", "text": "string"}
            ],
        }
        lc_messages = build_messages(
            system_prompt=system_prompt,
            user_text=json.dumps(user_payload, ensure_ascii=False),
            messages=None,
        )
        result = model.invoke(lc_messages)
        content = getattr(result, "content", "") or "[]"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return self._fallback_merge(texts, session_list, merge_prompt)
        if isinstance(data, dict) and "texts" in data:
            data = data.get("texts", [])
        if not isinstance(data, list):
            return self._fallback_merge(texts, session_list, merge_prompt)
        return data

    def _fallback_merge(self, texts: list, session_list: list, merge_prompt: str) -> list:
        merged_texts = []
        suggestions = []
        for session in session_list:
            for msg in session.get("messages", []):
                if msg.get("role") == "assistant":
                    suggestions.append(msg.get("content", ""))
        for item in texts:
            base_text = item.get("text", "")
            extra = "\n".join(suggestions[:2])
            if merge_prompt:
                extra = "\n".join(filter(None, [extra, f"参考提示：{merge_prompt}"]))
            merged_texts.append(
                {
                    "nodeId": item.get("nodeId"),
                    "level": item.get("level"),
                    "title": item.get("title"),
                    "text": base_text + ("\n" + extra if extra else ""),
                }
            )
        return merged_texts
