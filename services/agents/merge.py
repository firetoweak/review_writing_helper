from __future__ import annotations

import json
from typing import Dict

from models.llm_interface_async import build_chat_model, build_messages, is_llm_configured


class MergeAgent:
    def merge_texts(self, payload: Dict) -> Dict:
        write_rule = payload.get("writeRule") or {}
        text_list = payload.get("textList", [])
        session_list = payload.get("sessionList", [])
        history_text = payload.get("historyTextList", [])
        prompt = payload.get("prompt") or {}
        merge_prompt = prompt.get("mergePrompt") or ""

        if is_llm_configured():
            merged_text = self._llm_merge(write_rule, text_list, session_list, merge_prompt, history_text)
        else:
            merged_text = self._fallback_merge(text_list, session_list, merge_prompt)
        return {"text": merged_text}

    def _llm_merge(
        self,
        write_rule: dict,
        text_list: list,
        session_list: list,
        merge_prompt: str,
        history_text: list,
    ) -> str:
        model = build_chat_model(streaming=False)
        system_prompt = "你是合入重写助手，请严格输出 JSON。"
        user_payload = {
            "writeRule": write_rule,
            "textList": text_list,
            "sessionList": session_list,
            "historyTextList": history_text,
            "mergePrompt": merge_prompt,
            "output_format": {"text": "string"},
        }
        lc_messages = build_messages(
            system_prompt=system_prompt,
            user_text=json.dumps(user_payload, ensure_ascii=False),
            messages=None,
        )
        result = model.invoke(lc_messages)
        content = getattr(result, "content", "") or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return self._fallback_merge(text_list, session_list, merge_prompt)
        if isinstance(data, dict) and "text" in data:
            return data.get("text") or ""
        return self._fallback_merge(text_list, session_list, merge_prompt)

    def _fallback_merge(self, text_list: list, session_list: list, merge_prompt: str) -> str:
        suggestions = []
        for session in session_list:
            for msg in session.get("messages", []):
                if msg.get("role") == "assistant":
                    suggestions.append(msg.get("content", ""))
        text_blocks = []
        for item in text_list:
            title = item.get("sectionTitle") or ""
            base_text = item.get("text", "")
            block = "\n".join(filter(None, [title, base_text]))
            if block:
                text_blocks.append(block)
        extra = "\n".join(suggestions[:2])
        if merge_prompt:
            extra = "\n".join(filter(None, [extra, f"参考提示：{merge_prompt}"]))
        merged = "\n\n".join(text_blocks)
        return "\n\n".join(filter(None, [merged, extra]))
