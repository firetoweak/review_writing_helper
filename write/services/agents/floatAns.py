from __future__ import annotations

import json
from typing import Dict, Any, List

from models.llm_interface_async import build_chat_model, build_messages, is_vlm_configured
import asyncio

class FloatAgent:
    async def float_response(self, payload: Dict) -> Dict:
        promptList = payload.get("prompt", "") or ""
        float_prompt = promptList.get("floatPrompt", "") or ""


        title = payload.get("sectionTitle", "") or ""
        text = self._serialize_text_list(payload.get("textList", ""))
        target_text = payload.get("targetText")
        user_input = payload.get("userInput") or ""

        # 可选：把 title 并入 text，给模型更多语境（不新增字段）
        if title:
            text = f"标题：{title}\n\n{text}"

        if is_vlm_configured():
            floatText = await self._llm_float_response(text, target_text, user_input, float_prompt)
        else:
            floatText = self._fallback_float_response(text, target_text, user_input, float_prompt)

        return {
            "floatText": floatText.get("floatText", ""),
        }

    def _fallback_float_response(self, text: str, target_text: str, user_input: str, prompt: str) -> Dict:
        content = prompt or (target_text if target_text else f"已接收请求，用户输入为：{user_input}")
        return {"content": content}

    async def _llm_float_response(self, text: str, target_text: str, user_input: str, prompt: str) -> Dict:
        model = build_chat_model(streaming=False)
        user_payload = {
            "text": text,
            "targetText": target_text or "",
            "userInput": user_input or "",
        }
        lc_messages = build_messages(
            system_prompt=prompt,
            user_text=json.dumps(user_payload, ensure_ascii=False),
            messages=None,
        )
        result = await model.ainvoke(lc_messages)
        floatText = getattr(result, "content", "") or ""
        return {"floatText": floatText}
    
    def _serialize_text_list(self, text_list: Any) -> str:
        """
        textList: [{sectionId, sectionTitle, text}, ...] -> 稳定字符串
        """
        if not text_list or not isinstance(text_list, list):
            return ""
        lines: List[str] = []
        for it in text_list:
            if not isinstance(it, dict):
                continue
            sid = str(it.get("sectionId") or "").strip()
            st = str(it.get("sectionTitle") or "").strip()
            tx = str(it.get("text") or "").strip()
            head = f"{sid} {st}".strip()
            if head:
                lines.append(head)
            if tx:
                lines.append(tx)
            lines.append("")
        return "\n".join(lines).strip()