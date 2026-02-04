from __future__ import annotations

import json
from typing import Dict, List

import asyncio
from models.llm_interface_async import build_chat_model, build_messages, is_vlm_configured


class IndustryAgent:
    async def industry(self, payload: Dict) -> Dict:
        title = payload.get("title", "") or ""
        idea = payload.get("idea", "") or ""
        industryNameList = payload.get("industryNameList", []) or []
        prompt_obj = payload.get("prompt") or {}
        industryPrompt = prompt_obj.get("industryPrompt", "") or ""

        if is_vlm_configured():
            content = await self._llm_industry(title, idea, industryNameList, industryPrompt)
        else:
            content = self._fallback_industry(title, idea, industryNameList, industryPrompt)

        return {
            "industryName": content.get("content", ""),
        }
    
    def _fallback_industry(self, title: str, idea: str, industryNameList: List[str], industryPrompt: str) -> Dict:
        return {
            "task": "industry",
            "content": f"{title}\n{idea}\n{industryNameList}",
        }

    async def _llm_industry(self, title: str, idea: str, industryNameList: List[str], industryPrompt: str) -> Dict:
        model = build_chat_model(streaming=False)
        user_payload = {
            "title": title,
            "idea": idea,
            "industryNameList": industryNameList,
        }
        lc_messages = build_messages(
            system_prompt=industryPrompt,
            user_text=json.dumps(user_payload, ensure_ascii=False),
            messages=None,
        )
        print("industryPrompt:", industryPrompt)
        print(lc_messages)
        result = await model.ainvoke(lc_messages)
        content = getattr(result, "content", "") or ""
        return {
            "content": content,
        }