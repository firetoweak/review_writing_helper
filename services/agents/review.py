from __future__ import annotations

import json
from typing import Dict, List

from models.llm_interface_async import build_chat_model, build_messages, is_llm_configured


class ReviewAgent:
    def review_section(self, payload: Dict) -> Dict:
        node_id = payload.get("nodeId", "")
        title = payload.get("title", "")
        text_items = payload.get("text", [])
        review_prompt = payload.get("reviewpPrompt") or ""
        industry = payload.get("industry") or ""
        full_text = "\n".join(item.get("text", "") for item in text_items)
        if is_llm_configured():
            review = self._llm_review(title, full_text, review_prompt, industry)
        else:
            review = self._fallback_review(title, full_text, review_prompt)
        return {
            "task": payload.get("task", "sectionReview"),
            "title": title,
            "nodeId": node_id,
            "review": review,
        }

    def full_review(self, payload: Dict) -> Dict:
        full_text = payload.get("fullText", [])
        full_review_prompt = payload.get("fullReviewPrompt") or ""
        total_sections = sum(len(section.get("children", [])) for section in full_text)
        report = full_review_prompt or f"已接收全文评审请求（{len(full_text)} 个章节，{total_sections} 个小节）。"
        return {"task": payload.get("task", "fullReview"), "fullReviewAns": report}

    def _summary_from_prompt(self, prompt: str) -> List[str]:
        if not prompt:
            return []
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        if lines:
            return lines[:3]
        return [prompt.strip()] if prompt.strip() else []

    def _fallback_review(self, title: str, full_text: str, prompt: str) -> Dict:
        score = min(10, max(1, len(full_text) // 120 + 3))
        summary = self._summary_from_prompt(prompt)
        detail = prompt or f"已接收《{title}》评审请求。"
        help_list = self._summary_from_prompt(prompt)
        return {
            "score": score,
            "summaryList": summary,
            "detail": detail,
            "helpList": help_list,
        }

    def _llm_review(self, title: str, full_text: str, prompt: str, industry: str) -> Dict:
        model = build_chat_model(streaming=False)
        system_prompt = "你是资深评审助手，输出严格 JSON。"
        user_payload = {
            "title": title,
            "industry": industry,
            "text": full_text,
            "prompt": prompt,
            "output_format": {
                "score": "1-10",
                "summaryList": ["string"],
                "detail": "string",
                "helpList": ["string"],
            },
        }
        lc_messages = build_messages(
            system_prompt=system_prompt,
            user_text=json.dumps(user_payload, ensure_ascii=False),
            messages=None,
        )
        result = model.invoke(lc_messages)
        content = getattr(result, "content", "") or "{}"
        try:
            review = json.loads(content)
        except json.JSONDecodeError:
            return self._fallback_review(title, full_text, prompt)
        if not isinstance(review, dict):
            return self._fallback_review(title, full_text, prompt)
        return {
            "score": review.get("score", 5),
            "summaryList": review.get("summaryList", []),
            "detail": review.get("detail", ""),
            "helpList": review.get("helpList", []),
        }
