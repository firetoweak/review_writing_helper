from __future__ import annotations

import json
from typing import Dict, List

from models.llm_interface_async import (
    build_chat_model,
    build_messages,
    extract_image_urls_from_text_list,
    is_llm_configured,
)


class ReviewAgent:
    def review_section(self, payload: Dict) -> Dict:
        title = payload.get("title", "")
        section_title = payload.get("sectionTitle", "")
        text_items = payload.get("textList", [])
        industry = payload.get("industry") or ""
        review_rule = payload.get("sectionReviewRule") or ""
        prompt = payload.get("prompt") or {}
        review_prompt = prompt.get("sectionReviewPrompt") or ""
        full_text = "\n".join(item.get("text", "") for item in text_items)
        if is_llm_configured():
            review = self._llm_review(
                title,
                section_title,
                full_text,
                review_prompt,
                review_rule,
                industry,
                extract_image_urls_from_text_list(text_items),
            )
        else:
            review = self._fallback_review(title, section_title, full_text, review_prompt, review_rule)
        return {"review": review}

    def full_review(self, payload: Dict) -> Dict:
        full_text = payload.get("fullReviewText", "")
        prompt = payload.get("prompt") or {}
        full_review_prompt = prompt.get("fullReviewPrompt") or ""
        report = full_review_prompt or f"已接收全文评审请求，内容长度为 {len(full_text)}。"
        return {"fullReviewAns": report}

    def _summary_from_prompt(self, prompt: str) -> str:
        if not prompt:
            return ""
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        return "；".join(lines[:3]) if lines else prompt.strip()

    def _fallback_review(self, title: str, section_title: str, full_text: str, prompt: str, rule: str) -> Dict:
        score = min(10, max(1, len(full_text) // 120 + 3))
        evaluate = prompt or f"已接收《{title}》-{section_title} 评审请求。"
        suggestion = rule or "请根据写作规则补充关键论据。"
        to_do_list = [item for item in [self._summary_from_prompt(prompt), suggestion] if item]
        return {
            "score": score,
            "evaluate": evaluate,
            "suggestion": suggestion,
            "to_do_list": to_do_list,
        }

    def _llm_review(
        self,
        title: str,
        section_title: str,
        full_text: str,
        prompt: str,
        rule: str,
        industry: str,
        image_urls: List[str],
    ) -> Dict:
        model = build_chat_model(streaming=False)
        system_prompt = "你是资深评审助手，输出严格 JSON。"
        user_payload = {
            "title": title,
            "sectionTitle": section_title,
            "industry": industry,
            "text": full_text,
            "rule": rule,
            "prompt": prompt,
            "output_format": {
                "score": "1-10",
                "evaluate": "string",
                "suggestion": "string",
                "to_do_list": ["string"],
            },
        }
        lc_messages = build_messages(
            system_prompt=system_prompt,
            user_text=json.dumps(user_payload, ensure_ascii=False),
            messages=None,
            image_urls=image_urls,
        )
        result = model.invoke(lc_messages)
        content = getattr(result, "content", "") or "{}"
        try:
            review = json.loads(content)
        except json.JSONDecodeError:
            return self._fallback_review(title, section_title, full_text, prompt, rule)
        if not isinstance(review, dict):
            return self._fallback_review(title, section_title, full_text, prompt, rule)
        return {
            "score": review.get("score", 5),
            "evaluate": review.get("evaluate", ""),
            "suggestion": review.get("suggestion", ""),
            "to_do_list": review.get("to_do_list", []),
        }
