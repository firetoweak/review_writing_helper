from __future__ import annotations

from typing import Dict, List


class ReviewAgent:
    def review_section(self, payload: Dict) -> Dict:
        node_id = payload.get("nodeId", "")
        title = payload.get("title", "")
        text_items = payload.get("text", [])
        review_prompt = payload.get("reviewpPrompt") or ""
        full_text = "\n".join(item.get("text", "") for item in text_items)
        score = min(10, max(1, len(full_text) // 120 + 3))
        summary = self._summary_from_prompt(review_prompt)
        detail = review_prompt or f"已接收《{title}》评审请求。"
        help_list = self._summary_from_prompt(review_prompt)
        return {
            "task": payload.get("task", "sectionReview"),
            "title": title,
            "nodeId": node_id,
            "review": {
                "score": score,
                "summaryList": summary,
                "detail": detail,
                "helpList": help_list,
            },
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
