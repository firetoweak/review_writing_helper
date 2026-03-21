from __future__ import annotations

from typing import Dict


class MergeAgent:
    def merge_texts(self, payload: Dict) -> Dict:
        node_id = payload.get("nodeId", "")
        title = payload.get("title", "")
        texts = payload.get("text", [])
        session_list = payload.get("sessionList", [])
        merge_prompt = payload.get("mergePrompt") or ""
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
        return {
            "nodeId": node_id,
            "title": title,
            "task": payload.get("task", "merge"),
            "texts": merged_texts,
        }
