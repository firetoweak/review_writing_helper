from __future__ import annotations

from typing import Dict


class PolishAgent:
    def full_polish(self, payload: Dict) -> Dict:
        full_text = payload.get("fullText", [])
        polish_prompt = payload.get("polishPrompt") or ""
        polished = []
        for section in full_text:
            children = []
            for child in section.get("children", []):
                text = child.get("text", "")
                polished_text = text.strip()
                if polish_prompt:
                    polished_text = "\n".join(filter(None, [polished_text, polish_prompt]))
                children.append({**child, "text": polished_text})
            polished.append({**section, "children": children})
        return {"task": payload.get("task", "fullPolish"), "newFullText": polished}
