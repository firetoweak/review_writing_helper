from __future__ import annotations

from typing import Dict

from services.agents.outline import OutlineGenerator


class RestructAgent:
    def __init__(self, outline_generator: OutlineGenerator | None = None) -> None:
        self._outline_generator = outline_generator or OutlineGenerator()

    def text_restruct(self, payload: Dict) -> Dict:
        outline = self._outline_generator.generate_outline(
            {
                "project": {"title": "报告重组", "idea": ""},
                "outlinePrompt": payload.get("outlinePrompt") or "",
                "outlineSections": payload.get("outlineSections") or payload.get("outline_sections"),
            }
        )
        restruct_prompt = payload.get("restructPrompt") or ""
        full_text = []
        for section in outline["outline"]:
            full_text.append(
                {
                    "nodeId": section["nodeId"],
                    "title": section["title"],
                    "level": 1,
                    "children": [
                        {
                            "nodeId": child["nodeId"],
                            "level": 2,
                            "title": child["title"],
                            "text": "\n".join(
                                filter(None, [child.get("keyPoint", ""), restruct_prompt])
                            ),
                        }
                        for child in section["children"]
                    ],
                }
            )
        return {**outline, "fullText": full_text}
