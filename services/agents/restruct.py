from __future__ import annotations

from typing import Dict

from services.agents.outline import OutlineGenerator


class RestructAgent:
    def __init__(self, outline_generator: OutlineGenerator | None = None) -> None:
        self._outline_generator = outline_generator or OutlineGenerator()

    def text_restruct(self, payload: Dict) -> Dict:
        outline_prompt = payload.get("outlinePrompt") or ""
        outline = self._outline_generator.generate_outline(
            {
                "title": "报告重组",
                "idea": "",
                "fullWriteRule": outline_prompt or "重组结构",
                "industry": payload.get("industry") or "",
                "prompt": {"outlinePrompt": outline_prompt},
            }
        )
        restruct_prompt = payload.get("restructPrompt") or ""
        full_text = []
        for section in outline["outline"]:
            chapter_id = section.get("nodeId")
            chapter_title = section.get("title")
            children = []
            for child in section.get("children", []):
                children.append(
                    {
                        "sectionId": child.get("nodeId"),
                        "sectionTitle": child.get("title"),
                        "text": "\n".join(filter(None, [restruct_prompt])),
                    }
                )
            full_text.append(
                {
                    "chapterId": chapter_id,
                    "chapterTitle": chapter_title,
                    "children": children,
                }
            )
        doc_guide = outline.get("docGuide", "")
        if isinstance(doc_guide, list) and doc_guide:
            first = doc_guide[0]
            if isinstance(first, dict):
                doc_guide = first.get("content") or ""
            else:
                doc_guide = str(first)
        return {"docGuide": doc_guide, "outline": outline.get("outline", []), "fullText": full_text}
