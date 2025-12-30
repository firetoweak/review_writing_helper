from __future__ import annotations

from typing import Any, Dict, List, TypedDict

import httpx
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from models.llm_interface_async import LLMInterfaceAsync, extract_json


class OutlineState(TypedDict):
    payload: Dict[str, Any]
    outline: Dict[str, Any]


class OutlineGenerator:
    def __init__(self, llm_client: LLMInterfaceAsync | None = None) -> None:
        self._llm_client = llm_client or LLMInterfaceAsync()
        self._graph = self._build_graph()

    def generate_outline(self, payload: Dict) -> Dict:
        state = self._graph.invoke({"payload": payload})
        return state["outline"]

    def _coerce_outline_sections(self, payload: Dict, project: Dict) -> List[str]:
        sections = (
            payload.get("outlineSections")
            or payload.get("outline_sections")
            or project.get("outlineSections")
            or project.get("outline_sections")
        )
        if isinstance(sections, list) and sections:
            return [str(item) for item in sections]
        return []

    def _fallback_outline(self, outline_sections: List[str], outline_prompt: str, title: str) -> List[Dict]:
        outline = []
        for idx, section in enumerate(outline_sections, start=1):
            node_id = str(idx)
            children = [
                {
                    "nodeId": f"{node_id}.1",
                    "level": 2,
                    "title": str(section),
                    "keyPoint": outline_prompt or "",
                },
                {
                    "nodeId": f"{node_id}.2",
                    "level": 2,
                    "title": str(section),
                    "keyPoint": outline_prompt or "",
                },
            ]
            outline.append(
                {
                    "nodeId": node_id,
                    "level": 1,
                    "title": str(section),
                    "keyPoint": outline_prompt or "",
                    "children": children,
                }
            )
        return outline

    async def _call_outline_llm(
        self, project: Dict, outline_prompt: str, outline_sections: List[str]
    ) -> Dict | None:
        title = project.get("title", "")
        idea = project.get("idea", "")
        attachments = project.get("attachments", [])
        attachment_names = ", ".join(att.get("name", "") for att in attachments if att.get("name"))
        hints = "\n".join(outline_sections)
        prompt_parts = [
            f"立项标题：{title}",
            f"立项构想：{idea}" if idea else "",
            f"附件：{attachment_names}" if attachment_names else "",
            f"参考章节：{hints}" if hints else "",
            f"用户提示：{outline_prompt}",
        ]
        prompt = "\n".join(part for part in prompt_parts if part)
        system_prompt = (
            "你是立项写作助手，请输出JSON，结构为："
            '{"docGuide": "...", "outline": [{"nodeId": "1", "level": 1, "title": "...", '
            '"keyPoint": "...", "children": [{"nodeId": "1.1", "level": 2, "title": "...", "keyPoint": "..."}]}]}'
        )
        try:
            raw = await self._llm_client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
            )
        except httpx.HTTPError:
            return None
        data = extract_json(raw)
        if isinstance(data, dict) and "docGuide" in data and "outline" in data:
            return data
        return None

    def _build_graph(self):
        graph = StateGraph(OutlineState)

        async def generate(state: OutlineState) -> Dict[str, Any]:
            payload = state.get("payload", {})
            project = payload.get("project", {})
            outline_prompt = payload.get("outlinePrompt") or payload.get("outline_prompt", "")
            outline_sections = self._coerce_outline_sections(payload, project)
            title = project.get("title", "未命名立项")
            if outline_prompt and self._llm_client.is_configured():
                outline_response = await self._call_outline_llm(project, outline_prompt, outline_sections)
                if outline_response:
                    return {"outline": outline_response}
            outline = self._fallback_outline(outline_sections, outline_prompt, title)
            return {"outline": {"docGuide": outline_prompt or f"{title}", "outline": outline}}

        graph.add_node("generate", generate)
        graph.set_entry_point("generate")
        graph.set_finish_point("generate")
        return graph.compile(checkpointer=MemorySaver())
