from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from models.llm_interface_async import build_chat_model, build_messages, extract_json, is_llm_configured


class OutlineState(TypedDict):
    payload: Dict[str, Any]
    outline: Dict[str, Any]


class OutlineGenerator:
    def __init__(self) -> None:
        self._graph = self._build_graph()

    def generate_outline(self, payload: Dict) -> Dict:
        state = self._graph.invoke({"payload": payload})
        return state["outline"]

    def _coerce_outline_sections(self, full_write_rule: str) -> List[str]:
        lines = [line.strip() for line in (full_write_rule or "").splitlines() if line.strip()]
        if lines:
            return lines
        return []

    def _fallback_outline(self, outline_sections: List[str], outline_prompt: str, title: str) -> List[Dict]:
        outline = []
        if not outline_sections:
            outline_sections = [title or "章节一", "章节二"]
        for idx, section in enumerate(outline_sections, start=1):
            node_id = str(idx)
            children = [
                {
                    "nodeId": f"{node_id}.1",
                    "level": 2,
                    "title": str(section),
                },
                {
                    "nodeId": f"{node_id}.2",
                    "level": 2,
                    "title": str(section),
                },
            ]
            outline.append(
                {
                    "nodeId": node_id,
                    "level": 1,
                    "title": str(section),
                    "children": children,
                }
            )
        return outline

    async def _call_outline_llm(
        self, payload: Dict, outline_prompt: str, outline_sections: List[str]
    ) -> Dict | None:
        title = payload.get("title", "")
        idea = payload.get("idea", "")
        industry = payload.get("industry", "")
        full_write_rule = payload.get("fullWriteRule", "")
        hints = "\n".join(outline_sections)
        prompt_parts = [
            f"立项标题：{title}",
            f"立项构想：{idea}" if idea else "",
            f"行业：{industry}" if industry else "",
            f"写作规则：{full_write_rule}" if full_write_rule else "",
            f"参考章节：{hints}" if hints else "",
            f"用户提示：{outline_prompt}",
        ]
        prompt = "\n".join(part for part in prompt_parts if part)
        system_prompt = (
            "你是立项写作助手，请输出JSON，结构为："
            '{"docGuide": [{"title": "...", "content": "..."}], '
            '"outline": [{"nodeId": "1", "level": 1, "title": "...", '
            '"children": [{"nodeId": "1.1", "level": 2, "title": "..."}]}]}'
        )
        model = build_chat_model(streaming=False)
        result = await model.ainvoke(
            build_messages(system_prompt=system_prompt, user_text=prompt, messages=None)
        )
        content = getattr(result, "content", "")
        data = extract_json(content)
        if isinstance(data, dict) and "docGuide" in data and "outline" in data:
            return data
        return None

    def _build_graph(self):
        graph = StateGraph(OutlineState)

        async def generate(state: OutlineState) -> Dict[str, Any]:
            payload = state.get("payload", {})
            prompt_data = payload.get("prompt") or {}
            outline_prompt = prompt_data.get("outlinePrompt") or ""
            full_write_rule = payload.get("fullWriteRule") or ""
            outline_sections = self._coerce_outline_sections(full_write_rule)
            title = payload.get("title", "未命名立项")
            if outline_prompt and is_llm_configured():
                outline_response = await self._call_outline_llm(payload, outline_prompt, outline_sections)
                if outline_response:
                    return {"outline": outline_response}
            outline = self._fallback_outline(outline_sections, outline_prompt, title)
            doc_guide = [{"title": title, "content": outline_prompt or full_write_rule or title}]
            return {"outline": {"docGuide": doc_guide, "outline": outline}}

        graph.add_node("generate", generate)
        graph.set_entry_point("generate")
        graph.set_finish_point("generate")
        return graph.compile(checkpointer=MemorySaver())
