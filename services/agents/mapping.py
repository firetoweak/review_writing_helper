from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from models.llm_interface_async import build_chat_model, build_messages, extract_json, is_llm_configured

SYSTEM_PROMPT = """
你是“段落候选映射图生成器”，用于缩小后续RAG检索范围。

只输出严格JSON，不要输出任何解释或Markdown。

任务：
基于输入中的 title、idea、outline（每个chapter/section都含keypoint与对应规则），
为每个二级小节（level=2）的 nodeId 生成 relatedSectionIds（相关小节ID列表）：
  sectionId -> [relatedSectionId1, relatedSectionId2, ...]

选择 related 的目标：
这些 related 小节是在未来撰写该 section 时，最可能被“引用/复用/提供前置输入/形成对比约束”的小节，用于缩小检索范围。
不要因为“主题相似”就大量加入，宁缺毋滥。

硬性约束：
1) sectionId 与 relatedSectionIds 都必须来自输入 outline 的二级小节 nodeId（level=2）；禁止编造ID。
2) related 不得包含自身；不得重复。
3) 每个二级小节必须输出（即使 related 为空数组）。
4) 必须包含同一chapter内相邻小节（按children顺序的上一节/下一节，若存在）在 related 中。
5) 每个小节 related 总数 <= MAX_RELATED；其中跨章 related 数 <= MAX_CROSS。
   跨章定义：related 所在的 chapter nodeId 与目标 section 所在 chapter nodeId 不同。

输出格式（严格）：
{
  "version": "neighbors_map.v1",
  "neighbors": {
    "<sectionId>": ["<relatedSectionId>", "..."],
    ...
  }
}
"""

USER_PROMPT = """
请生成“section -> related sections”的映射图，用于缩小后续RAG检索范围。

输入数据如下（JSON）：
{payload_json}

生成参数：
MAX_RELATED = 12
MAX_CROSS = 4

要求：
- 只输出严格JSON
- 必须覆盖所有 level=2 的 nodeId
- 必须包含同章相邻小节（children顺序的前后邻居，若存在）
"""


class MappingAgent:
    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}

    def agenerate_mapping(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(payload.get("sessionId", ""))
        outline = payload.get("outline") or []
        if not is_llm_configured():
            mapping = self.transform_neighbors_to_hierarchy(
                llm_output={"neighbors": {}},
                outline=outline,
                max_related_per_section=12,
                max_cross=4,
                max_related_sections_per_chapter=30,
                chapter_related_cross_only=True,
            )
            self._cache[session_id] = mapping
            return mapping

        model = build_chat_model(streaming=False)
        payload_json = json.dumps(payload, ensure_ascii=False)

        lc_messages = build_messages(
            system_prompt=SYSTEM_PROMPT,
            user_text=USER_PROMPT.format(payload_json=payload_json),
            messages=None,
        )

        result = model.invoke(lc_messages)
        content = (getattr(result, "content", "") or "").strip()
        llm_obj = extract_json(content)
        if not isinstance(llm_obj, dict):
            llm_obj = {"neighbors": {}}

        mapping = self.transform_neighbors_to_hierarchy(
            llm_output=llm_obj,
            outline=outline,
            max_related_per_section=12,
            max_cross=4,
            max_related_sections_per_chapter=30,
            chapter_related_cross_only=True,
        )
        self._cache[session_id] = mapping
        return mapping

    def get_cached(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(session_id)

    def build_outline_maps(self, outline: Any) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
        if not isinstance(outline, list):
            return [], {}, {}

        valid_sections: List[str] = []
        chapter_of: Dict[str, str] = {}
        sections_in_chapter: Dict[str, List[str]] = {}

        for ch in outline:
            if not isinstance(ch, dict):
                continue
            if int(ch.get("level") or 0) != 1:
                continue

            chapter_id = str(ch.get("nodeId") or "").strip()
            children = ch.get("children") or []
            if not chapter_id or not isinstance(children, list):
                continue

            sec_ids: List[str] = []
            for sec in children:
                if not isinstance(sec, dict):
                    continue
                if int(sec.get("level") or 0) != 2:
                    continue
                sid = str(sec.get("nodeId") or "").strip()
                if sid:
                    sec_ids.append(sid)
                    valid_sections.append(sid)
                    chapter_of[sid] = chapter_id

            sections_in_chapter[chapter_id] = sec_ids

        seen = set()
        valid_sections = [x for x in valid_sections if not (x in seen or seen.add(x))]
        return valid_sections, chapter_of, sections_in_chapter

    def _dedupe_keep_order(self, items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def normalize_neighbors(
        self,
        llm_output: Dict[str, Any],
        outline: List[Dict[str, Any]],
        *,
        max_related: int = 12,
        max_cross: int = 4,
    ) -> Dict[str, Any]:
        valid_sections, chapter_of, sections_in_chapter = self.build_outline_maps(outline)
        valid_set = set(valid_sections)

        flat = llm_output.get("neighbors") or {}
        if not isinstance(flat, dict):
            flat = {}

        neighbors_adj: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        for secs in sections_in_chapter.values():
            for i, sid in enumerate(secs):
                prev_ = secs[i - 1] if i - 1 >= 0 else None
                next_ = secs[i + 1] if i + 1 < len(secs) else None
                neighbors_adj[sid] = (prev_, next_)

        norm: Dict[str, List[str]] = {}

        for sid in valid_sections:
            raw = flat.get(sid, [])
            if not isinstance(raw, list):
                raw = []

            cleaned: List[str] = []
            for rid in raw:
                rid = str(rid).strip()
                if not rid or rid == sid:
                    continue
                if rid not in valid_set:
                    continue
                cleaned.append(rid)
            cleaned = self._dedupe_keep_order(cleaned)

            prev_, next_ = neighbors_adj.get(sid, (None, None))
            mandatory = [x for x in (prev_, next_) if x]
            rest = [x for x in cleaned if x not in mandatory]

            cid = chapter_of.get(sid)
            in_chapter = [x for x in rest if chapter_of.get(x) == cid]
            cross = [x for x in rest if chapter_of.get(x) != cid][:max_cross]

            merged = mandatory + in_chapter + cross
            norm[sid] = merged[:max_related]

        return {
            "version": llm_output.get("version", "neighbors_map.v1"),
            "neighbors": norm,
        }

    def transform_neighbors_to_hierarchy(
        self,
        llm_output: Dict[str, Any],
        outline: List[Dict[str, Any]],
        max_related_per_section: int = 12,
        max_cross: int = 4,
        max_related_sections_per_chapter: int = 30,
        chapter_related_cross_only: bool = True,
    ) -> Dict[str, Any]:
        llm_output = self.normalize_neighbors(
            llm_output,
            outline,
            max_related=max_related_per_section,
            max_cross=max_cross,
        )

        valid_sections, chapter_of, sections_in_chapter = self.build_outline_maps(outline)
        valid_set = set(valid_sections)

        flat = llm_output.get("neighbors") or {}
        if not isinstance(flat, dict):
            flat = {}

        chapters: Dict[str, Any] = {}
        for chapter_id, sec_ids in sections_in_chapter.items():
            chapters[chapter_id] = {
                "relatedSections": [],
                "sections": {sid: [] for sid in sec_ids},
            }

        for sid in valid_sections:
            related = flat.get(sid, [])
            if not isinstance(related, list):
                related = []

            cleaned: List[str] = []
            for rid in related:
                rid = str(rid).strip()
                if not rid or rid == sid or rid not in valid_set:
                    continue
                cleaned.append(rid)

            cleaned = self._dedupe_keep_order(cleaned)[:max_related_per_section]

            cid = chapter_of.get(sid)
            if cid in chapters:
                chapters[cid]["sections"][sid] = cleaned

        for cid, ch_obj in chapters.items():
            union_related: List[str] = []
            for rels in ch_obj["sections"].values():
                for rid in rels:
                    if chapter_related_cross_only and chapter_of.get(rid) == cid:
                        continue
                    union_related.append(rid)

            ch_obj["relatedSections"] = self._dedupe_keep_order(union_related)[
                :max_related_sections_per_chapter
            ]

        return {
            "neighbors": {
                "chapters": chapters,
            }
        }
