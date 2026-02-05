from __future__ import annotations

from typing import Any, Dict, List, Optional, Any

from json_repair import repair_json

import asyncio
from models.llm_interface_async import (
    build_chat_model,
    build_messages,
    is_llm_configured,
    is_vlm_configured,
)
from tools.prompt_templating import (
    build_ctx,
    build_image_map,
    render_prompt,
    DEFAULT_PLACEHOLDER_MAP,
    prompt_has_image_tags,
)

placeHolder: Dict[str, str] = {
"macroLogic": "写作宏观逻辑",
"thinkingSteps":"思考步骤指导表",
"writingOrder":"新手推荐写作顺序"
}

def kv_list_to_numbered_text(
    items: List[Dict[str, Any]],
    placeholder: Dict[str, str],
    *,
    indent: str = "    ",
    start: int = 1,
) -> str:
    lines: List[str] = []
    idx = start

    for d in items:
        if not isinstance(d, dict) or not d:
            continue

        for k, v in d.items():
            title = placeholder.get(str(k), str(k))
            content = "" if v is None else str(v)

            lines.append(f"\n{idx}. {title}\n")
            if content.strip():
                # 支持多行内容：每行都缩进
                for ln in content.splitlines():
                    lines.append(f"{indent}{ln}")
            idx += 1

    return "\n".join(lines)


def safe_repair_json_to_dict(
    raw: Any,
    *,
    default: Optional[Dict[str, Any]] = None,
    keep_debug: bool = True,
    max_debug_chars: int = 2000,
) -> Dict[str, Any]:
    """
    把模型输出尽力修复/解析为 dict；失败则返回 default。
    - raw: 可能是 ""/None/非json/半截json/已经是dict/list
    - 对 keypoint/outline 这种强依赖 dict 的场景，统一返回 dict 最稳
    """
    if default is None:
        default = {}

    # 已经是 dict 直接返回
    if isinstance(raw, dict):
        return raw

    # list 也能解析出来，但下游用 .get()，这里给你包一层
    if isinstance(raw, list):
        return {"_list": raw} if keep_debug else default

    if raw is None:
        return default

    s = raw if isinstance(raw, str) else str(raw)
    s = s.strip()
    if not s:
        return default

    try:
        obj = repair_json(s, return_objects=True)
    except Exception as e:
        if keep_debug:
            out = dict(default)
            out["_parse_error"] = f"{type(e).__name__}: {e}"
            out["_raw"] = s[:max_debug_chars]
            return out
        return default

    if isinstance(obj, dict):
        return obj

    # repair_json 解析到 list/标量/字符串等：统一兜底成 dict
    if keep_debug:
        out = dict(default)
        out["_parse_error"] = "not a dict"
        out["_raw"] = s[:max_debug_chars]
        out["_parsed_type"] = type(obj).__name__
        out["_parsed_value"] = obj if isinstance(obj, (str, int, float, bool, type(None))) else str(obj)[:max_debug_chars]
        return out

    return default



def outline_to_text(outline: List[Dict[str, Any]], indent_unit: str = "    ") -> str:
    """把 outline(树) 转成可读的编号文本，便于检索/提示词注入。"""
    lines: List[str] = []

    def walk(nodes: List[Dict[str, Any]]) -> None:
        for n in nodes:
            level = int(n.get("level", 1))
            node_id = str(n.get("nodeId", "")).strip()
            title = str(n.get("title", "")).strip()

            indent = indent_unit * max(level - 1, 0)
            if node_id:
                lines.append(f"{indent}{node_id} {title}")
            else:
                lines.append(f"{indent}{title}")

            children = n.get("children") or []
            if isinstance(children, list) and children:
                walk(children)

    walk(outline)
    return "\n".join(lines)


class OutlineAgent:
    """支持：
    1) 调用知识库检索，把命中的 materials 注入 ctx['materials']
    2) 支持多模态输入：prompt 内含 [IMAGE_n] 时自动走 VLM，并从 KB / payload 汇总 image_map

    """

    def __init__(self, kb) -> None:
        self.kb = kb

    async def agenerate_outline(self, payload: Dict) -> Dict:
        prompt_obj = payload.get("prompt") or {}
        outline_prompt_tpl = prompt_obj.get("outlinePrompt") or ""

        ctx = build_ctx(payload)

        # KB: 用 title/idea/materials/industry 做 query（materials 可能为空）
        query_text = self._build_kb_query(
            ctx,
            extra="\n".join(
                x
                for x in [
                    str(payload.get("title") or "").strip(),
                    str(payload.get("idea") or "").strip(),
                    str(payload.get("materials") or "").strip(),
                ]
                if x
            ),
        )
        await self._ainject_kb_materials(payload, ctx, query_text=query_text, top_k=int(payload.get("kbTopK", 3) or 3))

        image_map = build_image_map(payload)

        prompt = render_prompt(
            outline_prompt_tpl,
            ctx,
            DEFAULT_PLACEHOLDER_MAP,
            keep_unknown=True,
        )

        data = await self._arun_once(prompt=prompt, image_map=image_map)
        data['docGuide'] = kv_list_to_numbered_text(
            items=data.get("docGuide") or [],
            placeholder=placeHolder,
        )
        # print("全文写作要点输出：", data)
        return data if isinstance(data, dict) else (data or {})

    async def chapter_key_point(self, payload: Dict) -> Dict:
        prompt_obj = payload.get("prompt") or {}
        chapter_keypoint_tpl = prompt_obj.get("chapterKeypointPrompt") or ""

        ctx = build_ctx(payload)

        # KB: chapter 级别 query 加上 chapterId/title
        query_text = self._build_kb_query(
            ctx,
            extra="\n".join(
                x
                for x in [
                    f"chapterId={payload.get('chapterId','')}",
                    str(payload.get("chapterTitle") or "").strip(),
                    str(payload.get("title") or "").strip(),
                    str(payload.get("idea") or "").strip(),
                ]
                if x
            ),
        )
        await self._ainject_kb_materials(payload, ctx, query_text=query_text, top_k=int(payload.get("kbTopK", 3) or 3))

        image_map = build_image_map(payload)

        prompt = render_prompt(
            chapter_keypoint_tpl,
            ctx,
            DEFAULT_PLACEHOLDER_MAP,
            keep_unknown=True,
        )

        data = await self._arun_once(prompt=prompt, image_map=image_map)
        # print("chapter keyPoint", data)

        return {
            "chapterId": data.get("chapterId", ""),
            "keyPoint": data.get("keyPoint", ""),
        }

    async def section_key_point(self, payload: Dict) -> Dict:
        prompt_obj = payload.get("prompt") or {}
        section_keypoint_tpl = prompt_obj.get("sectionKeypointPrompt") or ""

        outline = payload.get("outline") or []
        outline_text = outline_to_text(outline) if isinstance(outline, list) else str(outline)

        ctx = build_ctx(payload)
        ctx["outline"] = outline_text

        # KB: section 级别 query 加上 sectionId/title + outline_text
        query_text = self._build_kb_query(
            ctx,
            extra="\n".join(
                x
                for x in [
                    f"chapterId={payload.get('chapterId','')}",
                    f"sectionId={payload.get('sectionId','')}",
                    str(payload.get("sectionTitle") or "").strip(),
                    outline_text,
                ]
                if x
            ),
        )
        await self._ainject_kb_materials(payload, ctx, query_text=query_text, top_k=int(payload.get("kbTopK", 3) or 3))

        image_map = build_image_map(payload)

        prompt = render_prompt(
            section_keypoint_tpl,
            ctx,
            DEFAULT_PLACEHOLDER_MAP,
            keep_unknown=True,
        )

        data = await self._arun_once(prompt=prompt, image_map=image_map)
        # print("section keyPoint", data)
        return {
            "sectionId": data.get("sectionId", ""),
            "keyPoint": data.get("keyPoint", ""),
        }

    # -------------------------
    # KB helpers
    # -------------------------

    @staticmethod
    def _build_kb_query(ctx: Dict[str, str], *, extra: str = "") -> str:
        """拼一个相对稳的 query_text（KBClient 内部会做 token-safe split/truncate）。"""
        parts = []
        for k in ("title", "idea"):
            v = (ctx.get(k) or "").strip()
            if v:
                parts.append(v)
        if extra:
            parts.append(extra.strip())
        # 如果上游已经给了 materials，也一起喂给检索（能提升召回）
        m = (ctx.get("materials") or "").strip()
        if m:
            parts.append(m)
        return "\n\n".join([p for p in parts if p]).strip()

    async def _ainject_kb_materials(self, payload: Dict, ctx: Dict[str, str], *, query_text: str, top_k: int = 3) -> None:
        """从 KB 检索命中片段注入 ctx['materials']，同时把 image_maps 写回 payload 以支持多模态。"""
        if not payload.get("useKB", True):
            return

        project_id = (payload.get("projectId") or "").strip()
        if not project_id or not query_text:
            return

        hits, image_maps = await asyncio.to_thread(
            self.kb.search,
            project_id=project_id,
            query_text=query_text,
            top_k=top_k,
        )

        hits_text = "\n".join(
            h.document.strip()
            for h in (hits or [])
            if getattr(h, "document", None) and str(h.document).strip()
        ).strip()

        if image_maps:
            # 让 build_image_map() 能把 KB 的 docmeta 映射纳入统一映射
            payload["image_maps"] = image_maps

        if hits_text:
            orig = (ctx.get("materials") or "").strip()
            ctx["materials"] = (orig + "\n\n" + hits_text).strip() if orig else hits_text

    # -------------------------
    # Model runner (LLM/VLM)
    # -------------------------

    async def _arun_once(self, *, prompt: str, image_map: Optional[Dict[str, str]] = None) -> Any:
        prompt = (prompt or "").strip()
        image_map = image_map or {}

        has_tags = prompt_has_image_tags(prompt)

        if not (is_llm_configured() or is_vlm_configured()):
            raise ValueError("未配置任何可用模型（chatllm/chatvlm.base_url 都为空），无法生成 outline/keypoints。")

        return await self._allm_call(prompt=prompt, image_map=image_map)

    async def _allm_call(self, *, prompt: str, image_map: Optional[Dict[str, str]] = None) -> Any:
        prompt = (prompt or "").strip()
        image_map = image_map or {}

        has_tags = prompt_has_image_tags(prompt)

        # ✅ 选择模型：有标签走 VLM；无标签优先 LLM（若没配 LLM 则退到 VLM）
        use_multimodal = True
        model = build_chat_model(streaming=False, multimodal=use_multimodal)

        # ⚠️ 关键点：把最终 prompt 放进 user_text，system_prompt 置空
        # 因为 build_messages() 只会在 user_text 中识别 [IMAGE_n] 并切成多模态 blocks
        lc_messages = build_messages(
            system_prompt=None,
            user_text=prompt,
            messages=None,
            image_map=image_map if has_tags else None,
        )
        # print("lc_messages", lc_messages)
        try:
            result = await model.ainvoke(lc_messages)
            raw = getattr(result, "content", "") or ""
        except Exception as e:
            # ✅ 模型调用层兜底：不让异常把整个链路打断（你也可以选择 raise）
            return {"_call_error": f"{type(e).__name__}: {e}", "_raw": ""}

        data = safe_repair_json_to_dict(
            raw,
            default={},      # outline/keypoint 这种返回 dict 最稳
            keep_debug=True, # 线上建议 True，方便定位模型乱输出
        )

        return data
