from __future__ import annotations

import json
from typing import Dict, Any, List, Tuple
import asyncio

from models.llm_interface_async import build_chat_model, build_messages, is_vlm_configured
from tools.prompt_templating import build_image_map, replace_image_tags_with_markdown
from tools.verify import SmartVerifierCore, my_check_batch


PROMPT = """你是文档润色助手。请根据用户输入的 JSON 中的 text 字段，对其进行润色：
- 修正错别字、语病、标点与重复表达
- 让段落衔接更自然（只做轻微调整）
- 不新增任何事实/数据/结论，不扩写内容
- 保留所有图片占位符标签（如 [IMAGE_1]）原样不改
输出要求：只输出润色后的正文文本，不要输出任何解释、前后缀、Markdown、JSON。
"""

class PolishAgent:
    """
    最小可用“全文润色”：
    - 遍历 fullText[chapter].children[section]，逐段润色 text
    - 不改结构，不改 image_url
    - LLM 不可用：原样返回
    """

    async def full_polish(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        polish_prompt = PROMPT
        full_text = payload.get("fullText", []) or []
        image_maps = build_image_map(payload)
        print("图片映射:", image_maps)

        # 兜底：fullText 不是 list 直接原样返回
        if not isinstance(full_text, list):
            return {"task": "full_polish", "newFullText": full_text}


        new_full = await self._llm_polish_full_text(full_text, polish_prompt, image_maps)

        return {
            "task": "full_polish",
            "newFullText": new_full,
        }

    # -------------------------
    # fallback：直接原样返回
    # -------------------------
    async def _fallback_polish_full_text(self, full_text: Any) -> Any:
        return full_text

    async def _llm_polish_full_text(
        self,
        full_text: List[Dict[str, Any]],
        prompt: str,
        image_maps: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        # 并发上限：你可以按需调大/调小（建议先 6~12）
        MAX_CONCURRENCY = 12
        sem = asyncio.Semaphore(MAX_CONCURRENCY)

        # 共享 model（通常 OK，更快）；如果你怀疑并发不安全，把 shared_model 改成 None
        shared_model = build_chat_model(streaming=False)

        out: List[Dict[str, Any]] = []
        jobs: List[Tuple[int, int, str, str, str, str, str]] = []
        # jobs item: (chap_idx, sec_idx, chap_id, chap_title, sec_id, sec_title, text)

        # 先把结构复制出来（保留顺序），同时收集需要润色的 section
        for chap_idx, chap in enumerate(full_text):
            if not isinstance(chap, dict):
                out.append(chap)
                continue

            chap_id = chap.get("chapterId", "") or ""
            chap_title = chap.get("chapterTitle", "") or ""
            children = chap.get("children", []) or []
            if not isinstance(children, list):
                children = []

            new_children: List[Any] = []
            for sec_idx, sec in enumerate(children):
                if not isinstance(sec, dict):
                    new_children.append(sec)
                    continue

                new_children.append(dict(sec))  # 先拷贝占位，后面回填 text

                sec_id = sec.get("sectionId", "") or ""
                sec_title = sec.get("sectionTitle", "") or ""
                text = sec.get("text", "") or ""

                if str(text).strip():
                    jobs.append((chap_idx, sec_idx, chap_id, chap_title, sec_id, sec_title, str(text)))

            new_chap = dict(chap)
            new_chap["children"] = new_children
            out.append(new_chap)

        if not jobs:
            return out

        async def _run_one(job: Tuple[int, int, str, str, str, str, str]) -> Tuple[int, int, str]:
            chap_idx, sec_idx, chap_id, chap_title, sec_id, sec_title, text = job
            async with sem:
                try:
                    polished_text = await self._llm_polish_section(
                        model=shared_model,
                        prompt=prompt,
                        chapterId=str(chap_id),
                        chapterTitle=str(chap_title),
                        sectionId=str(sec_id),
                        sectionTitle=str(sec_title),
                        text=text,
                    )
                    polished_text_md = replace_image_tags_with_markdown(polished_text, image_maps)
                    return chap_idx, sec_idx, polished_text_md
                except Exception:
                    # 单段失败：回退原文，避免整批失败
                    return chap_idx, sec_idx, replace_image_tags_with_markdown(text, image_maps)

        results = await asyncio.gather(*[asyncio.create_task(_run_one(j)) for j in jobs])

        # 回填
        for chap_idx, sec_idx, polished in results:
            chap = out[chap_idx]
            children = chap.get("children") or []
            if 0 <= sec_idx < len(children) and isinstance(children[sec_idx], dict):
                children[sec_idx]["text"] = polished

        return out


    async def _llm_polish_section(
        self,
        *,
        model: Any,
        prompt: str,
        chapterId: str,
        chapterTitle: str,
        sectionId: str,
        sectionTitle: str,
        text: str,
    ) -> str:
        """
        让模型只输出润色后的 text（纯文本）。
        """
        user_payload = {
            "chapterId": chapterId,
            "chapterTitle": chapterTitle,
            "sectionId": sectionId,
            "sectionTitle": sectionTitle,
            "text": text,
            "requirements": [
                "仅做错别字/语病/标点修正与轻微衔接优化",
                "不新增事实信息，不扩写内容",
                "保留所有图片标签（如 [IMAGE_1]）原样不改",
                "只输出润色后的正文文本，不要输出解释",
            ],
        }

        lc_messages = build_messages(
            system_prompt=prompt,
            user_text=json.dumps(user_payload, ensure_ascii=False),
            messages=None,
        )
        result = await model.ainvoke(lc_messages)
        content = getattr(result, "content", "") or ""
        return str(content).strip()