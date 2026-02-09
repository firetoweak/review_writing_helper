from __future__ import annotations

import json
from typing import Any, Dict, Optional, List, Tuple

from json_repair import repair_json
from models.llm_interface_async import build_chat_model, build_messages, is_llm_configured, is_vlm_configured
from tools.prompt_templating import build_ctx, build_image_map, render_prompt, DEFAULT_PLACEHOLDER_MAP, prompt_has_image_tags, replace_image_tags_with_markdown
from tools.verify import SmartVerifierCore, my_check_batch
import asyncio



class MergeAgent:

    def __init__(self, kb) -> None:
        self._kb_client = kb
        
    async def amerge_texts(self, payload: Dict) -> Dict:
        prompt_obj = payload.get("prompt") or {}
        merge_prompt = prompt_obj.get("mergePrompt") or ""
        # merge_correct_prompt = prompt_obj.get("mergeCorrectPrompt") or ""

        ctx = build_ctx(payload)

        # 知识库调用（优先走引导词检索）
        text = ctx['textList']
        projectId = payload.get("projectId") or ""
        section_id = str(payload.get("sectionId") or ctx.get("sectionId") or "").strip()
        section_title = str(payload.get("sectionTitle") or ctx.get("sectionTitle") or "").strip()

        if hasattr(self._kb_client, "search_section") and projectId and section_id:
            kb_snapshot = await asyncio.to_thread(
                self._kb_client.search_section,
                project_id=projectId,
                section_id=section_id,
                section_title=section_title,
                context_fingerprint=None,
                snapshot=None,
                reuse_snapshot=False,
                k_each=3,
                k_total=12,
            )
            hit_dicts = kb_snapshot.get("hits") or []
            hits_text = "\n".join(
                str(h.get("document") or "").strip()
                for h in hit_dicts
                if isinstance(h, dict) and str(h.get("document") or "").strip()
            ).strip()
            kb_image_maps = kb_snapshot.get("image_maps") or {}
        else:
            kb_hits, kb_image_maps = await asyncio.to_thread(
                self._kb_client.search,
                query_text=text,
                project_id=projectId,
                top_k=3,
            )
            hits_text = "\n".join(
                h.document.strip()
                for h in (kb_hits or [])
                if getattr(h, "document", None) and h.document.strip()
            ).strip()

        if kb_image_maps:
            payload["image_maps"] = kb_image_maps
        ctx['materials'] = hits_text
        
        image_maps = build_image_map(payload)

        # 1) merge
        prompt = render_prompt(
            merge_prompt,
            ctx,
            DEFAULT_PLACEHOLDER_MAP,
            keep_unknown=True,
        )
        
        merged_texts = await self._run_merge_once(
            prompt=prompt,
            image_map=image_maps,
            fallback_texts=payload.get("text", []),
            fallback_session_list=payload.get("sessionList", []) or [],
        )

        # 校验
        correct_res = {"textList": []}

        evidence_text = f"交互资料：{ctx.get('qa', '')}\n\n相关素材：{ctx.get('materials', '')}\n\n前文：{ctx.get('historyText', '')}"
        print("======================一键合入===============================")
        print("校验资料：", evidence_text[:20])
        print("============================================================")

        for item in merged_texts.get("textList", []):
            correct_model = build_chat_model(streaming=False)
            core = SmartVerifierCore()
            correct_text = await  core.verify_async(
                original=ctx.get("textList", ""),
                merged=item.get("text", ""),
                evidence=evidence_text,
                model=correct_model,
                check_batch_fn=my_check_batch,
                batch_size=16,
                max_inflight_batches=4
            )
            correct_text = replace_image_tags_with_markdown(correct_text, image_maps)
            item['text'] = correct_text
            correct_res['textList'].append(item)
            
        return correct_res
    async def correct_textlist_concurrent_shared(
        merged_texts: Dict[str, Any],
        *,
        ctx: Dict[str, Any],
        evidence_text: str,
        image_maps: Dict[str, Any],
        concurrency: int = 8,
    ) -> Dict[str, Any]:
        text_list: List[Dict[str, Any]] = list(merged_texts.get("textList", []) or [])
        sem = asyncio.Semaphore(concurrency)

        # 共享资源
        correct_model = build_chat_model(streaming=False)
        core = SmartVerifierCore()

        async def _one(idx: int, item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
            async with sem:
                correct_text = await core.verify_async(
                    original=ctx.get("textList", ""),
                    merged=json.dumps(item.get("text", ""), ensure_ascii=False, indent=2),
                    evidence=evidence_text,
                    model=correct_model,
                    check_batch_fn=my_check_batch,
                    batch_size=10,
                    max_inflight_batches=4,
                )
                correct_text = replace_image_tags_with_markdown(correct_text, image_maps)
                new_item = dict(item)
                new_item["text"] = correct_text
                return idx, new_item

        results = await asyncio.gather(*[asyncio.create_task(_one(i, it)) for i, it in enumerate(text_list)])
        results.sort(key=lambda x: x[0])
        return {"textList": [it for _, it in results]}

    async def _run_merge_once(
        self,
        *,
        prompt: str,
        image_map: Dict[str, str],
        fallback_texts: Any,
        fallback_session_list: Any,
        model_name: Optional[str] = None,
    ) -> list:
        has_tags = prompt_has_image_tags(prompt)

        # 只要任一模型可用，就走模型；都不可用才 fallback
        if is_llm_configured() or is_vlm_configured():
            return await self._llm_merge(prompt, image_map=image_map, model_name=model_name)

        return self._fallback_merge(fallback_texts, fallback_session_list, prompt)

    async def _llm_merge(
        self,
        prompt: str,
        *,
        image_map: Optional[Dict[str, str]] = None,
        user_input: str = "",
        model_name: Optional[str] = None,
    ) -> list:
        prompt = (prompt or "").strip()
        image_map = image_map or {}

        has_tags = prompt_has_image_tags(prompt)

        # ✅ 选择模型：有标签走 VLM；无标签优先 LLM（若没配 LLM 则退到 VLM）
        use_multimodal = has_tags or (not is_llm_configured() and is_vlm_configured())
        model = build_chat_model(streaming=False, multimodal=use_multimodal, model_name=model_name)

        # ⚠️ 关键点：把最终 prompt 放进 user_text，system_prompt 置空
        # 因为 build_messages() 只会在 user_text 中识别 [IMAGE_n] 并切成多模态 blocks
        lc_messages = build_messages(
            system_prompt=None,
            user_text="\n\n".join([x for x in [prompt, user_input] if x]),
            messages=None,
            image_map=image_map if has_tags else None,
        )

        result = await model.ainvoke(lc_messages)
        content = getattr(result, "content", "") or "[]"
        data = repair_json(content, return_objects=True)
        # data = content
        print(data)
        return data 

    def _fallback_merge(self, texts: list, session_list: list, merge_prompt: str) -> list:
        merged_texts = []
        suggestions = []
        for session in session_list or []:
            for msg in (session.get("messages") or []):
                if msg and msg.get("role") == "assistant":
                    suggestions.append(msg.get("content", ""))
        for item in texts or []:
            merged_item = dict(item)
            text = merged_item.get("text", "")
            # ✅ 仅取前两条建议，避免太长
            merged_item["text"] = (
                f"{text}\n\n参考建议：\n" + "\n".join(suggestions[:2]) + f"\n\n参考提示：\n{merge_prompt}"
            )
            merged_texts.append(merged_item)
        print("merged_texts:", merged_texts)
        return merged_texts
