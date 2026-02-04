from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import diff_match_patch as dmp_module
from json_repair import repair_json

from models.llm_interface_async import build_messages


# ----------------------------
# 批量审计函数签名（你已有 my_check_batch）
# ----------------------------
# items: [{"id": int, "text": str, "context": str}, ...]
# evidence: str
# model: 外层初始化好的 chat model（支持 await model.ainvoke(messages)）
# return: {id: "VALID"/"HALLUCINATION"/"UNKNOWN"}
CheckBatchAsyncFn = Callable[[List[Dict[str, Any]], str, Any], Awaitable[Dict[int, str]]]


def _chunked(lst: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# ----------------------------
# 核心：diff + smart_slice + 异步审计 + HTML 拼装（无 CSS）
# ----------------------------
class SmartVerifierCore:
    """
    核心流程（异步 + 切片级并发 + 批量审计）：
    1) diff original vs merged
    2) 只抽取新增(op==1) -> smart_slice -> ai_queue
    3) 切片级并发 submit（max_concurrency），内部微批聚合调用 my_check_batch
    4) 输出 HTML（无 CSS；仅 <del> 和 <span title=...>）
    """

    def __init__(self) -> None:
        self.dmp = dmp_module.diff_match_patch()

    def smart_slice(self, text: str) -> List[str]:
        # 原逻辑：极短文本保护 + 智能切分
        if len(text) < 10:
            return [text]

        pattern = (
            r"([。！？；.!?;:\n])|"
            r"(?<=，)(?=(?:"
            r"且|并且|而且|同时|以及|还|并|加上|"
            r"但|但是|然而|不过|可是|却|只是|"
            r"其中|特别是|尤其是|例如|比如|即|"
            r"届时|随后|接着|之后|最终|"
            r"因此|所以|致使|导致|从而|"
            r"此外|另外|反之|相反"
            r"))"
        )
        parts = re.split(pattern, text)

        chunks: List[str] = []
        current_chunk = ""
        for part in parts:
            if not part:
                continue
            if re.match(r"^[。！？；.!?;:\n]$", part):
                if chunks:
                    chunks[-1] += part
                else:
                    current_chunk += part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                clean_part = part.strip()
                if clean_part:
                    chunks.append(clean_part)

        if current_chunk:
            chunks.append(current_chunk)

        # 兜底二次切分
        final_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) > 80 and "，" in chunk:
                sub_parts = chunk.split("，")
                sub_parts = [s + "，" for s in sub_parts[:-1]] + [sub_parts[-1]]
                final_chunks.extend([s for s in sub_parts if len(s.strip()) > 1])
            else:
                final_chunks.append(chunk)

        return final_chunks

    def build_plan(self, original: str, merged: str) -> Dict[str, Any]:
        diffs = self.dmp.diff_main(original or "", merged or "")
        self.dmp.diff_cleanupSemantic(diffs)

        final_segments: List[Dict[str, Any]] = []
        ai_queue: List[Dict[str, Any]] = []
        task_id_counter = 0

        for op, text in diffs:
            if op == 0:
                final_segments.append({"type": "equal", "text": text})
            elif op == -1:
                final_segments.append({"type": "deleted", "text": text})
            elif op == 1:
                slices = self.smart_slice(text)
                for s in slices:
                    t_id = task_id_counter
                    task_id_counter += 1
                    # 保持你原版逻辑：context 就是 ...s...
                    ai_queue.append({"id": t_id, "text": s, "context": f"...{s}..."})
                    final_segments.append({"type": "pending", "id": t_id, "text": s})

        return {"final_segments": final_segments, "ai_queue": ai_queue}

    async def verify_async(
        self,
        *,
        original: str,
        merged: str,
        evidence: str,
        model: Any,
        check_batch_fn,
        batch_size: int = 10,
        max_inflight_batches: int = 4,
        timeout_s: Optional[float] = None,
        default_label: str = "UNKNOWN",
    ) -> str:
        plan = self.build_plan(original, merged)

        final_segments = plan["final_segments"]
        ai_queue = plan["ai_queue"]

        # print(f"[VFY] build_plan done: final_segments={len(final_segments)} ai_queue={len(ai_queue)}", flush=True)
        # print(f"[VFY] check_batch_fn={getattr(check_batch_fn, '__name__', str(check_batch_fn))}", flush=True)

        ai_results_map: Dict[int, str] = {}
        ai_results_quote_map: Dict[int, str] = {}
        sem = asyncio.Semaphore(max_inflight_batches)

        async def _run_batch(batch: List[Dict[str, Any]]) -> None:
            async with sem:
                try:
                    print(f"[VFY] calling check_batch_fn batch_size={len(batch)}", flush=True)
                    coro = check_batch_fn(batch, evidence, model)
                    res_map = await asyncio.wait_for(coro, timeout=timeout_s) if timeout_s else await coro
                except Exception as e:
                    print(f"[VFY] batch failed: {e!r}", flush=True)
                    res_map = {}

                for it in batch:
                    rid = it.get("id", None)
                    if rid is None:
                        rid = it.get("match_id", None)
                    if rid is None:
                        # 这个 item 没有可用 id，直接跳过或标个日志
                        print(f"[VFY] skip item without id keys: keys={list(it.keys())}", flush=True)
                        continue

                    r = res_map.get(rid)

                    # 兼容旧返回：{id: "VALID"} ；新返回：{id: {"label": "...", "quote": "..."}}
                    if isinstance(r, dict):
                        lab = str(r.get("label", default_label)).strip().upper()
                        quote = str(r.get("quote", "") or "").strip()
                    else:
                        lab = str(r or default_label).strip().upper()
                        quote = ""

                    # 绿必须可解释：没有 quote 的 VALID 一律降级 UNKNOWN
                    if lab == "VALID" and not quote:
                        lab = "UNKNOWN"

                    ai_results_map[rid] = lab
                    if quote:
                        ai_results_quote_map[rid] = quote


        batches = _chunked(ai_queue, batch_size)
        print(f"[VFY] total batches={len(batches)}", flush=True)
        if batches:
            tasks = [asyncio.create_task(_run_batch(b)) for b in batches]
            await asyncio.gather(*tasks)
        
         # ---- HTML 拼装 ----
        html_output = ""
        style_del = "text-decoration: line-through; color: #999; background-color: #f0f0f0;"
        style_valid = "color: #15803d; background-color: #dcfce7; border-bottom: 2px solid #86efac;"
        style_hallucination = "color: #b91c1c; background-color: #fee2e2; text-decoration: underline wavy #ef4444;"
        style_unknown = "color: #92400e; background-color: #fef3c7; border-bottom: 2px dashed #f59e0b;"

        for seg in final_segments:
            seg_type = seg["type"]
            text = seg["text"]

            if seg_type == "equal":
                html_output += text
                continue

            if seg_type == "deleted":
                html_output += f'<span style="{style_del}">{text}</span>'
                continue

            if seg_type != "pending":
                # 理论上不会出现，但做个兜底，避免新类型炸
                html_output += text
                continue

            # ---- 只有 pending 才需要 id 映射审计结果 ----
            rid = seg.get("id")
            if rid is None:
                # 兜底：没有 id 就当 UNKNOWN
                title = "证据不足：建议补充数据/上传报告/改成弱断言"
                html_output += f'<span style="{style_unknown}" title="{title}">{text}</span>'
                continue

            label = (ai_results_map.get(rid) or default_label).strip().upper()
            quote = (ai_results_quote_map.get(rid) or "").strip()

            if label == "HALLUCINATION":
                title = "证据未覆盖：建议删改/补充来源"
                html_output += f'<span style="{style_hallucination}" title="{title}">{text}</span>'
            elif label == "VALID":
                title = f"证据命中：{quote}" if quote else "证据命中"
                html_output += f'<span style="{style_valid}" title="{title}">{text}</span>'
            else:
                title = "证据不足：建议补充数据/上传报告/改成弱断言"
                html_output += f'<span style="{style_unknown}" title="{title}">{text}</span>'



        return html_output
    
    async def verify_diff_only_async(
        self,
        *,
        original: str,
        merged: str,
    ) -> str:
        plan = self.build_plan(original, merged)
        final_segments = plan["final_segments"]

        html_output = ""
        for seg in final_segments:
            t = seg["type"]
            text = seg["text"]
            if t == "equal":
                html_output += text
            elif t == "deleted":
                html_output += f"<del>{text}</del>"
            elif t == "pending":
                # 直接认为 pending 全是“新增”
                html_output += f'<span title="新增">{text}</span>'

        return html_output
    
def _md_escape(s: str) -> str:
    # 轻量转义，避免反引号/方括号等影响渲染（按需扩展）
    s = s.replace("`", "\\`")
    return s

async def my_check_batch(items: List[Dict[str, Any]], evidence: str, model) -> Dict[int, Dict[str, str]]:
    """
    调用你的模型，把 evidence + items 送进去，拿回 JSON 结果。

    期望模型输出：
       [{"id":0,"label":"VALID"}, {"id":1,"label":"HALLUCINATION"}]
    """
    payload = json.dumps(
        [{"id": i["id"], "text": i["text"], "context": i.get("context", "")} for i in items],
        ensure_ascii=False,
    )

    system_prompt = """
你是一个内容合规审计员。你的唯一任务是判断【待检列表】中的信息是否严格忠实于【事实依据】。
判定标准（严格执行）：
1) [VALID]：待检内容能在事实依据中找到明确对应表述（必须返回 quote）。
2) [HALLUCINATION]：事实依据中找不到对应表述，或与事实依据冲突。
3) [UNKNOWN]：无法判断（证据可能不全/表述太泛/需要更多上下文）。

输出要求：仅返回 JSON 数组，每个元素包含字段：
- id: 待检项ID（数字）
- label: VALID / HALLUCINATION / UNKNOWN（三选一）
- quote: 仅当 label=VALID 时必填；必须从事实依据中摘取的“对应短句”，长度<=30字；否则置空字符串

不允许输出任何解释、额外文本或 Markdown 包裹。
""".strip()

    user_prompt = f"""
【事实依据】:
{evidence}

【待检列表】:
{payload}
""".strip()

    messages = build_messages(system_prompt=system_prompt, user_text=user_prompt)
    result = await model.ainvoke(messages)
    content = getattr(result, "content", "") or "[]"
            
    try:
        data = repair_json(content, return_objects=True)
        print("标记：", data)
    except Exception:
        out = {}
        for it in items:
            rid = it.get("id", it.get("match_id"))
            if rid is not None:
                out[int(rid)] = {"label": "UNKNOWN", "quote": ""}
        return out

    if isinstance(data, dict) and "result" in data:
        data = data["result"]

    out: Dict[int, Dict[str, str]] = {}
    if isinstance(data, list):
        for row in data:
            try:
                _id = int(row.get("id"))

                label = str(row.get("label", "")).upper().strip()
                quote = str(row.get("quote", "") or "").strip()
                if label not in ("VALID", "HALLUCINATION", "UNKNOWN"):
                    label = "UNKNOWN"
                if label != "VALID":
                    quote = ""
                else:
                    # 保险：quote 过长截断
                    if len(quote) > 30:
                        quote = quote[:30]
                out[_id] = {"label": label, "quote": quote}
            except Exception:
                continue

    for it in items:
        out.setdefault(it["id"], {"label": "UNKNOWN", "quote": ""})

    return out

