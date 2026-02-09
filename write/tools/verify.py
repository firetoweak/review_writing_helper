from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Set

try:
    import diff_match_patch as dmp_module
except Exception:  # pragma: no cover
    # Fallback for environments where the module path differs.
    # You still need the 'diff-match-patch' implementation installed.
    from diff_match_patch import diff_match_patch as _dmp  # type: ignore

    class _DMPModule:
        diff_match_patch = _dmp

    dmp_module = _DMPModule()
from json_repair import repair_json

import markdown
from bs4 import BeautifulSoup, NavigableString

from models.llm_interface_async import build_messages

_CONNECTORS = (
    "且|并且|而且|同时|以及|还|并|加上|"
    "但|但是|然而|不过|可是|却|只是|"
    "其中|特别是|尤其是|例如|比如|即|"
    "届时|随后|接着|之后|最终|"
    "因此|所以|致使|导致|从而|"
    "此外|另外|反之|相反"
)

_SENT_END = set("。！？；.!?;\n")  # 故意移除 ':'

# ---- HTML annotation helpers (stable: Markdown -> HTML -> wrap text nodes) ----

_VFY_CSS = """<style>
.vfy-del { text-decoration: line-through; color: #999; background-color: #f0f0f0; }
.vfy-valid { color: #15803d; background-color: #dcfce7; border-bottom: 2px solid #86efac; }
.vfy-hal { color: #b91c1c; background-color: #fee2e2; text-decoration: underline wavy #ef4444; }
.vfy-unk { color: #92400e; background-color: #fef3c7; border-bottom: 2px dashed #f59e0b; }
.vfy-added { color: #15803d; background-color: #dcfce7; border-bottom: 2px solid #86efac; }
.vfy-block-del { margin-top: 12px; padding: 8px; border: 1px dashed #999; background: #fafafa; }
.vfy-block-del h4 { margin: 0 0 6px 0; font-size: 14px; }
.vfy-block-del pre { margin: 0; white-space: pre-wrap; }
</style>"""


def _collect_text_nodes(soup: BeautifulSoup) -> List[NavigableString]:
    nodes: List[NavigableString] = []
    for t in soup.find_all(string=True):
        if not isinstance(t, NavigableString):
            continue
        parent = getattr(t, "parent", None)
        if parent is None:
            continue
        if parent.name in ("script", "style"):
            continue
        nodes.append(t)
    return nodes


def _wrap_range_in_dom(
    soup: BeautifulSoup,
    nodes: List[NavigableString],
    start: int,
    end: int,
    *,
    cls: str,
    title: Optional[str] = None,
    attrs: Optional[Dict[str, str]] = None,
) -> None:
    """Wrap plain-text range [start, end) in the DOM by splitting text nodes.

    Important: start/end are offsets in the concatenated plain text of `nodes`.
    """
    if start >= end:
        return

    # Build cumulative offsets
    cum = 0
    i = 0
    # Find first node that overlaps start
    while i < len(nodes) and cum + len(str(nodes[i])) <= start:
        cum += len(str(nodes[i]))
        i += 1

    pos = start
    while i < len(nodes) and pos < end:
        node = nodes[i]
        s = str(node)
        node_start = cum
        node_end = cum + len(s)

        seg_start = max(pos, node_start)
        seg_end = min(end, node_end)

        # Split within this node
        before = s[: seg_start - node_start]
        mid = s[seg_start - node_start : seg_end - node_start]
        after = s[seg_end - node_start :]

        # Replace node with before + span(mid) + after
        new_nodes: List[NavigableString] = []
        if before:
            before_ns = NavigableString(before)
            node.insert_before(before_ns)
            new_nodes.append(before_ns)

        span = soup.new_tag("span")
        span["class"] = cls.split()
        if title:
            span["title"] = title
        if attrs:
            for k, v in attrs.items():
                span[k] = v
        span.string = mid
        node.insert_before(span)

        if after:
            after_ns = NavigableString(after)
            node.insert_before(after_ns)
            new_nodes.append(after_ns)

        # Remove old node
        node.extract()

        # Rebuild nodes list locally: replace current node with (before?) + (after?) around span
        # Note: span is NOT a NavigableString so not in nodes; we only track text nodes.
        # We also need to update `nodes` for subsequent wraps in the same call sequence.
        nodes.pop(i)
        # Insert after_ns as current position, and before_ns earlier already inserted before span.
        insert_at = i
        # We inserted before_ns before span, so it is earlier in DOM; keep order
        if before:
            nodes.insert(insert_at, before_ns)
            insert_at += 1
            i += 1  # move past before node
            cum += len(before)
            node_start += len(before)  # adjust, but we re-calc below anyway

        if after:
            nodes.insert(insert_at, after_ns)
            # do not advance i; we want to continue from the node that contains remaining text (after)
        # Update cum and positions
        # After replacement, the "current" text position becomes seg_end
        pos = seg_end
        # cum should stay at node_start for before? We will recompute cum by summing up to i at loop top next iteration.
        # Simpler: recompute cum from scratch for correctness (cost is acceptable).
        cum = 0
        for j in range(i):
            cum += len(str(nodes[j]))
        # If we wrapped up to end of original node, move to next node
        if seg_end >= node_end:
            # if after exists, we should stay at current i (after node) because it contains remaining text
            if not after:
                i += 1
        # else, we remain on after node at index i
    return


def _md_to_html(md_text: str) -> str:
    # Choose a conservative set of extensions (avoid touching HTML once produced)
    return markdown.markdown(
        md_text,
        extensions=[
            "extra",  # includes fenced_code, tables, etc.
            "tables",
            "sane_lists",
            "nl2br",
        ],
        output_format="html5",
    )

def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans.sort()
    out = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out

def _find_table_spans(text: str) -> List[Tuple[int, int]]:
    # 简单识别：连续多行包含 '|' 且中间有表头分隔行（---|---）
    spans = []
    lines = text.splitlines(keepends=True)
    idx = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        is_row = line.count("|") >= 2
        if is_row and i + 1 < len(lines):
            sep = lines[i + 1]
            is_sep = bool(re.match(r"^\s*\|?[\s:-]+\|[\s|:-]*\|?\s*$", sep))
            if is_sep:
                start = idx
                j = i + 2
                idx2 = idx + len(line) + len(sep)
                while j < len(lines) and lines[j].count("|") >= 2:
                    idx2 += len(lines[j])
                    j += 1
                spans.append((start, idx2))
                # advance
                for k in range(i, j):
                    idx += len(lines[k])
                i = j
                continue
        idx += len(line)
        i += 1
    return spans

def _find_protected_spans(md: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []

    # fenced code blocks
    for m in re.finditer(r"```[\s\S]*?```", md):
        spans.append((m.start(), m.end()))

    # inline code (avoid跨行)
    for m in re.finditer(r"`[^`\n]*`", md):
        spans.append((m.start(), m.end()))

    # links / images (简单版：不处理嵌套括号的极端 URL)
    for m in re.finditer(r"!?(\[[^\]\n]*\])(\([^\)\n]*\))", md):
        spans.append((m.start(), m.end()))

    # autolinks / html tags
    for m in re.finditer(r"<[^>\n]+>", md):
        spans.append((m.start(), m.end()))

    # reference-style link def: [id]: url
    for m in re.finditer(r"^\s*\[[^\]]+\]:\s+\S+.*$", md, flags=re.M):
        spans.append((m.start(), m.end()))

    # tables
    spans.extend(_find_table_spans(md))

    return _merge_spans(spans)

def _in_span(pos: int, spans: List[Tuple[int, int]]) -> bool:
    # spans 已合并且有序，小文本用线性扫够用；大文本可改二分
    for s, e in spans:
        if pos < s:
            return False
        if s <= pos < e:
            return True
    return False

def smart_slice_markdown_safe(text: str) -> List[str]:
    if len(text) < 10:
        return [text]

    spans = _find_protected_spans(text)

    # 收集可切分点：句末符号；以及 “，+连接词”
    breakpoints = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if not _in_span(i, spans):
            if ch in _SENT_END:
                breakpoints.append(i + 1)  # 在符号后切
            elif ch == "，":
                # 仅当逗号后是连接词才切
                after = text[i + 1 : i + 10]  # 看一小段即可
                if re.match(rf"^(?:{_CONNECTORS})", after):
                    breakpoints.append(i + 1)

        i += 1

    # 根据断点生成 slices（不 strip，保留原始空格/缩进）
    out = []
    last = 0
    for b in breakpoints:
        if b <= last:
            continue
        seg = text[last:b]
        if seg:
            out.append(seg)
        last = b
    tail = text[last:]
    if tail:
        out.append(tail)

    # 兜底：对超长段落再切，但仍只允许在安全逗号处切
    final = []
    for seg in out:
        if len(seg) <= 120:  # 你可调阈值
            final.append(seg)
            continue
        # 找 seg 内安全的逗号位置
        local_spans = _find_protected_spans(seg)
        cut_positions = [j for j, c in enumerate(seg) if c == "，" and not _in_span(j, local_spans)]
        if not cut_positions:
            final.append(seg)
            continue
        # 贪心切到接近 80~120
        start = 0
        for cp in cut_positions:
            if cp - start >= 80:
                final.append(seg[start:cp+1])
                start = cp + 1
        if start < len(seg):
            final.append(seg[start:])
    return final


CheckBatchAsyncFn = Callable[[List[Dict[str, Any]], str, Any], Awaitable[Dict[int, Dict[str, str]]]]

# --- Sentence-level segmentation helpers (operate on rendered HTML text nodes) ---

_BLOCK_TAGS = {
    "p", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "pre",
    "td", "th",
}

def _nearest_block_tag(node: NavigableString) -> str:
    """Return the nearest block-level ancestor tag name for a text node."""
    p = getattr(node, "parent", None)
    while p is not None:
        name = getattr(p, "name", None)
        if name in _BLOCK_TAGS:
            return name
        p = getattr(p, "parent", None)
    return "__root__"

def _has_visible_content(s: str) -> bool:
    # at least one digit/latin/cjk
    return bool(re.search(r"[0-9A-Za-z\u4e00-\u9fff]", s))

def _normalize_sentence(s: str) -> str:
    # remove all whitespace for stable matching
    return re.sub(r"\s+", "", (s or "").strip())

def _sent_spans_from_nodes(nodes: List[NavigableString]) -> Tuple[str, List[Tuple[int, int]], Set[int]]:
    """Split concatenated text nodes into sentence-like spans.

    Returns:
      plain: concatenated text (exactly matches nodes concatenation)
      spans: list of (start, end) offsets into `plain`
      boundaries: offsets where a block boundary forced a cut (do not merge across)
    """
    plain_parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    boundaries: Set[int] = set()

    start = 0
    cursor = 0
    prev_block: Optional[str] = None

    for n in nodes:
        block = _nearest_block_tag(n)

        # force a cut when entering a new block (prevents cross-paragraph sentences)
        if prev_block is not None and block != prev_block and start < cursor:
            spans.append((start, cursor))
            boundaries.add(cursor)
            start = cursor

        txt = str(n)
        plain_parts.append(txt)

        for ch in txt:
            cursor += 1
            if ch in _SENT_END:
                spans.append((start, cursor))
                start = cursor

        prev_block = block

    if start < cursor:
        spans.append((start, cursor))

    plain = "".join(plain_parts)
    return plain, spans, boundaries

def _merge_short_spans(
    spans: List[Tuple[int, int]],
    boundaries: Set[int],
    plain: str,
    *,
    min_chars: int = 45,
    max_chars: int = 260,
) -> List[Tuple[int, int]]:
    """Merge very short spans to reduce 'word-level' annotation noise.

    - Will NOT merge across block boundaries in `boundaries`.
    - Stops when merged length exceeds `max_chars`.
    """
    out: List[Tuple[int, int]] = []
    i = 0
    n = len(spans)

    while i < n:
        s, e = spans[i]
        j = i

        while True:
            seg = (plain[s:e] if e <= len(plain) else "").strip()

            # already good enough / last
            if len(seg) >= min_chars or j + 1 >= n:
                break

            # do not cross block boundary
            next_s, next_e = spans[j + 1]
            if next_s in boundaries:
                break

            # do not exceed max chars
            if next_e - s > max_chars:
                break

            j += 1
            e = next_e

        out.append((s, e))
        i = j + 1

    return out

def _sentence_set_from_markdown(md_text: str, *, min_chars: int = 45, max_chars: int = 260) -> Set[str]:
    """Render markdown->html, then build a normalized sentence set for 'unchanged' detection."""
    html = _md_to_html(md_text or "")
    soup = BeautifulSoup(html, "html.parser")
    nodes = _collect_text_nodes(soup)
    plain, spans, boundaries = _sent_spans_from_nodes(nodes)
    spans = _merge_short_spans(spans, boundaries, plain, min_chars=min_chars, max_chars=max_chars)

    out: Set[str] = set()
    for s, e in spans:
        seg = plain[s:e].strip()
        if not seg:
            continue
        if not _has_visible_content(seg):
            continue
        if len(seg) < max(8, min_chars // 3):
            continue
        out.add(_normalize_sentence(seg))
    return out



def _chunked(lst: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

class SmartVerifierCore:
    """
    核心流程（异步 + 切片级并发 + 批量审计）：
    1) diff original vs merged
    2) 只抽取新增(op==1) -> smart_slice -> ai_queue
    3) 切片级并发 submit（max_concurrency），内部微批聚合调用 check_batch_fn
    4) 输出 HTML（稳定：Markdown -> HTML -> wrap 文本节点）
    """

    def __init__(self) -> None:
        self.dmp = dmp_module.diff_match_patch()

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
                # slices = self.smart_slice(text)
                slices = smart_slice_markdown_safe(text)
                for s in slices:
                    if s.strip() == "":
                        final_segments.append({"type": "equal", "text": s})
                        continue
                    t_id = task_id_counter
                    task_id_counter += 1
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
        check_batch_fn: "CheckBatchAsyncFn",
        batch_size: int = 10,
        max_inflight_batches: int = 4,
        timeout_s: Optional[float] = None,
        default_label: str = "UNKNOWN",
        # sentence-level knobs
        min_sentence_chars: int = 20,
        max_sentence_chars: int = 90,
    ) -> str:
        """Return annotated HTML (stable) with **sentence-level** labels.

        Why sentence-level:
        - diff_main often marks very small insertions (a word / number) which makes annotation extremely碎;
        - sentence-level plan expands 'changed' detection to whole sentences/clauses, which is what humans want.
        """

        merged_md = merged or ""
        base_html = _md_to_html(merged_md)

        soup = BeautifulSoup(base_html, "html.parser")
        nodes = _collect_text_nodes(soup)

        # Build sentence spans on rendered plain text (offsets match DOM text nodes)
        plain, spans, boundaries = _sent_spans_from_nodes(nodes)
        spans = _merge_short_spans(
            spans,
            boundaries,
            plain,
            min_chars=min_sentence_chars,
            max_chars=max_sentence_chars,
        )

        # Sentences that already exist in original => treat as equal (no verification)
        original_set = _sentence_set_from_markdown(
            original or "",
            min_chars=min_sentence_chars,
            max_chars=max_sentence_chars,
        )

        def _clip(s: str, n: int = 80) -> str:
            s = (s or "").strip()
            if len(s) <= n:
                return s
            return s[:n] + "…"

        plan: List[Dict[str, Any]] = []
        ai_queue: List[Dict[str, Any]] = []
        task_id_counter = 0

        for i, (s, e) in enumerate(spans):
            seg = plain[s:e]
            seg_stripped = seg.strip()

            if (not seg_stripped) or (not _has_visible_content(seg_stripped)):
                plan.append({"type": "equal", "start": s, "end": e})
                continue

            norm = _normalize_sentence(seg_stripped)
            if norm in original_set:
                plan.append({"type": "equal", "start": s, "end": e})
                continue

            rid = task_id_counter
            task_id_counter += 1

            prev_seg = plain[spans[i - 1][0] : spans[i - 1][1]] if i > 0 else ""
            next_seg = plain[spans[i + 1][0] : spans[i + 1][1]] if i + 1 < len(spans) else ""

            # 保持你原本思想：context 包裹当前句，同时加上前后文减少 UNKNOWN。
            ctx = (
                f"前文：{_clip(prev_seg)}\n"
                f"当前：{seg_stripped}\n"
                f"后文：{_clip(next_seg)}"
            )

            ai_queue.append({"id": rid, "text": seg_stripped, "context": ctx})
            plan.append({"type": "pending", "id": rid, "start": s, "end": e})

        # Run LLM checks (batched + concurrency-limited)
        ai_results_map: Dict[int, str] = {}
        ai_results_quote_map: Dict[int, str] = {}
        sem = asyncio.Semaphore(max_inflight_batches)

        async def _run_batch(batch: List[Dict[str, Any]]) -> None:
            async with sem:
                try:
                    if timeout_s:
                        res = await asyncio.wait_for(
                            check_batch_fn(batch, evidence, model),
                            timeout=timeout_s,
                        )
                    else:
                        res = await check_batch_fn(batch, evidence, model)
                except Exception:
                    # Any failure => default label (UNKNOWN) for the whole batch
                    res = {}

                if not isinstance(res, dict):
                    return

                for k, v in res.items():
                    try:
                        rid = int(k)
                    except Exception:
                        continue

                    # 兼容两种返回：
                    # 1) {id: {"label": "...", "quote": "..."}}
                    # 2) {id: "VALID"/"HALLUCINATION"/"UNKNOWN"}
                    if isinstance(v, dict):
                        label = str(v.get("label", "")).upper().strip() or default_label
                        quote = str(v.get("quote", "") or "").strip()
                    else:
                        label = str(v).upper().strip() or default_label
                        quote = ""

                    if label not in ("VALID", "HALLUCINATION", "UNKNOWN"):
                        label = default_label

                    ai_results_map[rid] = label
                    ai_results_quote_map[rid] = quote

        tasks: List[asyncio.Task] = []
        for batch in _chunked(ai_queue, batch_size):
            tasks.append(asyncio.create_task(_run_batch(batch)))
        if tasks:
            await asyncio.gather(*tasks)

        # Apply wrapping on sentence spans (ordered, non-overlapping)
        for seg in plan:
            if seg.get("type") != "pending":
                continue

            rid = int(seg["id"])
            start = int(seg["start"])
            end = int(seg["end"])

            label = (ai_results_map.get(rid) or default_label).strip().upper()
            quote = (ai_results_quote_map.get(rid) or "").strip()

            if label == "HALLUCINATION":
                cls = "vfy-hal"
                title = "证据未覆盖AI自动推演"
            elif label == "VALID":
                cls = "vfy-valid"
                title = f"证据命中：{quote}" if quote else "证据命中"
            else:
                cls = "vfy-unk"
                title = "证据不足难以判断"

            _wrap_range_in_dom(soup, nodes, start, end, cls=cls, title=title)

        # Deleted (removed) text: show in a dedicated block so users can see what was dropped.
        diffs = self.dmp.diff_main(original or "", merged or "")
        self.dmp.diff_cleanupSemantic(diffs)
        deleted_text = "".join(t for op, t in diffs if op == -1 and t and t.strip())

        if deleted_text.strip():
            block = soup.new_tag("div", attrs={"class": "vfy-block-del"})
            h4 = soup.new_tag("h4")
            h4.string = "删除内容"
            pre = soup.new_tag("pre")
            pre.string = deleted_text
            block.append(h4)
            block.append(pre)
            soup.append(block)

        return _VFY_CSS + str(soup)

    async def verify_diff_only_async(
        self,
        *,
        original: str,
        merged: str,
        min_sentence_chars: int = 45,
        max_sentence_chars: int = 260,
    ) -> str:
        """Pure diff highlighting (no LLM): mark sentences that are *not* present in original.""" 
        merged_md = merged or ""
        base_html = _md_to_html(merged_md)

        soup = BeautifulSoup(base_html, "html.parser")
        nodes = _collect_text_nodes(soup)

        plain, spans, boundaries = _sent_spans_from_nodes(nodes)
        spans = _merge_short_spans(
            spans,
            boundaries,
            plain,
            min_chars=min_sentence_chars,
            max_chars=max_sentence_chars,
        )

        original_set = _sentence_set_from_markdown(
            original or "",
            min_chars=min_sentence_chars,
            max_chars=max_sentence_chars,
        )

        for s, e in spans:
            seg = plain[s:e].strip()
            if (not seg) or (not _has_visible_content(seg)):
                continue
            if _normalize_sentence(seg) in original_set:
                continue
            _wrap_range_in_dom(soup, nodes, s, e, cls="vfy-added", title="新增/修改（未验证）")

        # Deleted text block
        diffs = self.dmp.diff_main(original or "", merged or "")
        self.dmp.diff_cleanupSemantic(diffs)
        deleted_text = "".join(t for op, t in diffs if op == -1 and t and t.strip())
        if deleted_text.strip():
            block = soup.new_tag("div", attrs={"class": "vfy-block-del"})
            h4 = soup.new_tag("h4")
            h4.string = "删除内容"
            pre = soup.new_tag("pre")
            pre.string = deleted_text
            block.append(h4)
            block.append(pre)
            soup.append(block)

        return _VFY_CSS + str(soup)
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
        # print("标记：", data)
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

