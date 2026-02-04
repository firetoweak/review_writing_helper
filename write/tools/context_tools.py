from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple, Iterable, Optional

import httpx


# ----------------------------
# Embedding client (sglang /v1/embeddings)
# ----------------------------

class EmbeddingClient:
    def __init__(self, *, base_url: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        url = f"{self.base_url}/v1/embeddings"
        r = self.client.post(url, json={"model": self.model, "input": texts})
        r.raise_for_status()
        data = r.json()
        return [item["embedding"] for item in data["data"]]

    def embed_batched(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self.embed(texts[i:i + batch_size]))
        return out


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


# ----------------------------
# Helpers
# ----------------------------

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def chunk_text(text: str, *, chunk_size: int = 1000, overlap: int = 150) -> Iterable[str]:
    t = (text or "").strip()
    if not t:
        return
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(t), step):
        yield t[i:i + chunk_size].strip()

def compress(s: str, max_chars: int = 1200) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    head = s[:800]
    tail = s[-250:]
    return (head + "\n...\n" + tail).strip()

def safe_str(x: Any) -> str:
    return str(x or "").strip()


# ----------------------------
# Parse / normalize inputs
# ----------------------------

def flatten_history(history_text_list: Any) -> Dict[str, Dict[str, str]]:
    """
    historyTextList:
    [
      {chapterId, chapterTitle, children:[{sectionId, sectionTitle, text}, ...]},
      ...
    ]
    -> sectionId -> meta
    """
    # print("[FH] type=", type(history_text_list).__name__,
    #     "len=", len(history_text_list) if isinstance(history_text_list, list) else None)
    # if isinstance(history_text_list, list) and history_text_list:
    #     print("[FH] first_keys=", list(history_text_list[0].keys()))
    #     print("[FH] first_chapterId=", history_text_list[0].get("chapterId"),
    #         "children_type=", type(history_text_list[0].get("children")).__name__)


    out: Dict[str, Dict[str, str]] = {}
    if not isinstance(history_text_list, list):
        return out

    for ch in history_text_list:
        if not isinstance(ch, dict):
            continue
        cid = safe_str(ch.get("chapterId"))
        ct = safe_str(ch.get("chapterTitle"))
        children = ch.get("children") or []
        if not isinstance(children, list):
            continue

        for sec in children:
            if not isinstance(sec, dict):
                continue
            sid = safe_str(sec.get("sectionId"))
            st = safe_str(sec.get("sectionTitle") or sec.get("title"))
            tx = safe_str(sec.get("text"))
            if not sid:
                continue
            out[sid] = {
                "chapterId": cid,
                "chapterTitle": ct,
                "sectionId": sid,
                "sectionTitle": st,
                "text": tx,
            }
    return out


def normalize_neighbors_map(neighbors_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    - 保证结构合法：neighbors.chapters.{chapterId}.sections / relatedSections
    - 清理：去重、去自身、空值
    """
    nm = neighbors_map or {}
    neighbors = nm.get("neighbors") or {}
    chapters = neighbors.get("chapters") or {}
    if not isinstance(chapters, dict):
        return {"neighbors": {"chapters": {}}}

    # 收集所有 sectionId（用于过滤非法ID时可选）
    all_sections = set()
    for _, ch in chapters.items():
        if not isinstance(ch, dict):
            continue
        sec_map = ch.get("sections") or {}
        if isinstance(sec_map, dict):
            for sid in sec_map.keys():
                all_sections.add(str(sid).strip())
            for rels in sec_map.values():
                if isinstance(rels, list):
                    for rid in rels:
                        all_sections.add(str(rid).strip())
        rel_secs = ch.get("relatedSections") or []
        if isinstance(rel_secs, list):
            for rid in rel_secs:
                all_sections.add(str(rid).strip())

    # normalize 每章
    new_chapters: Dict[str, Any] = {}
    for cid, ch in chapters.items():
        cid = safe_str(cid)
        if not cid or not isinstance(ch, dict):
            continue

        sec_map = ch.get("sections") or {}
        rel_ch = ch.get("relatedSections") or []
        if not isinstance(sec_map, dict):
            sec_map = {}
        if not isinstance(rel_ch, list):
            rel_ch = []

        # 先清 chapter.relatedSections
        rel_ch_clean = [safe_str(x) for x in rel_ch]
        rel_ch_clean = [x for x in rel_ch_clean if x]
        rel_ch_clean = dedupe_keep_order(rel_ch_clean)

        # 再清每个 section 的 related
        new_sec_map: Dict[str, List[str]] = {}
        for sid, rels in sec_map.items():
            sid = safe_str(sid)
            if not sid:
                continue
            if not isinstance(rels, list):
                rels = []
            rels_clean = [safe_str(x) for x in rels]
            rels_clean = [x for x in rels_clean if x and x != sid]  # 去自身
            rels_clean = dedupe_keep_order(rels_clean)
            new_sec_map[sid] = rels_clean

        new_chapters[cid] = {
            "relatedSections": rel_ch_clean,
            "sections": new_sec_map,
        }

    return {"neighbors": {"chapters": new_chapters}}


# ----------------------------
# Candidate selection (mapping -> candidate sections)
# ----------------------------

def get_candidates(
    *,
    chapter_id: str,
    section_id: str,
    neighbors_map: Dict[str, Any],
    history_index: Dict[str, Dict[str, str]],
    max_candidates: int = 60,
) -> List[str]:
    chapters = ((neighbors_map.get("neighbors") or {}).get("chapters") or {})
    ch = chapters.get(chapter_id) or {}
    sec_map = ch.get("sections") or {}
    rel_secs = sec_map.get(section_id) or []
    ch_rel = ch.get("relatedSections") or []

    # 候选：自己 + 映射相关 + chapter级相关
    cands = [section_id] + list(rel_secs) + list(ch_rel)

    # 过滤：必须在 history 里存在且有正文
    out: List[str] = []
    seen = set()
    for sid in cands:
        sid = safe_str(sid)
        if not sid or sid in seen:
            continue
        seen.add(sid)
        meta = history_index.get(sid)
        if not meta:
            continue
        if not meta.get("text"):
            continue
        out.append(sid)
        if len(out) >= max_candidates:
            break
    return out


# ----------------------------
# Retrieval + packing
# ----------------------------

def build_query(
    *,
    title: str,
    idea: str,
    section_id: str,
    section_title: str,
    current_text: str,
    extra_intent: str = "",
) -> str:
    return "\n".join([
        f"title: {safe_str(title)}",
        f"idea: {compress(idea, 600)}",
        f"target_section: {section_id} {safe_str(section_title)}",
        f"intent: {compress(extra_intent, 400)}",
        f"current_text: {compress(current_text, 1200)}",
    ]).strip()


def pack_blocks(blocks: List[Dict[str, str]], max_chars: int = 6000) -> str:
    parts: List[str] = []
    used = 0
    for b in blocks:
        header = f"[Chapter {b['chapterId']} {b['chapterTitle']}]\n[Section {b['sectionId']} {b['sectionTitle']}]"
        body = b["text"].strip()
        if not body:
            continue
        chunk = (header + "\n" + body).strip()
        add = len(chunk) + 2
        if used + add > max_chars:
            break
        parts.append(chunk)
        used += add
    return "\n\n".join(parts).strip()


def retrieve_history_context_for_section(
    *,
    embedding: EmbeddingClient,
    neighbors_map: Dict[str, Any],
    history_text_list: Any,
    title: str,
    idea: str,
    target_section_id: str,
    target_section_title: str,
    current_text: str,
    chapter_id_hint: Optional[str] = None,
    extra_intent: str = "",
    # params
    chunk_size: int = 1000,
    overlap: int = 150,
    embed_batch_size: int = 16,
    topk_chunks: int = 10,
    per_section_cap: int = 3,
    max_context_chars: int = 6000,
) -> str:
    # 建 history 索引
    hist = flatten_history(history_text_list)


    if not hist:
        return ""

    # 目标章：优先 hint；否则从 history 反查
    section_id = safe_str(target_section_id)
    chapter_id = (
        safe_str(chapter_id_hint)
        or safe_str(hist.get(section_id, {}).get("chapterId"))
        or (section_id.split(".", 1)[0] if "." in section_id else "")
    )

    candidates = get_candidates(
        chapter_id=chapter_id,
        section_id=section_id,
        neighbors_map=neighbors_map,
        history_index=hist,
    )
    # print("[RAG][CAND] candidates_n=", len(candidates), "head=", candidates[:20])


    # print("===========图候选阶段============")
    # print("candidates:", candidates)
    if not candidates:
        return ""

    # 构造 query 向量
    q = build_query(
        title=title, idea=idea,
        section_id=section_id,
        section_title=target_section_title or hist.get(section_id, {}).get("sectionTitle", ""),
        current_text=current_text,
        extra_intent=extra_intent,
    )
    qv = embedding.embed([q])[0]

    # 准备候选 chunks（带 meta）
    chunk_texts: List[str] = []
    metas: List[Dict[str, str]] = []

    for sid in candidates:
        meta = hist[sid]
        for ck in chunk_text(meta["text"], chunk_size=chunk_size, overlap=overlap):
            if not ck:
                continue
            chunk_texts.append(ck)
            metas.append({
                "chapterId": meta["chapterId"],
                "chapterTitle": meta["chapterTitle"],
                "sectionId": meta["sectionId"],
                "sectionTitle": meta["sectionTitle"],
                "text": ck,
            })

    if not chunk_texts:
        return ""

    # 批量 embedding + 打分
    vecs = embedding.embed_batched(chunk_texts, batch_size=embed_batch_size)

    scored: List[Tuple[float, Dict[str, str]]] = []
    for v, m in zip(vecs, metas):
        scored.append((cosine(qv, v), m))
    scored.sort(key=lambda x: x[0], reverse=True)

    # TopK + 每节限额
    picked: List[Dict[str, str]] = []
    count_by_sec: Dict[str, int] = {}
    for s, m in scored:
        sid = m["sectionId"]
        if count_by_sec.get(sid, 0) >= per_section_cap:
            continue
        picked.append(m)
        count_by_sec[sid] = count_by_sec.get(sid, 0) + 1
        if len(picked) >= topk_chunks:
            break

    return pack_blocks(picked, max_chars=max_context_chars)


def retrieve_history_context_for_textlist(
    *,
    embedding: EmbeddingClient,
    neighbors_map: Dict[str, Any],
    history_text_list: Any,
    title: str,
    idea: str,
    text_list: Any,
    extra_intent: str = "",
    max_context_chars: int = 3000,
    total_context_chars: int = 10000, 
    min_per_section: int = 800,             # 可选：每节最少给多少额度（不足则不给）
) -> Dict[str, str]:
    """
    textList 可能是一节，也可能是一章（多节）
    返回：sectionId -> context_str
    """
    out: Dict[str, str] = {}
    if not isinstance(text_list, list):
        return out

    remaining = max(0, int(total_context_chars))

    for it in text_list:
        if not isinstance(it, dict):
            continue
        sid = safe_str(it.get("sectionId"))
        st = safe_str(it.get("sectionTitle"))
        tx = safe_str(it.get("text"))
        if not sid:
            continue

        budget = min(int(max_context_chars), remaining)
        if budget < min_per_section:
            # 剩余太少，不值得再塞上下文了（也可以改成继续塞）
            break

        ctx = retrieve_history_context_for_section(
            embedding=embedding,
            neighbors_map=neighbors_map,
            history_text_list=history_text_list,
            title=title,
            idea=idea,
            target_section_id=sid,
            target_section_title=st,
            current_text=tx,
            extra_intent=extra_intent,
            max_context_chars=budget,
        )
        out[sid] = ctx
        remaining -= len(ctx)  

    return out
