"""
✅ 统一占位符注入工具（给所有 agent 复用）
- 只替换 {identifier}（英文变量名），避免误伤 JSON 示例里的 {}
- 仅替换白名单 PLACEHOLDER_MAP 中声明的变量；未声明则保留原样
- payload/list/dict 会先序列化成稳定字符串，缺失字段默认空串
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Literal
from tools.context_tools import retrieve_history_context_for_textlist, EmbeddingClient, normalize_neighbors_map

# 只匹配 {title} 这类占位符（不匹配 { "a": 1 }）    
_VAR = re.compile(r"\{([a-zA-Z_]\w*)\}")


# -------------------------
# Placeholder whitelist
# -------------------------

DEFAULT_PLACEHOLDER_MAP: Dict[str, str] = {
    # project
    "title": "title",
    "idea": "idea",
    "materials": "materials",
    "industry": "industry",
    "industryNameList": "industryNameList",

    # session
    "sessionId": "sessionId",

    # writing rules
    "fullWriteRule": "fullWriteRule",
    "chapterWriteRule": "chapterWriteRule",
    "sectionWriteRule": "sectionWriteRule",

    # review rules / text
    "fullReviewText": "fullReviewText",
    "chapterReviewRule": "chapterReviewRule",
    "sectionReviewRule": "sectionReviewRule",
    "review": "review",
    "reviewList": "reviewList",

    # outline / text / history
    "outline": "outline",
    "textList": "textList",
    "historyTextList": "historyTextList",

    # current focus
    "chapterId": "chapterId",
    "chapterTitle": "chapterTitle",
    "sectionId": "sectionId",
    "sectionTitle": "sectionTitle",

    # help
    "helpText": "helpText",

    # merge/chat
    "sessionList": "sessionList",
    "sessionQA": "sessionQA",
    "uncorrectText": "uncorrectText",

    # runtime injectables (agents can set these at runtime)
    "draft": "draft",
    "qa": "qa",
}

MAPPING = {
"neighbors": {
"chapters": {
"1": {
"sections": {
"1.1": ["1.2", "1.3", "1.5", "1.4", "2.1", "2.6", "3.8", "5.2"],
"1.2": ["1.1", "1.3", "1.5", "1.4", "2.2", "2.4", "3.1", "3.2"],
"1.3": ["1.2", "1.4", "1.1", "1.5", "2.6", "3.8", "5.2"],
"1.4": ["1.3", "1.5", "1.2", "3.4", "3.3", "2.5", "2.3"],
"1.5": ["1.4", "1.3", "1.2", "1.1", "2.6", "3.8", "5.2", "2.2"]
},
"relatedSections": ["1.2", "1.3", "1.5", "1.4", "2.1", "2.6", "3.8", "5.2", "1.1", "2.2", "2.4", "3.1", "3.2", "3.4", "3.3", "2.5", "2.3"]
},
"2": {
"sections": {
"2.1": ["2.2", "2.3", "1.1", "1.5", "2.6", "3.5", "3.8"],
"2.2": ["2.1", "2.3", "2.4", "1.2", "3.1", "3.5", "2.6", "3.2"],
"2.3": ["2.2", "2.4", "2.5", "3.3", "3.5", "4.1", "1.4"],
"2.4": ["2.3", "2.5", "2.2", "3.7", "3.1", "1.2", "2.6"],
"2.5": ["2.4", "2.6", "2.3", "3.3", "4.1", "4.8", "3.5"],
"2.6": ["2.5", "2.4", "2.2", "1.5", "1.1", "3.8", "5.2", "3.6"]
},
"relatedSections": ["2.2", "2.3", "1.1", "1.5", "2.6", "3.5", "3.8", "2.1", "2.4", "1.2", "3.1", "3.2", "2.5", "3.3", "4.1", "1.4", "3.7", "4.8", "5.2", "3.6"]
},
"3": {
"sections": {
"3.1": ["3.2", "3.3", "1.2", "2.2", "2.4", "3.7", "3.8"],
"3.2": ["3.1", "3.3", "3.5", "1.2", "1.4", "2.2", "3.7", "3.8"],
"3.3": ["3.2", "3.4", "3.5", "2.5", "4.1", "4.8", "1.4"],
"3.4": ["3.3", "3.5", "1.4", "3.2", "4.8", "2.5"],
"3.5": ["3.4", "3.6", "3.2", "3.3", "3.7", "2.2", "2.3", "4.1"],
"3.6": ["3.5", "3.7", "2.2", "2.6", "5.2", "5.1", "3.8"],
"3.7": ["3.6", "3.8", "3.5", "2.4", "2.2", "3.1", "4.8"],
"3.8": ["3.7", "3.6", "1.5", "2.6", "5.2", "4.4", "4.8"]
},
"relatedSections": ["3.2", "3.3", "1.2", "2.2", "2.4", "3.7", "3.8", "3.1", "3.5", "1.4", "3.4", "2.5", "4.1", "4.8", "3.6", "2.3", "2.6", "5.2", "5.1", "1.5", "4.4"]
},
"4": {
"sections": {
"4.1": ["4.2", "4.3", "2.5", "3.3", "3.5", "4.8", "4.4"],
"4.2": ["4.1", "4.3", "4.8", "4.7", "3.5", "2.5"],
"4.3": ["4.2", "4.4", "4.7", "4.8", "2.5", "3.2", "3.5"],
"4.4": ["4.3", "4.5", "4.6", "4.7", "3.8", "5.1", "5.2"],
"4.5": ["4.4", "4.6", "4.1", "3.5", "3.8", "2.1"],
"4.6": ["4.5", "4.7", "4.4", "4.1", "2.5", "4.8"],
"4.7": ["4.6", "4.8", "4.4", "5.1", "5.2", "4.3"],
"4.8": ["4.7", "4.6", "4.4", "2.5", "3.3", "5.2", "3.8"]
},
"relatedSections": ["4.2", "4.3", "2.5", "3.3", "3.5", "4.8", "4.4", "4.1", "4.7", "3.2", "4.5", "4.6", "3.8", "5.1", "5.2", "2.1"]
},
"5": {
"sections": {
"5.1": ["5.2", "4.7", "4.4", "2.5", "3.6", "3.8"],
"5.2": ["5.1", "3.8", "2.6", "1.5", "4.8", "4.4", "3.6"]
},
"relatedSections": ["5.2", "4.7", "4.4", "2.5", "3.6", "3.8", "5.1", "2.6", "1.5", "4.8"]
}
}
}
}


# -------------------------
# Serialization helpers
# -------------------------

def _json(obj: Any, *, indent: int = 2) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, indent=indent)
    except Exception:
        return str(obj)


def _serialize_text_list(text_list: Any) -> str:
    """
    textList: [{sectionId, sectionTitle, text, image_url/attachments_url}, ...] -> stable string
    - 如果存在 image_url（或旧字段 attachments_url），会把图片标签列表附在文本中，确保多模态可被 build_messages() 识别。
    """
    if not isinstance(text_list, list):
        return ""
    lines: List[str] = []
    for it in text_list:
        if not isinstance(it, dict):
            continue
        sid = str(it.get("sectionId") or "").strip()
        st = str(it.get("sectionTitle") or "").strip()
        tx = str(it.get("text") or "").strip()

        head = f"{sid} {st}".strip()
        if head:
            lines.append(head)
        if tx:
            lines.append(tx)

        img_map = it.get("image_url") or it.get("attachments_url") or {}
        if isinstance(img_map, dict) and img_map:
            tags = [str(k).strip() for k in img_map.keys() if str(k).strip()]
            if tags:
                lines.append("Images: " + " ".join(tags))

        lines.append("")
    return "\n".join(lines).strip()



def _serialize_history_text_list(history: Any) -> str:
    """
    historyTextList:
    [
      {chapterId, chapterTitle, children:[{sectionId, sectionTitle, text, attachments_url}, ...]},
      ...
    ] -> stable string
    """
    if not isinstance(history, list):
        return ""
    lines = []
    for ch in history:
        if not isinstance(ch, dict):
            continue
        cid = str(ch.get("chapterId") or "").strip()
        ct = str(ch.get("chapterTitle") or "").strip()
        if cid or ct:
            lines.append(f"Chapter {cid} {ct}".strip())

        children = ch.get("children") or []
        if isinstance(children, list):
            for sec in children:
                if not isinstance(sec, dict):
                    continue
                sid = str(sec.get("sectionId") or "").strip()
                st = str(sec.get("sectionTitle") or "").strip()
                tx = str(sec.get("text") or "").strip()
                if sid or st:
                    lines.append(f"- {sid} {st}".strip())
                if tx:
                    lines.append(f"  {tx}")
        lines.append("")
    return "\n".join(lines).strip()


# -------------------------
# Multimodal + Session QA helpers
# -------------------------

_IMAGE_TAG_RE = re.compile(r"\[IMAGE_\d+\]")

def _safe_str(x: Any) -> str:
    return str(x or "").strip()

def _compress_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    head = s[: max(0, max_chars - 280)]
    tail = s[-250:] if max_chars > 300 else ""
    return (head + ("\n...\n" + tail if tail else "")).strip()

def _norm_role(role: Any) -> str:
    r = _safe_str(role).lower()
    if r in ("assistant", "assisant", "ai"):
        return "assistant"
    if r in ("user", "human"):
        return "user"
    return r or "user"

def _tags_line(img_map: Any) -> str:
    if not isinstance(img_map, dict) or not img_map:
        return ""
    tags = [str(k).strip() for k in img_map.keys() if str(k).strip()]
    return ("Images: " + " ".join(tags)).strip() if tags else ""

def _serialize_attachments(atts: Any, *, max_each_chars: int = 800) -> str:
    if not isinstance(atts, list) or not atts:
        return ""
    parts: List[str] = []
    for i, a in enumerate(atts, start=1):
        if not isinstance(a, dict):
            continue
        atext = _compress_text(_safe_str(a.get("text")), max_each_chars)
        line = _tags_line(a.get("image_url") or a.get("attachments_url"))
        block_parts: List[str] = []
        if atext:
            block_parts.append(atext)
        if line:
            block_parts.append(line)
        if not block_parts:
            continue
        parts.append(f"[Attachment {i}]\n" + "\n".join(block_parts))
    return "\n\n".join(parts).strip()

def _serialize_message(m: Mapping[str, Any], *, max_content_chars: int = 1200) -> str:
    content = _compress_text(_safe_str(m.get("content")), max_content_chars)
    img_line = _tags_line(m.get("image_url") or m.get("attachments_url"))
    att_block = _serialize_attachments(m.get("attachments"), max_each_chars=800)

    parts: List[str] = []
    if content:
        parts.append(content)
    if img_line:
        parts.append(img_line)
    if att_block:
        parts.append(att_block)
    return "\n".join(parts).strip()

def _serialize_messages_to_qa(messages: Any, *, max_pairs: int = 20) -> str:
    """
    新接口结构：messages 只有 role/content（不再依赖 type=='question'）
    规则：Q=用户(user/human)消息；A=其后连续 assistant 消息（可多条合并）
    """
    if not isinstance(messages, list) or not messages:
        return ""

    pairs: List[tuple[str, str]] = []
    current_q: Optional[str] = None
    current_a_parts: List[str] = []

    def flush():
        nonlocal current_q, current_a_parts
        if current_q is None:
            return
        q = (current_q or "").strip()
        a = "\n".join([p for p in (x.strip() for x in current_a_parts) if p]).strip()
        pairs.append((q, a))
        current_q = None
        current_a_parts = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = _norm_role(msg.get("role"))
        txt = _serialize_message(msg)
        if not txt:
            continue

        if role == "user":
            if current_q is not None:
                flush()
                if len(pairs) >= max_pairs:
                    break
            current_q = txt
            current_a_parts = []
        elif role == "assistant":
            if current_q is None:
                current_q = "（上下文：助手补充说明）"
                current_a_parts = [txt]
            else:
                current_a_parts.append(txt)
        else:
            if current_q is None:
                current_q = txt
            else:
                current_a_parts.append(txt)

    if len(pairs) < max_pairs and current_q is not None:
        flush()

    pairs = pairs[-max_pairs:]

    out_lines: List[str] = []
    for i, (q, a) in enumerate(pairs, start=1):
        out_lines.append(f"Q{i}: {q}".strip())
        out_lines.append(f"A{i}: {a}".strip() if a else f"A{i}: （无）")
        out_lines.append("")
    return "\n".join(out_lines).strip()

def _serialize_session_list_qa(
    session_list: Any,
    *,
    max_sessions: int = 5,
    max_pairs_per_session: int = 20,
    max_total_chars: int = 8000,
) -> str:
    if not isinstance(session_list, list) or not session_list:
        return ""
    sessions = [s for s in session_list if isinstance(s, dict)]
    sessions = sessions[-max_sessions:]

    blocks: List[str] = []
    for s in sessions:
        sid = _safe_str(s.get("sessionId")) or _safe_str(s.get("id")) or ""
        msgs = s.get("messages") or []
        qa = _serialize_messages_to_qa(msgs, max_pairs=max_pairs_per_session)
        if not qa:
            continue
        header = f"[Session {sid}]".strip() if sid else "[Session]"
        blocks.append(header + "\n" + qa)

    text = "\n\n".join(blocks).strip()
    return _compress_text(text, max_total_chars) if text else ""

from typing import Any, Dict, Mapping


def build_image_map(payload: Mapping[str, Any]) -> Dict[str, str]:
    """
    汇总 payload 中所有图片标签 -> url 映射（支持 image_url / 旧字段 attachments_url / image_maps）
    覆盖范围：
    - textList[*].image_url / attachments_url
    - historyTextList[*].children[*].image_url / attachments_url
    - sessionList[*].messages[*].image_url / attachments_url
    - sessionList[*].messages[*].attachments[*].image_url / attachments_url
    - image_maps: {doc_id: {tag:url}} 或 {tag:url}（兼容）
    """
    p = dict(payload or {})
    out: Dict[str, str] = {}
    def _is_valid_media_url(u: str) -> bool:
        u = (u or "").strip()
        if not u:
            return False
        # ✅ 只允许线上 URL（最安全）
        if u.startswith(("http://", "https://")):
            return True
        # 可选：如果你确认后端/模型服务支持 data url，再放开
        if u.startswith("data:image/"):
            return True
        return False

    def add_map(m: Any):
        """m expected: {tag:url}"""
        if not isinstance(m, dict):
            return
        for k, v in m.items():
            kk = _safe_str(k)
            vv = _safe_str(v)
            if not kk or not vv:
                continue
            # ✅ 兜底：过滤 file:// / /home/... / C:\... 等本地路径
            if not _is_valid_media_url(vv):
                continue
            if kk not in out:
                out[kk] = vv

    def _safe_str(x: Any) -> str:
        if x is None:
            return ""
        try:
            s = str(x).strip()
            return s
        except Exception:
            return ""

    # 1) textList
    tl = p.get("textList")
    if isinstance(tl, list):
        for it in tl:
            if isinstance(it, dict):
                add_map(it.get("image_url") or it.get("attachments_url"))

    # 2) historyTextList
    hist = p.get("historyTextList")
    if isinstance(hist, list):
        for ch in hist:
            if not isinstance(ch, dict):
                continue
            for sec in (ch.get("children") or []):
                if isinstance(sec, dict):
                    add_map(sec.get("image_url") or sec.get("attachments_url"))
    
    # *) 全文，格式和历史一样
    ft = p.get("fullText")
    if isinstance(ft, list):
        for ch in ft:
            if not isinstance(ch, dict):
                continue
            for sec in (ch.get("children") or []):
                if isinstance(sec, dict):
                    add_map(sec.get("image_url") or sec.get("attachments_url"))

    # 3) sessionList/messages/attachments
    sl = p.get("sessionList")
    if isinstance(sl, list):
        for sess in sl:
            if not isinstance(sess, dict):
                continue
            for msg in (sess.get("messages") or []):
                if not isinstance(msg, dict):
                    continue
                add_map(msg.get("image_url") or msg.get("attachments_url"))
                for att in (msg.get("attachments") or []):
                    if isinstance(att, dict):
                        add_map(att.get("image_url") or att.get("attachments_url"))

    # 4) NEW: image_maps
    # 支持两种形态：
    #   A) {"doc_1": {"[IMAGE_1]": "http://..."}, "doc_2": {...}}
    #   B) {"[IMAGE_1]": "http://..."}  (有些地方可能直接给扁平 map)
    ims = p.get("image_maps")
    if isinstance(ims, dict):
        # 如果 value 里“至少有一个 dict”，按 {doc_id:{tag:url}} 处理
        has_nested = any(isinstance(v, dict) for v in ims.values())
        if has_nested:
            for _, m in ims.items():
                add_map(m)
        else:
            # 扁平 map
            add_map(ims)

    return out


def prompt_has_image_tags(text: str) -> bool:
    return bool(_IMAGE_TAG_RE.search(text or ""))




def replace_image_tags_with_markdown(
    text: str,
    image_map: Dict[str, str],
    *,
    mode: Literal["image", "link"] = "image",
    alt_prefix: str = "IMAGE",
    strict_missing: bool = False,
    ensure_separation: bool = True,
) -> str:
    """
    把 text 中的 [IMAGE_x] 替换为 markdown:
      - mode="image": ![IMAGE_1](<url>)
      - mode="link" : [IMAGE_1](<url>)

    参数：
    - strict_missing: True 时遇到缺失映射就抛错；False 时保留原 tag
    - ensure_separation: True 时在 tag 两侧尽量补空格/换行，避免紧贴文字导致渲染不稳
    """
    if not text:
        return ""
    if not isinstance(image_map, dict) or not image_map:
        return text

    # 预清洗：去掉空 key/value
    clean_map: Dict[str, str] = {}
    for k, v in image_map.items():
        kk = str(k or "").strip()
        vv = str(v or "").strip()
        if kk and vv:
            clean_map[kk] = vv

    def repl(m: re.Match) -> str:
        tag = m.group(0)  # "[IMAGE_1]"
        url = clean_map.get(tag, "").strip()
        if not url:
            if strict_missing:
                raise ValueError(f"Missing image url mapping for tag: {tag}")
            return tag  # 宽松：保留原样

        # 用 <...> 包起来，防止 url 里有括号、空格等导致 markdown 解析失败
        alt = f"{alt_prefix}_{tag[len('[IMAGE_'):-1]}"  # 例：IMAGE_1
        if mode == "link":
            out = f"[{tag}](<{url}>)"
        else:
            out = f"![{alt}](<{url}>)"

        # 轻量分隔：避免紧挨着中文/英文导致渲染不稳定
        if ensure_separation:
            return f"\n\n{out}\n\n"
        return out

    return _IMAGE_TAG_RE.sub(repl, text)



# -------------------------
# Context builder
# -------------------------

@dataclass
class PromptContextBuilder:
    """
    将 payload 规范化为“可注入字符串 ctx”，供所有 agent 复用
    - primitives: str/number -> str
    - list/dict: json.dumps 或自定义序列化
    """
    json_indent: int = 2

    def build(self, payload: Mapping[str, Any]) -> Dict[str, str]:
        p = dict(payload or {})
        ctx: Dict[str, str] = {}

        # primitive-ish fields (stringify)
        for k in [
            "title", "idea", "materials",
            "sessionId",
            "qa", "draft", "uncorrectText",
            "chapterId", "chapterTitle",
            "sectionId", "sectionTitle",
            "fullWriteRule", "chapterWriteRule", "sectionWriteRule",
            "fullReviewText", "chapterReviewRule", "sectionReviewRule",
            "industry",
            "helpText",
        ]:
            v = p.get(k, "")
            if v is None:
                ctx[k] = ""
            elif isinstance(v, str):
                ctx[k] = v.strip()
            else:
                ctx[k] = str(v)

        # json-ish fields
        ctx["industryNameList"] = _json(p.get("industryNameList"), indent=self.json_indent)
        ctx["outline"] = _json(p.get("outline"), indent=self.json_indent)
        ctx["review"] = _json(p.get("review"), indent=self.json_indent)
        ctx['reviewList'] = _json(p.get("reviewList"), indent=self.json_indent)
        ctx["sessionList"] = _json(p.get("sessionList"), indent=self.json_indent)
        ctx["sessionQA"] = _serialize_session_list_qa(p.get("sessionList"))
        if not ctx.get("qa"):
            ctx["qa"] = ctx["sessionQA"]


        # special serialized fields
        ctx["textList"] = _serialize_text_list(p.get("textList"))

        # historyTextList: 默认先用“全量串行化”，后面若能精排则覆盖
        history = p.get("historyTextList")
        # print("history:", history)
        ctx_hist = _serialize_history_text_list(history)

        text_list = p.get("textList") or []
        if history is not None and isinstance(text_list, list) and text_list:
            projectId = p.get("projectId", "") or ""

            # print("===========================对历史信息进行rag=========================")
            # print("历史信息：", history)
            # print("=====================================================================")
            try:
                # mapping = mapping_agent.get_cached(projectId) or {}
                print("启动前文rag！")
                neighbors_map = normalize_neighbors_map(MAPPING)

                chapters = (neighbors_map.get("neighbors") or {}).get("chapters") or {}
                if not chapters:
                    print("=====没章节的标识，rag不执行，直接塞全部历史了======")
                    ctx["historyTextList"] = ctx_hist
                    return ctx


                ec = EmbeddingClient(
                    base_url="http://127.0.0.1:30025",
                    model="/home/netzone22/data/LLM/Qwen3-Embedding-8B",
                )

                # ✅ extra_intent 建议塞 helpText/qa/用户改动点（可为空）
                extra_intent = p.get("helpText") or p.get("qa") or ""

                contexts_map = retrieve_history_context_for_textlist(
                    embedding=ec,
                    neighbors_map=neighbors_map,
                    history_text_list=history,
                    title=p.get("title", ""),     # ✅ 可选字段：用 get
                    idea=p.get("idea", ""),       # ✅ 可选字段：用 get
                    text_list=text_list,
                    extra_intent=extra_intent,
                    max_context_chars=8000,
                )  # 返回: {sectionId: "精选上下文字符串"}

                print("=====================rag后的精选的上下文：==========================\n", contexts_map[:20])
                print("==================================================================")

                # ✅ 将 dict 转成“可注入字符串”
                if len(text_list) == 1:
                    sid = str(text_list[0].get("sectionId") or "").strip()
                    ctx_hist = (contexts_map.get(sid) or "").strip()
                else:
                    parts = []
                    for it in text_list:
                        sid = str(it.get("sectionId") or "").strip()
                        st = str(it.get("sectionTitle") or "").strip()
                        c = (contexts_map.get(sid) or "").strip()
                        if not c:
                            continue
                        parts.append(f"[Target {sid} {st}]\n{c}".strip())
                    ctx_hist = "\n\n".join(parts).strip()
                print("压缩后的历史正文：", ctx_hist[:20])
            except Exception:
                # ✅ 任意失败都回退到全量 history 串
                ctx_hist = _serialize_history_text_list(history)
            finally:
                if ec is not None:
                    ec.close()

        ctx["historyTextList"] = ctx_hist


        # runtime injectables default
        ctx.setdefault("draft", "")
        ctx.setdefault("qa", "")

        # ensure everything is str
        for k, v in list(ctx.items()):
            if v is None:
                ctx[k] = ""
            elif not isinstance(v, str):
                ctx[k] = str(v)

        return ctx


# -------------------------
# Renderer
# -------------------------

def render_prompt(
    template: str,
    ctx: Mapping[str, str],
    placeholder_map: Optional[Mapping[str, str]] = None,
    *,
    keep_unknown: bool = True,
) -> str:
    """
    ✅ 只替换 {identifier}：
      - identifier 必须在 placeholder_map 白名单内
      - mapping 指向 ctx 的 key
    ✅ 未识别占位符：
      - keep_unknown=True -> 原样保留（推荐，避免误伤示例/其他用途）
      - keep_unknown=False -> 替换成空串
    """
    t = template or ""
    mapping = dict(placeholder_map or DEFAULT_PLACEHOLDER_MAP)

    def repl(m: re.Match) -> str:
        key = m.group(1)
        field = mapping.get(key)
        if not field:
            return m.group(0) if keep_unknown else ""
        val = ctx.get(field, "")
        return "" if val is None else str(val)

    return _VAR.sub(repl, t)





# -------------------------
# Convenience API
# -------------------------

_default_builder = PromptContextBuilder()

def build_ctx(payload: Mapping[str, Any]) -> Dict[str, str]:
    return _default_builder.build(payload)