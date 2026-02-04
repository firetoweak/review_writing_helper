from __future__ import annotations

import asyncio
import json
import uuid
from enum import Enum
import re
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph

from models.llm_interface_async import (
    build_chat_model,
    build_messages,
    extract_image_map_from_text_list,
    is_vlm_configured,
)
from tools.prompt_templating import build_ctx, render_prompt, replace_image_tags_with_markdown, DEFAULT_PLACEHOLDER_MAP
from tools.verify import SmartVerifierCore, my_check_batch
import asyncio


MAX_QUESTIONS = 5

# 固定通用 SystemMessage：只放强约束/抗注入/协议要求
COMMON_SYSTEM_PROMPT = """Given the following conversation, relevant context, and a follow up question, reply with an answer to the current question the user is asking. Return only your response to the question given the above information following the users instructions as needed.""".strip()



class Phase(str, Enum):
    ASK = "ask"
    DRAFT = "draft"
    ENDED = "ended"


class HeuristicState(TypedDict, total=False):
    # minimal identity & lifecycle
    sessionId: str
    projectId: str
    ended: bool

    ctx_base: Dict[str, str]   # ✅ 首轮 build_ctx 的快照（字符串化后的上下文）

    # rendered, persisted once (first turn)
    context_text: str

    # optional persisted correct template (so draft->correct always works)
    correct_tpl: str

    # multimodal mapping
    image_map: Dict[str, str]

    # conversation history
    messages: List[Dict[str, Any]]
    incoming_messages: List[Dict[str, Any]]


class HeuristicAgent:
    """
    ✅ 目标：5 轮 ASK -> DRAFT（hidden draft + correct streaming）-> ended

    输入：payload 含 prompt / materials / rules / textList 等（首轮用于渲染 context_text）。
    后续轮：只需 messages 增量（用户回复）即可。

    组装消息顺序：
      SystemMessage(COMMON_SYSTEM_PROMPT)
      HumanMessage(context_text + image_map)   # 隐藏上下文，多模态
      history messages                          # Q/A
      HumanMessage(trigger)                     # 短触发
    """

    def __init__(
        self,
        *,
        kb: Optional[Any] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> None:
        g = StateGraph(HeuristicState)
        g.add_node("merge", self._merge_node)
        g.set_entry_point("merge")
        g.set_finish_point("merge")
        self.graph = g.compile(checkpointer=checkpointer or MemorySaver())
        self._ph_map = dict(DEFAULT_PLACEHOLDER_MAP)
        self._kb_client = kb


    # =========================
    # Public
    # =========================
    async def stream(self, payload: Dict[str, Any]) -> AsyncIterator[str]:

        session_id = str(payload.get("sessionId") or payload.get("sectionId") or "default").strip()
        project_id = str(payload.get("projectId") or "").strip()

        # thread id: prefer projectId to isolate sessions per project
        key = project_id or session_id
        cfg = {"configurable": {"thread_id": f"heuristic:{key}:{session_id}"}}



        snap = await self.graph.aget_state(config=cfg)
        current = (snap.values if snap else {}) or {}

        print("session_id", payload.get("sessionId"), "project_id", payload.get("projectId"), "thread_id", cfg["configurable"]["thread_id"])
        print("user messages:", payload.get("messages", [])[:200])
        print("before invoke state keys:", list(current.keys()))
        print("textList:", payload.get("textList", [])[:80])

        if current.get("ended"):
            async for line in self._emit_static(
                session_id=session_id,
                status=Phase.DRAFT.value,
                msg_type="text",
                content="该 session 已结束（5问->出稿->修正已完成）。请创建新的 sessionId 重新开始。",
                is_done=True,
            ):
                yield line
            return

        # update state (first turn will materialize context_text & correct_tpl)
        update = self._prepare_update_payload(session_id=session_id, project_id=project_id, payload=payload, current=current)
        await self.graph.ainvoke(update, config=cfg)

        snap2 = await self.graph.aget_state(config=cfg)
        state = (snap2.values if snap2 else {}) or {}

        
        print("after invoke state keys:", list(state.keys()))
        print("context_text len:", len(state.get("context_text") or "NULL"))
        print("messages n:", len(state.get("messages") or []))
        print("messages:", state.get("messages", []))
        phase = self._decide_phase(state.get("messages", []) or [])
        print("DECIDED phase=", phase.value, "vlm=", is_vlm_configured(), "messages_n=", len(state.get("messages") or []))
        if not is_vlm_configured():
            async for line in self._emit_static(
                session_id=session_id,
                status=phase.value if phase != Phase.ENDED else Phase.DRAFT.value,
                msg_type="text" if phase == Phase.DRAFT else "question",
                content=self._fallback_content(phase.value),
                is_done=True,
            ):
                yield line
            return

        if phase == Phase.ASK:
            print("GO ASK")
            async for line in self._handle_ask(session_id=session_id, state=state, cfg=cfg):
                yield line
            return

        if phase == Phase.DRAFT:
            print("GO DRAFT")
            async for line in self._handle_draft(session_id=session_id, state=state, cfg=cfg):
                yield line
            return
        print("GO ???")
        # ENDED should be short-circuited at top
        async for line in self._emit_static(session_id=session_id, status=Phase.DRAFT.value, msg_type="text", content="状态异常。", is_done=True):
            yield line

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """聚合 stream，为非流式调用提供一个简单返回（前端如果只吃 run 可用）。"""
        meta: Optional[Dict[str, Any]] = None
        content = ""

        async for line in self.stream(payload):
            obj = self._safe_json(line)
            if not obj:
                continue
            t = obj.get("type")
            if t == "message.start":
                meta = obj
            elif t == "message.delta":
                content += (obj.get("delta") or "")
            elif t == "message.end":
                break

        session_id = (meta.get("sessionId") if meta else None) or str(payload.get("sessionId") or "").strip()
        status = (meta.get("status") if meta else None) or Phase.ASK.value
        assistant = (meta.get("assistantMessage") if meta else None) or {"messageId": "", "role": "assistant", "type": "text", "content": ""}
        assistant = dict(assistant)
        assistant["content"] = content
        return {"sessionId": session_id, "status": status, "assistantMessage": assistant}

    # =========================
    # State merge
    # =========================
    def _merge_node(self, state: HeuristicState) -> Dict[str, Any]:
        history = state.get("messages", []) or []
        incoming = state.get("incoming_messages", []) or []
        if not incoming:
            return {"incoming_messages": []}

        merged = self._merge_by_message_id(history, incoming)
        return {"messages": merged, "incoming_messages": []}

    def _prepare_update_payload(
        self,
        *,
        session_id: str,
        project_id: str,
        payload: Dict[str, Any],
        current: Dict[str, Any],
    ) -> Dict[str, Any]:
        
        out: Dict[str, Any] = {
            "sessionId": session_id,
            "incoming_messages": payload.get("messages", []) or [],
        }
        if project_id:
            out["projectId"] = project_id

        # image map: merge incrementally
        merged_map: Dict[str, str] = dict(current.get("image_map") or {})

        # top-level image map
        for k in ("image_map", "imageMap", "image_url", "imageUrl"):
            m = payload.get(k)
            if isinstance(m, dict):
                for kk, vv in m.items():
                    if kk and vv:
                        merged_map[str(kk)] = str(vv)

        # from textList items: {"image_url": {"[IMAGE_1]": "..."}}
        tl = payload.get("textList")
        if isinstance(tl, list):
            merged_map.update(extract_image_map_from_text_list(tl))

        if merged_map:
            out["image_map"] = merged_map

        # ✅ materialize context_text only once
        if not current.get("context_text"):
            prompt_obj = payload.get("prompt") or {}
            writing_tpl = (prompt_obj.get("heuristicWritingPrompt") or "").rstrip()
            correct_tpl = (prompt_obj.get("heuristicCorrectPrompt") or "").rstrip()
            # print("payload=", json.dumps(payload, ensure_ascii=False))
            ctx = build_ctx(payload)

            # 添加知识库功能
            context_text = render_prompt(writing_tpl, ctx, self._ph_map, keep_unknown=True)


            out["context_text"] = context_text
            # out["correct_tpl"] = correct_tpl

            out["ctx_base"] = ctx   # ✅ 关键：持久化

            # print("writing_tpl_len=", len(writing_tpl))
            # print("=======writing_tpl====", writing_tpl)
            print("context_text_len=", len(context_text), "context_text_head=", repr(context_text[:80]))
            # 可选：看看模板里是否仍有未替换占位符（渲染失败的信号）
            import re
            left = re.findall(r"\{[a-zA-Z0-9_]+\}", context_text)
            print("context_left_placeholders=", left[:10])

        else:
            # allow updating correct_tpl explicitly if provided
            prompt_obj = payload.get("prompt") or {}
            if "heuristicCorrectPrompt" in prompt_obj and prompt_obj.get("heuristicCorrectPrompt") is not None:
                out["correct_tpl"] = (prompt_obj.get("heuristicCorrectPrompt") or "").rstrip()

        return out

    # =========================
    # Phase logic
    # =========================
    def _decide_phase(self, messages: List[Dict[str, Any]]) -> Phase:
        if self._ready_to_draft_after_n(messages, n=MAX_QUESTIONS):
            return Phase.DRAFT
        return Phase.ASK

    def _assistant_question_indices(self, messages: List[Dict[str, Any]]) -> List[int]:
        idxs: List[int] = []
        for i, m in enumerate(messages or []):
            role = (m.get("role") or "").strip().lower()
            if role == "assistant" and m.get("type") == "question":
                idxs.append(i)
        return idxs

    def _ready_to_draft_after_n(self, messages: List[Dict[str, Any]], n: int) -> bool:
        msgs = messages or []
        q_idxs = self._assistant_question_indices(msgs)
        if len(q_idxs) < n:
            return False

        nth_q_idx = q_idxs[n - 1]

        last_user_idx: Optional[int] = None
        for i in range(len(msgs) - 1, -1, -1):
            r = (msgs[i].get("role") or "").strip().lower()
            if r in ("user", "human"):
                last_user_idx = i
                break

        return (last_user_idx is not None) and (last_user_idx > nth_q_idx)

    def _serialize_last_qa(self, messages: List[Dict[str, Any]], n: int = 5) -> str:
        msgs = messages or []
        q_idxs: List[int] = []
        for i, m in enumerate(msgs):
            if (m.get("role") == "assistant") and (m.get("type") == "question"):
                q_idxs.append(i)

        q_idxs = q_idxs[-n:]
        out: List[str] = []
        for k, qi in enumerate(q_idxs, start=1):
            q = (msgs[qi].get("content") or "").strip()
            a = ""
            for j in range(qi + 1, len(msgs)):
                r = (msgs[j].get("role") or "").strip().lower()
                if r in ("user", "human"):
                    a = (msgs[j].get("content") or "").strip()
                    break
            out.append(f"Q{k}: {q}")
            out.append(f"A{k}: {a}")
        return "\n".join(out).strip()

    # =========================
    # Handlers
    # =========================
    async def _handle_ask(self, *, session_id: str, state: Dict[str, Any], cfg: Dict[str, Any]) -> AsyncIterator[str]:
        context_text = (state.get("context_text") or "").rstrip()
        image_map = state.get("image_map") or {}
        history = state.get("messages", []) or []

        # ✅ 计算问号：已问 question 的数量 + 1
        q_no = len(self._assistant_question_indices(history)) + 1
        q_no = min(q_no, MAX_QUESTIONS)  # 防御：最多显示到 5
        prefix = f"（第{q_no}/{MAX_QUESTIONS}问）"  # 你也可以加个空格或换行：+ "\n"

        trigger = "请基于以上资料与历史问答，输出下一条追问。只输出问题正文。"

        lc_messages = build_messages(
            system_prompt=COMMON_SYSTEM_PROMPT,
            context_text=context_text,
            context_image_map=image_map,
            messages=history,
            user_text="【系统提示，本次任务是追问】",
            image_map=None,
        )

        async for line in self._stream_llm_with_lc_messages(
            session_id=session_id,
            status=Phase.ASK.value,
            msg_type="question",
            lc_messages=lc_messages,
            cfg=cfg,
            prelude=prefix,
        ):
            yield line

    async def _handle_draft(self, *, session_id: str, state: Dict[str, Any], cfg: Dict[str, Any]) -> AsyncIterator[str]:
        print("ENTER _handle_draft")
        context_text = (state.get("context_text") or "").rstrip()
        image_map = state.get("image_map") or {}
        history = state.get("messages", []) or []
        correct_tpl = (state.get("correct_tpl") or "").rstrip()

        print("draft: context_len=", len(context_text), "history_n=", len(history))
        print("draft: correct_tpl_len=", len(correct_tpl))


        # A) hidden draft (non-stream)
        lc_msgs_draft = build_messages(
            system_prompt=COMMON_SYSTEM_PROMPT,
            context_text=context_text,
            context_image_map=image_map,
            messages=history,
            user_text="【系统提示，本次任务是生成正文】",
            image_map=None,
        )
        draft_text = await self._generate_silent_from_lc_messages(lc_msgs_draft)

        print("draft_text_len=", len(draft_text), "draft_text_head=", repr(draft_text[:80]))


        # B) render correct prompt with {draft}{qa}
        qa = self._serialize_last_qa(history, n=MAX_QUESTIONS)

        base = dict(state.get("ctx_base") or {})
        historyText = base.get("historyTextList", "")
        materials = base.get("materials", "")
        base.update({"draft": draft_text, "qa": qa})
        textList = base.get("textList", "")
        evidence_text = f"交互问答：{qa}\n\n相关素材：{materials}\n\n前文：{historyText}"
        # correct_prompt = render_prompt(correct_tpl, base, self._ph_map, keep_unknown=True)
        correct_model = build_chat_model(streaming=False)
        core = SmartVerifierCore()
        correct_text =  await core.verify_async(
            original=textList,
            merged=draft_text,  
            evidence=evidence_text,
            model=correct_model,
            check_batch_fn=my_check_batch,
            batch_size=10,
            max_inflight_batches=4,
        )
        correct_text = replace_image_tags_with_markdown(text=correct_text, image_map=image_map)

        # print("=================最终校验结果===========================")
        # print("最终校验结果：", correct_text[80:])
        # print("======================================================")

        n = 0
        try:
            async for line in self.stream_markdown_as_message(
                session_id=session_id,
                status="draft",
                msg_type="assistant",
                correct_text=correct_text,
                cfg=cfg,
                chunk_size=2048,
            ):
                n += 1
                try:
                    json.loads(line)
                except Exception:
                    print("[DRAFT] INVALID_JSON_LINE head=", repr(line[:200]), flush=True)
                    raise

                if n <= 3:
                    print("[DRAFT] line head=", repr(line[:200]), flush=True)

                yield line
                
        except Exception as e:
            import traceback; 
            print("[AUDIT_markdown] stream_markdown_as_message FAILED:", repr(e), flush=True)
            traceback.print_exc()


        # await asyncio.sleep(0)

        await self.graph.ainvoke({"ended": True, "incoming_messages": []}, config=cfg)

    def _chunk_text(self, s: str, *, chunk_size: int = 2048):
        # 按字符分块即可；如果你担心拆坏标签，可以改成按 "\n" / 句子边界分块
        for i in range(0, len(s), chunk_size):
            yield s[i:i + chunk_size]

    async def stream_markdown_as_message(
        self,
        *,
        session_id: str,
        status: str,
        msg_type: str,
        correct_text: str,          # markdown ：原样流式输出
        cfg: Dict[str, Any],
        chunk_size: int = 2048,
    ) -> AsyncIterator[str]:
        msg_id = f"m_ai_{uuid.uuid4().hex[:10]}"
        content_acc = ""

        # 1) start（与 _stream_llm_with_lc_messages 对齐）
        yield self._ndjson(
            {
                "type": "message.start",
                "sessionId": session_id,
                "status": status,
                "assistantMessage": {"messageId": msg_id, "role": "assistant", "type": msg_type, "content": ""},
            }
        )

        completed_normally = False
        sent = 0
        text = correct_text or ""
        total = len(text)

        try:
            print(
                f"CALL markdown stream start status={status} msg_type={msg_type} total_len={total} chunk_size={chunk_size}",
                flush=True,
            )

            if total == 0:
                print("[markdown_STREAM] empty correct_text", flush=True)

            i = 0
            while i < total:
                chunk = text[i : i + chunk_size]
                i += chunk_size
                if not chunk:
                    continue

                content_acc += chunk
                sent += 1

                yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": chunk})

                # 给 event loop 一点喘息，避免长串输出把别的任务饿死
                if sent % 10 == 0:
                    await asyncio.sleep(0)

            completed_normally = True
            print(
                f"CALL markdown stream end status={status} msg_type={msg_type} completed={completed_normally} "
                f"content_len={len(content_acc)} sent_chunks={sent}",
                flush=True,
            )

        except asyncio.CancelledError:
            # 典型：前端断开连接/中途取消请求
            print("[markdown_STREAM] cancelled (client disconnected?)", flush=True)
            raise
        except Exception as e:
            import traceback

            print("[markdown_STREAM] ERROR:", repr(e), flush=True)
            traceback.print_exc()
            raise
        finally:
            # ✅ 与 _stream_llm_with_lc_messages 对齐：写入 graph（持久化/入库）
            if content_acc:
                await self.graph.ainvoke(
                    {
                        "incoming_messages": [
                            {"messageId": msg_id, "role": "assistant", "type": msg_type, "content": content_acc}
                        ]
                    },
                    config=cfg,
                )

            # 3) end（与 _stream_llm_with_lc_messages 对齐：done + finishReason）
            yield self._ndjson(
                {
                    "type": "message.end",
                    "messageId": msg_id,
                    "done": True,
                    "finishReason": "stop" if completed_normally else "error",
                }
            )
            yield self._ndjson({"type": "done"})

    # =========================
    # LLM helpers
    # =========================
    async def _stream_llm_with_lc_messages(
        self,
        *,
        session_id: str,
        status: str,
        msg_type: str,
        lc_messages: List[Any],
        cfg: Dict[str, Any],
        model_name: Optional[str] = None,
        prelude: str = "",  # ✅ 新增：开头先输出的一段文字
    ) -> AsyncIterator[str]:
        model = build_chat_model(streaming=True, multimodal=True, model_name=model_name)

        msg_id = f"m_ai_{uuid.uuid4().hex[:10]}"
        content_acc = ""

        yield self._ndjson(
            {
                "type": "message.start",
                "sessionId": session_id,
                "status": status,
                "assistantMessage": {"messageId": msg_id, "role": "assistant", "type": msg_type, "content": ""},
            }
        )

        # ✅ 在模型 token 前先输出前缀
        # if prelude:
        #     yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": prelude})


        completed_normally = False
        try:
            print(f"CALL stream start status={status} msg_type={msg_type} lc_messages_n={len(lc_messages)} model_name={model_name}")

            async for event in model.astream_events(lc_messages, version="v1"):
                if event.get("event") != "on_chat_model_stream":
                    continue

                chunk = event.get("data", {}).get("chunk")
                text = getattr(chunk, "content", "") if chunk is not None else ""
                if not text:
                    continue   

                content_acc += text
                yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": text})

            completed_normally = True
            print(f"CALL stream end status={status} msg_type={msg_type} completed={completed_normally} content_len={len(content_acc)}")
        finally:
            if content_acc:
                await self.graph.ainvoke(
                    {"incoming_messages": [{"messageId": msg_id, "role": "assistant", "type": msg_type, "content": content_acc}]},
                    config=cfg,
                )

            if prelude:
                yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": prelude})

            yield self._ndjson(
                {
                    "type": "message.end",
                    "messageId": msg_id,
                    "done": True,
                    "finishReason": "stop" if completed_normally else "error",
                }
            )
            yield self._ndjson({"type": "done"})

    async def _generate_silent_from_lc_messages(self, lc_messages: List[Any]) -> str:
        model = build_chat_model(streaming=False, multimodal=True)
        print("CALL draft ainvoke start")
        resp = await model.ainvoke(lc_messages)
        print("CALL draft ainvoke end, resp_len=", len(str(getattr(resp,"content","") or "")))
        return str(getattr(resp, "content", resp) or "").strip()

    # =========================
    # Utilities
    # =========================
    def _fallback_content(self, mode: str) -> str:
        if mode == Phase.DRAFT.value:
            return "（未配置模型）无法生成正文。"
        return "（未配置模型）请补充关键信息：你最想强调的关键约束/目标是什么？"

    def _merge_by_message_id(self, history: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        index: Dict[str, int] = {}

        def mid_of(m: Dict[str, Any]) -> str:
            return m.get("messageId") or json.dumps(m, ensure_ascii=False, sort_keys=True)

        def prefer_new(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
            old_c = old.get("content") or ""
            new_c = new.get("content") or ""
            out = dict(old)
            for k, v in (new or {}).items():
                if v is not None:
                    out[k] = v
            out["content"] = new_c if len(new_c) >= len(old_c) else old_c
            return out

        for m in history or []:
            mid = mid_of(m)
            index[mid] = len(merged)
            merged.append(m)

        for m in incoming or []:
            mid = mid_of(m)
            if mid in index:
                i = index[mid]
                merged[i] = prefer_new(merged[i], m)
            else:
                index[mid] = len(merged)
                merged.append(m)

        return merged

    async def _emit_static(
        self,
        *,
        session_id: str,
        status: str,
        msg_type: str,
        content: str,
        is_done: bool = False,
    ) -> AsyncIterator[str]:
        msg_id = f"sys_{uuid.uuid4().hex[:8]}"

        yield self._ndjson(
            {
                "type": "message.start",
                "sessionId": session_id,
                "status": status,
                "assistantMessage": {"messageId": msg_id, "role": "assistant", "type": msg_type, "content": ""},
            }
        )

        chunk = 16
        for i in range(0, len(content), chunk):
            yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": content[i : i + chunk]})
            await asyncio.sleep(0.005)

        yield self._ndjson({"type": "message.end", "messageId": msg_id, "done": True, "finishReason": "stop"})
        if is_done:
            yield self._ndjson({"type": "done"})

    def _safe_json(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(line)
        except Exception:
            return None

    def _ndjson(self, obj: Dict[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"
