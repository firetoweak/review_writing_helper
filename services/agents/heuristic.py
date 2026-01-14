from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from models.llm_interface_async import build_chat_model, build_messages, is_llm_configured
from services.streaming_langgraph import graph_to_ndjson_tokens

MAX_QUESTIONS = 5


class HeuristicState(TypedDict, total=False):
    session_id: str
    section_title: str
    heuristic_prompt: str
    incoming_messages: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    sync_only: bool


class HeuristicAgent:
    """
    仅保留流式输出：
    - ask：伪流式（分片 delta）
    - draft：真流式（token delta）

    boss/PM 路由规则（后端仅做路由，不注入任何辅助提示）：
    - 固定问 5 次（以历史中 assistant 的 type == "question" 计数）
    - 用户回答完第 5 问后（最后一条 user/human 且在第5问之后），下一次输出 status=draft
    """

    def __init__(self) -> None:
        self._graph = self._build_graph()
        self._stream_graph = self._build_stream_graph()

    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._handle_non_stream(payload)

    def message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._handle_non_stream(payload)

    async def stream(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        session_id = str(payload.get("sessionId", "") or "")
        title = payload.get("title", "") or ""
        section_title = payload.get("sectionTitle") or title
        heuristic_prompt = self._build_system_prompt(payload)
        thread_id = f"heuristic:{session_id}"
        messages = payload.get("Messages") or payload.get("messages") or []

        state = self._graph.invoke(
            {
                "session_id": session_id,
                "section_title": section_title,
                "heuristic_prompt": heuristic_prompt,
                "incoming_messages": messages,
                "sync_only": True,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        messages = state.get("messages", []) or []

        if not is_llm_configured():
            q = self._fallback_question(section_title)
            async for line in self._fake_stream_ask_async(
                session_id=session_id,
                section_title=section_title,
                thread_id=thread_id,
                heuristic_prompt=heuristic_prompt,
                question=q,
            ):
                yield line
            return

        if self._ready_to_draft_after_n(messages, n=MAX_QUESTIONS):
            async for line in self._true_stream_draft(
                session_id=session_id,
                section_title=section_title,
                thread_id=thread_id,
                heuristic_prompt=heuristic_prompt,
                messages=messages,
            ):
                yield line
            return

        question = (self._gen_question(messages=messages, heuristic_prompt=heuristic_prompt) or "").strip()
        if not question:
            question = self._fallback_question(section_title)

        async for line in self._fake_stream_ask_async(
            session_id=session_id,
            section_title=section_title,
            thread_id=thread_id,
            heuristic_prompt=heuristic_prompt,
            question=question,
        ):
            yield line
        return

    def _handle_non_stream(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(payload.get("sessionId", "") or "")
        title = payload.get("title", "") or ""
        section_title = payload.get("sectionTitle") or title
        heuristic_prompt = self._build_system_prompt(payload)
        thread_id = f"heuristic:{session_id}"
        messages = payload.get("Messages") or payload.get("messages") or []

        state = self._graph.invoke(
            {
                "session_id": session_id,
                "section_title": section_title,
                "heuristic_prompt": heuristic_prompt,
                "incoming_messages": messages,
                "sync_only": True,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        messages = state.get("messages", []) or []

        if not is_llm_configured():
            question = self._fallback_question(section_title)
            return {
                "sessionId": session_id,
                "status": "ask",
                "assistantMessage": {
                    "messageId": self._next_message_uuid(),
                    "role": "assistant",
                    "type": "question",
                    "content": question,
                },
            }

        if self._ready_to_draft_after_n(messages, n=MAX_QUESTIONS):
            draft = self._gen_draft(messages=messages, heuristic_prompt=heuristic_prompt)
            return {
                "sessionId": session_id,
                "status": "draft",
                "assistantMessage": {
                    "messageId": self._next_message_uuid(),
                    "role": "assistant",
                    "type": "text",
                    "content": draft,
                },
            }

        question = (self._gen_question(messages=messages, heuristic_prompt=heuristic_prompt) or "").strip()
        if not question:
            question = self._fallback_question(section_title)
        return {
            "sessionId": session_id,
            "status": "ask",
            "assistantMessage": {
                "messageId": self._next_message_uuid(),
                "role": "assistant",
                "type": "question",
                "content": question,
            },
        }

    def _build_graph(self):
        graph = StateGraph(HeuristicState)

        def merge(state: HeuristicState) -> Dict[str, Any]:
            history = state.get("messages", []) or []
            incoming = state.get("incoming_messages", []) or []
            merged = self._merge_by_message_id(history, incoming)
            return {"messages": merged, "incoming_messages": []}

        graph.add_node("merge", merge)
        graph.set_entry_point("merge")
        graph.set_finish_point("merge")
        return graph.compile(checkpointer=MemorySaver())

    def _gen_question(self, messages: List[Dict[str, Any]], heuristic_prompt: str) -> str:
        model = build_chat_model(streaming=False)
        system_prompt = (heuristic_prompt or "").rstrip()
        lc_messages = build_messages(system_prompt=system_prompt, user_text="", messages=messages)
        res = model.invoke(lc_messages)
        return (getattr(res, "content", "") or "").strip()

    def _gen_draft(self, messages: List[Dict[str, Any]], heuristic_prompt: str) -> str:
        model = build_chat_model(streaming=False)
        system_prompt = (heuristic_prompt or "").rstrip()
        lc_messages = build_messages(system_prompt=system_prompt, user_text="", messages=messages)
        res = model.invoke(lc_messages)
        return (getattr(res, "content", "") or "").strip()

    def _build_stream_graph(self):
        model = build_chat_model(streaming=True)
        graph = StateGraph(dict)

        async def call_model(state: dict, config: RunnableConfig) -> dict:
            resp = await model.ainvoke(state["messages"], config=config)
            return {"response": resp}

        graph.add_node("call_model", call_model)
        graph.set_entry_point("call_model")
        graph.set_finish_point("call_model")
        return graph.compile()

    async def _true_stream_draft(
        self,
        session_id: str,
        section_title: str,
        thread_id: str,
        heuristic_prompt: str,
        messages: List[Dict[str, Any]],
    ) -> AsyncIterator[str]:
        msg_id = self._next_message_uuid()

        yield self._ndjson(
            {
                "type": "message.start",
                "sessionId": session_id,
                "status": "draft",
                "assistantMessage": {"messageId": msg_id, "role": "assistant", "type": "text", "content": ""},
            }
        )

        system_prompt = (heuristic_prompt or "").rstrip()
        lc_messages = build_messages(system_prompt=system_prompt, user_text="", messages=messages)

        content_acc = ""
        async for line in graph_to_ndjson_tokens(self._stream_graph, {"messages": lc_messages}):
            obj = self._safe_json(line)
            if not obj:
                continue
            if obj.get("type") == "token":
                tok = obj.get("text") or ""
                if tok:
                    content_acc += tok
                    yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": tok})
            elif obj.get("type") == "done":
                break

        yield self._ndjson({"type": "message.end", "messageId": msg_id, "done": True, "finishReason": "stop"})
        yield self._ndjson({"type": "done"})

        self._graph.invoke(
            {
                "session_id": session_id,
                "section_title": section_title,
                "heuristic_prompt": heuristic_prompt,
                "incoming_messages": [{"messageId": msg_id, "role": "assistant", "type": "text", "content": content_acc}],
                "sync_only": True,
            },
            config={"configurable": {"thread_id": thread_id}},
        )

    async def _fake_stream_ask_async(
        self,
        session_id: str,
        section_title: str,
        thread_id: str,
        heuristic_prompt: str,
        question: str,
    ) -> AsyncIterator[str]:
        msg_id = self._next_message_uuid()

        yield self._ndjson(
            {
                "type": "message.start",
                "sessionId": session_id,
                "status": "ask",
                "assistantMessage": {"messageId": msg_id, "role": "assistant", "type": "question", "content": ""},
            }
        )

        for part in self._split_chunks(question):
            yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": part})
            await asyncio.sleep(0.01)

        yield self._ndjson({"type": "message.end", "messageId": msg_id, "done": True, "finishReason": "stop"})
        yield self._ndjson({"type": "done"})

        self._graph.invoke(
            {
                "session_id": session_id,
                "section_title": section_title,
                "heuristic_prompt": heuristic_prompt,
                "incoming_messages": [{"messageId": msg_id, "role": "assistant", "type": "question", "content": question}],
                "sync_only": True,
            },
            config={"configurable": {"thread_id": thread_id}},
        )

    def _assistant_question_indices(self, messages: List[Dict[str, Any]]) -> List[int]:
        idxs: List[int] = []
        for i, m in enumerate(messages or []):
            if m.get("role") == "assistant" and m.get("type") == "question":
                idxs.append(i)
        return idxs

    def _ready_to_draft_after_n(self, messages: List[Dict[str, Any]], n: int) -> bool:
        msgs = messages or []
        q_idxs = self._assistant_question_indices(msgs)
        if len(q_idxs) < n:
            return False

        nth_q_idx = q_idxs[n - 1]

        if not msgs:
            return False
        last_role = (msgs[-1].get("role") or "").lower()
        if last_role not in ("user", "human"):
            return False

        last_user_idx = None
        for i in range(len(msgs) - 1, -1, -1):
            r = (msgs[i].get("role") or "").lower()
            if r in ("user", "human"):
                last_user_idx = i
                break

        return (last_user_idx is not None) and (last_user_idx > nth_q_idx)

    def _merge_by_message_id(self, history: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        merged: List[Dict[str, Any]] = []

        def add(m: Dict[str, Any]):
            mid = m.get("messageId") or json.dumps(m, ensure_ascii=False, sort_keys=True)
            if mid in seen:
                return
            seen.add(mid)
            merged.append(m)

        for m in history:
            add(m)
        for m in incoming:
            add(m)
        return merged

    def _fallback_question(self, section_title: str) -> str:
        return f"信息还不够。我需要你补充一个关键点：在[{section_title}]里，你最想强调的市场变化/竞争趋势是哪一条？"

    def _build_system_prompt(self, payload: Dict[str, Any]) -> str:
        prompt_data = payload.get("prompt") or {}
        if not isinstance(prompt_data, dict):
            prompt_data = {}
        heuristic_prompt = (
            prompt_data.get("heuristicWritingPrompt")
            or prompt_data.get("heuristicCorrectPrompt")
            or ""
        )
        section_write_rule = payload.get("sectionWriteRule") or ""
        section_review_rule = payload.get("sectionReviewRule") or ""
        industry = payload.get("industry") or ""
        title = payload.get("title") or ""
        section_title = payload.get("sectionTitle") or ""
        text_context = self._format_text_list(payload.get("textList") or [])
        history_context = self._format_history_text(payload.get("historyTextList") or [])
        parts = [
            heuristic_prompt,
            f"标题：{title}" if title else "",
            f"小节标题：{section_title}" if section_title else "",
            f"行业：{industry}" if industry else "",
            f"写作规则：{section_write_rule}" if section_write_rule else "",
            f"审阅规则：{section_review_rule}" if section_review_rule else "",
            f"已有内容：{text_context}" if text_context else "",
            f"历史内容：{history_context}" if history_context else "",
        ]
        return "\n".join(part for part in parts if part).strip()

    def _format_text_list(self, text_list: List[Dict[str, Any]]) -> str:
        parts = []
        for item in text_list:
            title = item.get("sectionTitle") or ""
            text = item.get("text") or ""
            if title or text:
                parts.append(f"{title}: {text}".strip())
        return "\n".join(parts)

    def _format_history_text(self, history_list: List[Dict[str, Any]]) -> str:
        parts = []
        for chapter in history_list:
            chapter_title = chapter.get("chapterTitle") or ""
            for child in chapter.get("children", []):
                section_title = child.get("sectionTitle") or ""
                text = child.get("text") or ""
                label = " / ".join(filter(None, [chapter_title, section_title]))
                if label or text:
                    parts.append(f"{label}: {text}".strip())
        return "\n".join(parts)

    def _split_chunks(self, text: str, chunk_size: int = 24) -> List[str]:
        t = (text or "").strip()
        if not t:
            return []
        seps = ["？", "?", "。", "；", ";", "\n"]
        out: List[str] = []
        buf = t
        while buf:
            cut = None
            for sep in seps:
                idx = buf.find(sep)
                if 0 <= idx < 40:
                    cut = idx + 1
                    break
            if cut is None:
                cut = min(chunk_size, len(buf))
            out.append(buf[:cut])
            buf = buf[cut:]
        return [x for x in out if x]

    def _safe_json(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(line)
        except Exception:
            return None

    def _ndjson(self, obj: Dict[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    def _next_message_uuid(self, prefix: str = "m_ai") -> str:
        return f"{prefix}_{uuid.uuid4().hex}"
