from __future__ import annotations

import asyncio
import itertools
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
    node_id: str
    title: str
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
        self._counter = itertools.count(1)
        self._graph = self._build_graph()
        self._stream_graph = self._build_stream_graph()

    async def stream(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        node_id = payload.get("nodeId", "") or ""
        title = payload.get("title", "") or ""
        heuristic_prompt = payload.get("heuristicPrompt") or ""
        session_id = payload.get("sessionId") or node_id
        thread_id = f"heuristic:{session_id}"
        task = payload.get("task", "heuristicWriting")

        state = self._graph.invoke(
            {
                "node_id": node_id,
                "title": title,
                "heuristic_prompt": heuristic_prompt,
                "incoming_messages": payload.get("messages", []) or [],
                "sync_only": True,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        messages = state.get("messages", []) or []

        if not is_llm_configured():
            q = self._fallback_question(title)
            async for line in self._fake_stream_ask_async(
                session_id=session_id,
                node_id=node_id,
                title=title,
                task=task,
                thread_id=thread_id,
                heuristic_prompt=heuristic_prompt,
                question=q,
            ):
                yield line
            return

        if self._ready_to_draft_after_n(messages, n=MAX_QUESTIONS):
            async for line in self._true_stream_draft(
                session_id=session_id,
                node_id=node_id,
                title=title,
                task=task,
                thread_id=thread_id,
                heuristic_prompt=heuristic_prompt,
                messages=messages,
            ):
                yield line
            return

        question = (self._gen_question(messages=messages, heuristic_prompt=heuristic_prompt) or "").strip()
        if not question:
            question = self._fallback_question(title)

        async for line in self._fake_stream_ask_async(
            session_id=session_id,
            node_id=node_id,
            title=title,
            task=task,
            thread_id=thread_id,
            heuristic_prompt=heuristic_prompt,
            question=question,
        ):
            yield line
        return

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
        node_id: str,
        title: str,
        task: str,
        thread_id: str,
        heuristic_prompt: str,
        messages: List[Dict[str, Any]],
    ) -> AsyncIterator[str]:
        msg_id = self._next_message_uuid()

        yield self._ndjson(
            {
                "type": "message.start",
                "sessionId": session_id,
                "nodeId": node_id,
                "title": title,
                "task": task,
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
                "node_id": node_id,
                "title": title,
                "heuristic_prompt": heuristic_prompt,
                "incoming_messages": [{"messageId": msg_id, "role": "assistant", "type": "text", "content": content_acc}],
                "sync_only": True,
            },
            config={"configurable": {"thread_id": thread_id}},
        )

    async def _fake_stream_ask_async(
        self,
        session_id: str,
        node_id: str,
        title: str,
        task: str,
        thread_id: str,
        heuristic_prompt: str,
        question: str,
    ) -> AsyncIterator[str]:
        msg_id = self._next_message_uuid()

        yield self._ndjson(
            {
                "type": "message.start",
                "sessionId": session_id,
                "nodeId": node_id,
                "title": title,
                "task": task,
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
                "node_id": node_id,
                "title": title,
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

    def _fallback_question(self, title: str) -> str:
        return f"信息还不够。我需要你补充一个关键点：在[{title}]里，你最想强调的市场变化/竞争趋势是哪一条？"

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
