from __future__ import annotations

import itertools
import json
import operator
import re
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict, Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig

from models.llm_interface_async import build_chat_model, build_messages, is_llm_configured
from services.streaming_langgraph import graph_to_ndjson_tokens


POST_ANSWER_CONSTRAINT = "[后续追问必须紧扣最终写作目标]"

CONTROL_HEADER_INSTRUCTION = """
【输出格式强约束（必须遵守）】
- 第 1 行只输出严格 JSON（不要代码块/不要多余文字）：
  {"status":"ask","type":"question"} 或 {"status":"draft","type":"text"}
- 第 2 行开始输出 content：
  - 若 status=ask：只输出 1 个关键问题
  - 若 status=draft：输出本节完整正文（不加额外说明）
"""

DEFAULT_FALLBACK_PROMPT = "你是写作教练，请用循序追问方式提出一个清晰问题。"


class HeuristicState(TypedDict, total=False):
    node_id: str
    title: str
    heuristic_prompt: str
    incoming_messages: List[Dict[str, Any]]
    messages: Annotated[List[Dict[str, Any]], operator.add]
    last_response: Dict[str, Any]


class HeuristicAgent:
    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._graph = self._build_graph()

    def start(self, payload: Dict) -> Dict:
        node_id = payload.get("nodeId", "")
        title = payload.get("title", "")
        heuristic_prompt = payload.get("heuristicPrompt") or ""
        session_id = payload.get("sessionId") or node_id
        thread_id = f"heuristic:{session_id}"

        state = self._graph.invoke(
            {
                "node_id": node_id,
                "title": title,
                "heuristic_prompt": heuristic_prompt,
                "incoming_messages": payload.get("messages", []) or [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        response = state["last_response"]
        return {
            "nodeId": node_id,
            "title": title,
            "task": payload.get("task", "heuristicWriting"),
            **response,
        }

    def message(self, payload: Dict) -> Dict:
        node_id = payload.get("nodeId", "")
        title = payload.get("title", "")
        heuristic_prompt = payload.get("heuristicPrompt") or ""
        session_id = payload.get("sessionId") or node_id
        thread_id = f"heuristic:{session_id}"

        state = self._graph.invoke(
            {
                "node_id": node_id,
                "title": title,
                "heuristic_prompt": heuristic_prompt,
                "incoming_messages": payload.get("messages", []) or [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        response = state["last_response"]
        return {
            "nodeId": node_id,
            "title": title,
            "task": payload.get("task", "heuristicWriting"),
            **response,
        }

    async def stream(self, payload: Dict) -> AsyncIterator[str]:
        node_id = payload.get("nodeId", "")
        title = payload.get("title", "")
        base_prompt = payload.get("heuristicPrompt") or ""
        messages = payload.get("messages", []) or []

        if not is_llm_configured():
            msg_id = self._next_message_id()
            content = self._heuristic_question(1, title)
            yield self._ndjson(
                {
                    "type": "message.start",
                    "nodeId": node_id,
                    "title": title,
                    "task": payload.get("task", "heuristicWriting"),
                    "status": "ask",
                    "assistantMessage": {
                        "messageId": msg_id,
                        "role": "assistant",
                        "type": "question",
                        "content": "",
                    },
                }
            )
            yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": content})
            yield self._ndjson({"type": "message.end", "messageId": msg_id, "done": True, "finishReason": "stop"})
            yield self._ndjson({"type": "done"})
            return

        user_answers = self._count_user_answers(messages)
        if user_answers >= 5 and not self._awaiting_confirmation(messages):
            msg_id = self._next_message_id()
            content = "信息已经足够。请确认是否开始撰写本节正文？请回复“确认”或说明需要补充的点。"
            yield self._ndjson(
                {
                    "type": "message.start",
                    "nodeId": node_id,
                    "title": title,
                    "task": payload.get("task", "heuristicWriting"),
                    "status": "ask",
                    "assistantMessage": {
                        "messageId": msg_id,
                        "role": "assistant",
                        "type": "question",
                        "content": "",
                    },
                }
            )
            yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": content})
            yield self._ndjson({"type": "message.end", "messageId": msg_id, "done": True, "finishReason": "stop"})
            yield self._ndjson({"type": "done"})
            return

        force_mode = {"status": "draft", "type": "text"} if self._should_force_draft(messages) else None
        system_prompt = self._effective_system_prompt(base_prompt, force_mode=force_mode)
        user_text = f"当前小节：{title}\n请严格按输出格式强约束输出。"
        lc_messages = build_messages(system_prompt=system_prompt, user_text=user_text, messages=messages)
        graph = self._build_stream_graph()

        msg_id = self._next_message_id()
        header_parsed = False
        header_buf = ""
        status = "ask"
        msg_type = "question"

        yield self._ndjson({"type": "response.meta", "stage": "generating"})

        async for line in graph_to_ndjson_tokens(graph, {"messages": lc_messages}):
            obj = self._safe_json(line)
            if not obj:
                continue

            if obj.get("type") == "token":
                token = obj.get("text") or ""
                if not header_parsed:
                    header_buf += token
                    if "\n" in header_buf:
                        first_line, rest = header_buf.split("\n", 1)
                        parsed = self._parse_control_header(first_line.strip())
                        if parsed:
                            status = parsed.get("status", status)
                            msg_type = parsed.get("type", msg_type)
                        header_parsed = True

                        yield self._ndjson(
                            {
                                "type": "message.start",
                                "nodeId": node_id,
                                "title": title,
                                "task": payload.get("task", "heuristicWriting"),
                                "status": status,
                                "assistantMessage": {
                                    "messageId": msg_id,
                                    "role": "assistant",
                                    "type": msg_type,
                                    "content": "",
                                },
                            }
                        )

                        if rest:
                            yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": rest})
                    else:
                        if len(header_buf) > 1024:
                            header_parsed = True
                            yield self._ndjson(
                                {
                                    "type": "message.start",
                                    "nodeId": node_id,
                                    "title": title,
                                    "task": payload.get("task", "heuristicWriting"),
                                    "status": status,
                                    "assistantMessage": {
                                        "messageId": msg_id,
                                        "role": "assistant",
                                        "type": msg_type,
                                        "content": "",
                                    },
                                }
                            )
                            yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": header_buf})
                            header_buf = ""
                else:
                    if token:
                        yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": token})

            elif obj.get("type") == "done":
                if not header_parsed:
                    header_parsed = True
                    yield self._ndjson(
                        {
                            "type": "message.start",
                            "nodeId": node_id,
                            "title": title,
                            "task": payload.get("task", "heuristicWriting"),
                            "status": status,
                            "assistantMessage": {
                                "messageId": msg_id,
                                "role": "assistant",
                                "type": msg_type,
                                "content": "",
                            },
                        }
                    )
                    if header_buf:
                        yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": header_buf})

                yield self._ndjson({"type": "message.end", "messageId": msg_id, "done": True, "finishReason": "stop"})
                yield self._ndjson({"type": "done"})
                return

    def _build_graph(self):
        graph = StateGraph(HeuristicState)

        def merge_messages(state: HeuristicState) -> Dict[str, Any]:
            history = state.get("messages", []) or []
            incoming = state.get("incoming_messages", []) or []
            merged = self._merge_by_message_id(history, incoming)
            return {"messages": merged, "incoming_messages": []}

        def generate(state: HeuristicState) -> Dict[str, Any]:
            title = state.get("title", "")
            base_prompt = state.get("heuristic_prompt", "") or ""
            messages = state.get("messages", []) or []

            if not is_llm_configured():
                question = self._heuristic_question(self._count_user_answers(messages) + 1, title)
                response = {
                    "status": "ask",
                    "assistantMessage": {
                        "messageId": self._next_message_id(),
                        "role": "assistant",
                        "type": "question",
                        "content": question,
                    },
                }
                return {"messages": messages + [response["assistantMessage"]], "last_response": response}

            if self._count_user_answers(messages) >= 5 and not self._awaiting_confirmation(messages):
                content = "信息已经足够。请确认是否开始撰写本节正文？请回复“确认”或说明需要补充的点。"
                response = {
                    "status": "ask",
                    "assistantMessage": {
                        "messageId": self._next_message_id(),
                        "role": "assistant",
                        "type": "question",
                        "content": content,
                    },
                }
                return {"messages": messages + [response["assistantMessage"]], "last_response": response}

            force_mode = {"status": "draft", "type": "text"} if self._should_force_draft(messages) else None

            system_prompt = self._effective_system_prompt(base_prompt, force_mode=force_mode)
            user_text = f"当前小节：{title}\n请严格按输出格式强约束输出。"

            model = build_chat_model(streaming=False)
            lc_messages = build_messages(system_prompt=system_prompt, user_text=user_text, messages=messages)
            result = model.invoke(lc_messages)
            raw = getattr(result, "content", "") or ""

            ctrl, content = self._split_control_header(raw)
            status = (ctrl or {}).get("status") or "ask"
            msg_type = (ctrl or {}).get("type") or ("text" if status == "draft" else "question")

            response = {
                "status": status,
                "assistantMessage": {
                    "messageId": self._next_message_id(),
                    "role": "assistant",
                    "type": msg_type,
                    "content": content.strip(),
                },
            }
            return {"messages": messages + [response["assistantMessage"]], "last_response": response}

        graph.add_node("merge", merge_messages)
        graph.add_node("generate", generate)
        graph.set_entry_point("merge")
        graph.add_edge("merge", "generate")
        graph.set_finish_point("generate")
        return graph.compile(checkpointer=MemorySaver())

    def _build_stream_graph(self):
        model = build_chat_model(streaming=True)
        graph = StateGraph(dict)

        async def call_model(state: dict, config: RunnableConfig) -> dict:
            response = await model.ainvoke(state["messages"], config=config)
            return {"response": response}

        graph.add_node("call_model", call_model)
        graph.set_entry_point("call_model")
        graph.set_finish_point("call_model")
        return graph.compile()

    def _effective_system_prompt(
        self,
        base_prompt: str,
        force_mode: Optional[Dict[str, str]] = None,
    ) -> str:
        prompt = (base_prompt or DEFAULT_FALLBACK_PROMPT).rstrip()
        prompt += "\n\n" + POST_ANSWER_CONSTRAINT
        prompt += "\n\n" + CONTROL_HEADER_INSTRUCTION.strip()

        if force_mode:
            prompt += (
                "\n\n【强制模式】本轮你必须输出以下控制头（第一行 JSON）并遵循其含义：\n"
                f'{{"status":"{force_mode["status"]}","type":"{force_mode["type"]}"}}'
            )
        return prompt

    def _merge_by_message_id(self, history: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        merged: List[Dict[str, Any]] = []

        def add(msg: Dict[str, Any]):
            mid = msg.get("messageId")
            key = mid if mid else json.dumps(msg, ensure_ascii=False, sort_keys=True)
            if key in seen:
                return
            seen.add(key)
            merged.append(msg)

        for msg in history:
            add(msg)
        for msg in incoming:
            add(msg)

        return merged

    def _count_user_answers(self, messages: List[Dict[str, Any]]) -> int:
        return sum(1 for m in messages if m.get("role") == "user")

    def _awaiting_confirmation(self, messages: List[Dict[str, Any]]) -> bool:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return "请确认" in (msg.get("content", "") or "")
        return False

    def _is_confirmed(self, content: str) -> bool:
        return bool(re.search(r"(确认|可以开始|开始写|可以写)", content or ""))

    def _should_force_draft(self, messages: List[Dict[str, Any]]) -> bool:
        if not self._awaiting_confirmation(messages):
            return False
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if not last_user:
            return False
        return self._is_confirmed(last_user.get("content", "") or "")

    def _parse_control_header(self, line: str) -> Optional[Dict[str, str]]:
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                return None
            status = obj.get("status")
            typ = obj.get("type")
            if status in ("ask", "draft") and typ in ("question", "text"):
                return {"status": status, "type": typ}
            return None
        except Exception:
            return None

    def _split_control_header(self, raw: str) -> tuple[Optional[Dict[str, str]], str]:
        raw = raw or ""
        if "\n" not in raw:
            return None, raw
        first, rest = raw.split("\n", 1)
        ctrl = self._parse_control_header(first.strip())
        return ctrl, rest

    def _heuristic_question(self, index: int, title: str) -> str:
        if title:
            return f"[{title}] 请补充第 {index} 条关键信息。"
        return f"请补充第 {index} 条关键信息。"

    def _next_message_id(self, prefix: str = "m_ai") -> str:
        return f"{prefix}_{next(self._counter)}"

    def _ndjson(self, obj: Dict[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    def _safe_json(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(line)
        except Exception:
            return None
