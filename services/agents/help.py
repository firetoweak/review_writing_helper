from __future__ import annotations

import itertools
import json
import operator
from typing import Any, Dict, List, Optional, TypedDict, Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from models.llm_interface_async import build_chat_model, build_messages, is_llm_configured
from services.streaming_langgraph import graph_to_ndjson_tokens

class HelpState(TypedDict):
    node_id: str
    title: str
    session_text: str
    help_prompt: str
    messages: Annotated[List[Dict[str, Any]], operator.add]
    last_response: Dict[str, Any]


class HelpAgent:
    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._graph = self._build_graph()
        self._stream_graph = self._build_stream_graph()

    def start(self, payload: Dict) -> Dict:
        session_id = str(payload.get("sessionId", ""))
        node_id = payload.get("nodeId", "")
        title = payload.get("title", "")
        session_text = payload.get("helpText") or payload.get("sessionText", "")
        help_prompt = payload.get("helpPrompt") or ""
        thread_id = f"help:{session_id}"
        state = self._graph.invoke(
            {
                "node_id": node_id,
                "title": title,
                "session_text": session_text,
                "messages": payload.get("messages", []),
                "help_prompt": help_prompt,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        response = state["last_response"]
        return {
            "task": payload.get("task", "help"),
            "nodeId": node_id,
            "title": title,
            "sessionId": session_id,
            **response,
        }

    def message(self, payload: Dict) -> Dict:
        session_id = str(payload.get("sessionId", ""))
        node_id = payload.get("nodeId", "")
        title = payload.get("title", "")
        help_prompt = payload.get("helpPrompt") or ""
        thread_id = f"help:{session_id}"
        messages = payload.get("messages", []) or []
        if payload.get("message"):
            messages = messages + [payload["message"]]
        state = self._graph.invoke(
            {
                "node_id": node_id,
                "title": title,
                "messages": messages,
                "help_prompt": help_prompt,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        response = state["last_response"]
        return {
            "task": payload.get("task", "help"),
            "nodeId": node_id,
            "title": title,
            "sessionId": session_id,
            **response,
        }

    async def stream(self, payload: Dict):
        session_id = str(payload.get("sessionId", ""))
        node_id = payload.get("nodeId", "")
        title = payload.get("title", "")
        session_text = payload.get("helpText") or payload.get("sessionText", "")
        help_prompt = payload.get("helpPrompt") or ""
        messages = payload.get("messages", []) or []
        if payload.get("message"):
            messages = messages + [payload["message"]]

        if not is_llm_configured():
            content = self._help_response(messages, session_text)
            if help_prompt:
                content = f"{content} 参考提示：{help_prompt}"
            msg_id = self._next_message_id()
            yield self._ndjson(
                {
                    "type": "message.start",
                    "nodeId": node_id,
                    "title": title,
                    "task": payload.get("task", "help"),
                    "sessionId": session_id,
                    "assistantMessage": {
                        "messageId": msg_id,
                        "role": "assistant",
                        "content": "",
                    },
                }
            )
            yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": content})
            yield self._ndjson({"type": "message.end", "messageId": msg_id, "done": True, "finishReason": "stop"})
            yield self._ndjson({"type": "done"})
            return

        system_prompt = help_prompt or ""
        user_text = session_text
        lc_messages = build_messages(system_prompt=system_prompt, user_text=user_text, messages=messages)

        msg_id = self._next_message_id()
        yield self._ndjson(
            {
                "type": "message.start",
                "nodeId": node_id,
                "title": title,
                "task": payload.get("task", "help"),
                "sessionId": session_id,
                "assistantMessage": {
                    "messageId": msg_id,
                    "role": "assistant",
                    "content": "",
                },
            }
        )
        async for line in graph_to_ndjson_tokens(self._stream_graph, {"messages": lc_messages}):
            obj = self._safe_json(line)
            if not obj:
                continue
            if obj.get("type") == "token":
                token = obj.get("text") or ""
                if token:
                    yield self._ndjson({"type": "message.delta", "messageId": msg_id, "delta": token})
            elif obj.get("type") == "done":
                break

        yield self._ndjson({"type": "message.end", "messageId": msg_id, "done": True, "finishReason": "stop"})
        yield self._ndjson({"type": "done"})

    def _build_graph(self):
        graph = StateGraph(HelpState)

        def respond(state: HelpState) -> Dict[str, Any]:
            session_text = state.get("session_text", "")
            messages = state.get("messages", [])
            help_prompt = state.get("help_prompt", "")
            content = self._help_response(messages, session_text)
            if help_prompt:
                content = f"{content} 参考提示：{help_prompt}"
            response = {
                "assistantMessage": {
                    "messageId": self._next_message_id(),
                    "role": "assistant",
                    "content": content,
                }
            }
            return {"messages": [response["assistantMessage"]], "last_response": response}

        graph.add_node("respond", respond)
        graph.set_entry_point("respond")
        graph.set_finish_point("respond")
        return graph.compile(checkpointer=MemorySaver())

    def _build_stream_graph(self):
        model = build_chat_model(streaming=True)
        graph = StateGraph(dict)

        async def call_model(state: dict) -> dict:
            response = await model.ainvoke(state["messages"])
            return {"response": response}

        graph.add_node("call_model", call_model)
        graph.set_entry_point("call_model")
        graph.set_finish_point("call_model")
        return graph.compile()

    def _help_response(self, messages: List[Dict], session_text: str) -> str:
        if messages:
            last_content = messages[-1].get("content", "")
            if "解释" in last_content or "是什么" in last_content:
                return f"关于“{last_content}”，这是对关键概念的解释，并结合你的需求给出示例。"
            return f"已收到你的问题：{last_content}。我将结合“{session_text}”给出改进建议。"
        return f"围绕“{session_text}”，请补充你希望完善的具体方向。"

    def _ndjson(self, obj: Dict[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    def _safe_json(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(line)
        except Exception:
            return None

    def _next_message_id(self, prefix: str = "m_ai") -> str:
        return f"{prefix}_{next(self._counter)}"
