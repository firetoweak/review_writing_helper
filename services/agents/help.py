from __future__ import annotations

import itertools
import operator
from typing import Any, Dict, List, TypedDict, Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph


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

    def start(self, payload: Dict) -> Dict:
        session_id = str(payload.get("sessionId", ""))
        node_id = payload.get("nodeId", "")
        title = payload.get("title", "")
        session_text = payload.get("sessionText", "")
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
        messages = payload.get("messages", [])
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

    def _help_response(self, messages: List[Dict], session_text: str) -> str:
        if messages:
            last_content = messages[-1].get("content", "")
            if "解释" in last_content or "是什么" in last_content:
                return f"关于“{last_content}”，这是对关键概念的解释，并结合你的需求给出示例。"
            return f"已收到你的问题：{last_content}。我将结合“{session_text}”给出改进建议。"
        return f"围绕“{session_text}”，请补充你希望完善的具体方向。"

    def _next_message_id(self, prefix: str = "m_ai") -> str:
        return f"{prefix}_{next(self._counter)}"
