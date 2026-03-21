from __future__ import annotations

import itertools
import operator
from typing import Any, Dict, List, TypedDict, Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph


class HeuristicState(TypedDict):
    node_id: str
    title: str
    heuristic_prompt: str
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
        thread_id = f"heuristic:{node_id}"
        state = self._graph.invoke(
            {
                "node_id": node_id,
                "title": title,
                "messages": [],
                "heuristic_prompt": heuristic_prompt,
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
        thread_id = f"heuristic:{node_id}"
        messages = payload.get("messages", [])
        state = self._graph.invoke(
            {
                "node_id": node_id,
                "title": title,
                "messages": messages,
                "heuristic_prompt": heuristic_prompt,
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

    def _build_graph(self):
        graph = StateGraph(HeuristicState)

        def generate(state: HeuristicState) -> Dict[str, Any]:
            title = state.get("title", "")
            messages = state.get("messages", [])
            heuristic_prompt = state.get("heuristic_prompt", "")
            user_answers = [msg for msg in messages if msg.get("role") == "user"]
            if len(user_answers) >= 5:
                content = self._heuristic_draft(title, user_answers)
                response = {
                    "status": "draft",
                    "assistantMessage": {
                        "messageId": self._next_message_id(),
                        "role": "assistant",
                        "type": "text",
                        "content": content,
                    },
                }
                return {"messages": [response["assistantMessage"]], "last_response": response}
            question = self._heuristic_question(len(user_answers) + 1, title)
            if heuristic_prompt:
                question = f"{question} 提示：{heuristic_prompt}"
            response = {
                "status": "ask",
                "assistantMessage": {
                    "messageId": self._next_message_id(),
                    "role": "assistant",
                    "type": "question",
                    "content": question,
                },
            }
            return {"messages": [response["assistantMessage"]], "last_response": response}

        graph.add_node("generate", generate)
        graph.set_entry_point("generate")
        graph.set_finish_point("generate")
        return graph.compile(checkpointer=MemorySaver())

    def _heuristic_question(self, index: int, title: str) -> str:
        if title:
            return f"[{title}] 请补充第 {index} 条关键信息。"
        return f"请补充第 {index} 条关键信息。"

    def _heuristic_draft(self, title: str, answers: List[Dict]) -> str:
        summary = "\n".join(
            f"- {item.get('content', '')}" for item in answers if item.get("content")
        )
        return "\n".join(filter(None, [title, summary]))

    def _next_message_id(self, prefix: str = "m_ai") -> str:
        return f"{prefix}_{next(self._counter)}"
