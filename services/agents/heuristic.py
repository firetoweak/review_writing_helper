from __future__ import annotations

import itertools
import operator
import re
from typing import Any, Dict, List, TypedDict, Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from models.llm_interface_async import build_chat_model, build_messages, is_llm_configured


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
            if not is_llm_configured():
                question = self._heuristic_question(len(user_answers) + 1, title)
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

            if self._should_write_draft(messages):
                content = self._generate_draft(title, heuristic_prompt, messages)
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

            if len(user_answers) >= 5 and not self._awaiting_confirmation(messages):
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
                return {"messages": [response["assistantMessage"]], "last_response": response}

            question = self._generate_question(title, heuristic_prompt, messages)
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

    def _should_write_draft(self, messages: List[Dict]) -> bool:
        if not messages:
            return False
        last_user = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
        if not last_user:
            return False
        content = last_user.get("content", "")
        return self._awaiting_confirmation(messages) and self._is_confirmed(content)

    def _awaiting_confirmation(self, messages: List[Dict]) -> bool:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return "请确认" in msg.get("content", "")
        return False

    def _is_confirmed(self, content: str) -> bool:
        return bool(re.search(r"(确认|可以开始|开始写|可以写)", content))

    def _generate_question(self, title: str, prompt: str, messages: List[Dict]) -> str:
        model = build_chat_model(streaming=False)
        system_prompt = prompt or "你是写作教练，请用循序追问方式提出一个清晰问题。"
        user_text = (
            f"当前小节：{title}\n"
            "请基于已有信息提出下一个最关键的问题，只输出一个问题。"
        )
        lc_messages = build_messages(system_prompt=system_prompt, user_text=user_text, messages=messages)
        result = model.invoke(lc_messages)
        return getattr(result, "content", "") or self._heuristic_question(len(messages) + 1, title)

    def _generate_draft(self, title: str, prompt: str, messages: List[Dict]) -> str:
        model = build_chat_model(streaming=False)
        system_prompt = prompt or "你是写作助手，请基于对话内容生成该小节正文。"
        user_text = (
            f"当前小节：{title}\n"
            "请根据对话内容输出完整正文，不要添加额外说明。"
        )
        lc_messages = build_messages(system_prompt=system_prompt, user_text=user_text, messages=messages)
        result = model.invoke(lc_messages)
        return getattr(result, "content", "")

    def _next_message_id(self, prefix: str = "m_ai") -> str:
        return f"{prefix}_{next(self._counter)}"
