import json

from ai_writer_agent.models.schemas import FullReviewRequest, SectionReviewRequest, FullReviewResponse
from langgraph.graph import StateGraph
from models.llm_interface_async import build_chat_model, build_messages, is_llm_configured
from services.streaming_langgraph import graph_to_ndjson_tokens
from services.factory import get_service_bundle


async def review_section(payload: dict) -> dict:
    service = get_service_bundle().review
    response = await service.section_review(SectionReviewRequest(**payload))
    return response.model_dump()

async def full_review(payload: dict) -> dict:
    service = get_service_bundle().review
    response = await service.full_review(FullReviewRequest(**payload))
    return response.model_dump()


async def full_review_stream(payload: dict):
    if not is_llm_configured():
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        return
    graph, input_messages = _build_stream_graph(
        system_prompt=payload.get("fullReviewPrompt") or "",
        user_text=str(payload.get("fullText", [])),
        messages=payload.get("messages", []),
    )
    async for line in graph_to_ndjson_tokens(graph, {"messages": input_messages}):
        yield line


async def full_polish_stream(payload: dict):
    if not is_llm_configured():
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        return
    graph, input_messages = _build_stream_graph(
        system_prompt=payload.get("polishPrompt") or "",
        user_text=str(payload.get("fullText", [])),
        messages=payload.get("messages", []),
    )
    async for line in graph_to_ndjson_tokens(graph, {"messages": input_messages}):
        yield line


def _build_stream_graph(*, system_prompt: str, user_text: str, messages: list) -> tuple[StateGraph, list]:
    model = build_chat_model(streaming=True)
    lc_messages = build_messages(system_prompt=system_prompt, user_text=user_text, messages=messages)
    graph = StateGraph(dict)

    async def call_model(state: dict) -> dict:
        response = await model.ainvoke(state["messages"])
        return {"response": response}

    graph.add_node("call_model", call_model)
    graph.set_entry_point("call_model")
    graph.set_finish_point("call_model")
    compiled = graph.compile()
    return compiled, lc_messages
