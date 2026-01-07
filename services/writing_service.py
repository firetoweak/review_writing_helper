import json

from ai_writer_agent.models.schemas import (
    HeuristicCreateRequest,
    HeuristicMessageRequest,
    ICanCreateRequest,
    ICanMessageRequest,
    KBDocumentActionRequest,
    FullPolishRequest,
    MergeRequest,
    ProjectOutlineRequest,
    TextRestructRequest,
)
from langgraph.graph import StateGraph
from models.llm_interface_async import build_chat_model, build_messages, is_llm_configured
from services.factory import get_service_bundle
from services.streaming_langgraph import graph_to_ndjson_tokens


async def generate_outline(payload: dict) -> dict:
    service = get_service_bundle().writing
    response = await service.project_outline(ProjectOutlineRequest(**payload))
    return response.model_dump()


async def start_heuristic(payload: dict) -> dict:
    service = get_service_bundle().writing
    response = await service.heuristic_create(HeuristicCreateRequest(**payload))
    return response.model_dump()


async def continue_heuristic(payload: dict) -> dict:
    service = get_service_bundle().writing
    response = await service.heuristic_message(HeuristicMessageRequest(**payload))
    return response.model_dump()


async def merge_texts(payload: dict) -> dict:
    service = get_service_bundle().merge
    response = await service.merge(MergeRequest(**payload))
    return response.model_dump()


async def full_polish(payload: dict) -> dict:
    service = get_service_bundle().review
    response = await service.full_polish(FullPolishRequest(**payload))
    return response.model_dump()


async def text_restruct(payload: dict) -> dict:
    service = get_service_bundle().writing
    response = await service.text_restruct(TextRestructRequest(**payload))
    return response.model_dump()


async def help_chat_start(payload: dict) -> dict:
    service = get_service_bundle().help
    response = await service.i_can_create(ICanCreateRequest(**payload))
    return response.model_dump()


async def help_chat_message(payload: dict) -> dict:
    service = get_service_bundle().help
    response = await service.i_can_message(ICanMessageRequest(**payload))
    return response.model_dump()


async def help_chat_start_stream(payload: dict):
    service = get_service_bundle().help
    help_agent = getattr(service, "help_agent", None)
    if help_agent:
        async for line in help_agent.stream(payload):
            yield line
        return
    if not is_llm_configured():
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        return
    graph, input_messages = _build_stream_graph(
        system_prompt=payload.get("helpPrompt") or "",
        user_text=payload.get("sessionText") or "",
        messages=payload.get("messages", []),
    )
    async for line in graph_to_ndjson_tokens(graph, {"messages": input_messages}):
        yield line


async def help_chat_message_stream(payload: dict):
    service = get_service_bundle().help
    help_agent = getattr(service, "help_agent", None)
    if help_agent:
        async for line in help_agent.stream(payload):
            yield line
        return
    if not is_llm_configured():
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        return
    graph, input_messages = _build_stream_graph(
        system_prompt=payload.get("helpPrompt") or "",
        user_text=payload.get("sessionText") or "",
        messages=payload.get("messages", []),
    )
    async for line in graph_to_ndjson_tokens(graph, {"messages": input_messages}):
        yield line


async def heuristic_start_stream(payload: dict):
    service = get_service_bundle().writing
    heuristic_agent = getattr(service, "heuristic_agent", None)
    if heuristic_agent:
        async for line in heuristic_agent.stream(payload):
            yield line
        return
    if not is_llm_configured():
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        return
    graph, input_messages = _build_stream_graph(
        system_prompt=payload.get("heuristicPrompt") or "",
        user_text=payload.get("text") or payload.get("title") or "",
        messages=payload.get("messages", []),
    )
    async for line in graph_to_ndjson_tokens(graph, {"messages": input_messages}):
        yield line


async def heuristic_message_stream(payload: dict):
    service = get_service_bundle().writing
    heuristic_agent = getattr(service, "heuristic_agent", None)
    if heuristic_agent:
        async for line in heuristic_agent.stream(payload):
            yield line
        return
    if not is_llm_configured():
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        return
    graph, input_messages = _build_stream_graph(
        system_prompt=payload.get("heuristicPrompt") or "",
        user_text=payload.get("text") or payload.get("title") or "",
        messages=payload.get("messages", []),
    )
    async for line in graph_to_ndjson_tokens(graph, {"messages": input_messages}):
        yield line


async def merge_texts_stream(payload: dict):
    if not is_llm_configured():
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
        return
    graph, input_messages = _build_stream_graph(
        system_prompt=payload.get("mergePrompt") or "",
        user_text="",
        messages=payload.get("messages", []) + payload.get("text", []),
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


async def kb_action(payload: dict) -> dict:
    service = get_service_bundle().kb
    response = await service.documents(KBDocumentActionRequest(**payload))
    return response.model_dump()
