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
from models.llm_interface_async import LLMInterfaceAsync
from services.factory import get_service_bundle
from services.streaming import stream_tokens


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
    llm = LLMInterfaceAsync()
    messages = payload.get("messages", [])
    system_prompt = payload.get("helpPrompt") or ""
    user_text = payload.get("sessionText") or ""
    tokens = llm.stream_chat_tokens(
        ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + messages
        + [{"role": "user", "content": user_text}],
        max_tokens=2000,
    )
    async for line in stream_tokens(tokens):
        yield line


async def help_chat_message_stream(payload: dict):
    llm = LLMInterfaceAsync()
    system_prompt = payload.get("helpPrompt") or ""
    user_text = json.dumps(payload.get("messages", []), ensure_ascii=False)
    tokens = llm.stream_chat_tokens(
        ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + [{"role": "user", "content": user_text}],
        max_tokens=2000,
    )
    async for line in stream_tokens(tokens):
        yield line


async def heuristic_start_stream(payload: dict):
    llm = LLMInterfaceAsync()
    system_prompt = payload.get("heuristicPrompt") or ""
    user_text = payload.get("text") or payload.get("title") or ""
    tokens = llm.stream_chat_tokens(
        ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + [{"role": "user", "content": user_text}],
        max_tokens=2000,
    )
    async for line in stream_tokens(tokens):
        yield line


async def heuristic_message_stream(payload: dict):
    llm = LLMInterfaceAsync()
    system_prompt = payload.get("heuristicPrompt") or ""
    user_text = json.dumps(payload.get("messages", []), ensure_ascii=False)
    tokens = llm.stream_chat_tokens(
        ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + [{"role": "user", "content": user_text}],
        max_tokens=2000,
    )
    async for line in stream_tokens(tokens):
        yield line


async def merge_texts_stream(payload: dict):
    llm = LLMInterfaceAsync()
    system_prompt = payload.get("mergePrompt") or ""
    user_text = json.dumps(payload.get("text", []), ensure_ascii=False)
    tokens = llm.stream_chat_tokens(
        ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + [{"role": "user", "content": user_text}],
        max_tokens=2000,
    )
    async for line in stream_tokens(tokens):
        yield line


async def kb_action(payload: dict) -> dict:
    service = get_service_bundle().kb
    response = await service.documents(KBDocumentActionRequest(**payload))
    return response.model_dump()
