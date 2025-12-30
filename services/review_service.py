import json

from ai_writer_agent.models.schemas import FullReviewRequest, SectionReviewRequest, FullReviewResponse
from models.llm_interface_async import LLMInterfaceAsync
from services.streaming import stream_tokens
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
    llm = LLMInterfaceAsync()
    system_prompt = payload.get("fullReviewPrompt") or ""
    user_text = json.dumps(payload.get("fullText", []), ensure_ascii=False)
    tokens = llm.stream_chat_tokens(
        ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + [{"role": "user", "content": user_text}],
        max_tokens=2000,
    )
    async for line in stream_tokens(tokens):
        yield line


async def full_polish_stream(payload: dict):
    llm = LLMInterfaceAsync()
    system_prompt = payload.get("polishPrompt") or ""
    user_text = json.dumps(payload.get("fullText", []), ensure_ascii=False)
    tokens = llm.stream_chat_tokens(
        ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + [{"role": "user", "content": user_text}],
        max_tokens=2000,
    )
    async for line in stream_tokens(tokens):
        yield line
