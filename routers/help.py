from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ai_writer_agent.models.schemas import ICanCreateRequest, ICanMessageRequest
from services.writing_service import (
    help_chat_message,
    help_chat_message_stream,
    help_chat_start,
    help_chat_start_stream,
)


router = APIRouter()


@router.post("/api/i-can/chat")
async def help_chat(request: ICanCreateRequest):
    payload = request.model_dump()
    if payload.get("stream"):
        return StreamingResponse(help_chat_start_stream(payload), media_type="application/x-ndjson")
    response = await help_chat_start(payload)
    return response


@router.post("/api/i-can/chat/message")
async def help_chat_message_endpoint(request: ICanMessageRequest):
    payload = request.model_dump()
    if payload.get("stream"):
        return StreamingResponse(help_chat_message_stream(payload), media_type="application/x-ndjson")
    response = await help_chat_message(payload)
    return response
