from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from services.streaming import stream_json
from services.writing_service import (
    help_chat_message,
    help_chat_message_stream,
    help_chat_start,
    help_chat_start_stream,
)


router = APIRouter()


class SectionText(BaseModel):
    nodeId: str
    level: int
    title: str
    text: str


class Message(BaseModel):
    messageId: str
    role: str
    content: str
    attachmentPath: Optional[str] = None


class HelpChatRequest(BaseModel):
    task: str
    nodeId: str
    title: str
    text: List[SectionText]
    sessionId: str
    sessionText: str
    helpPrompt: Optional[str] = None
    messages: List[Message] = []
    stream: Optional[bool] = False


class HelpChatMessageRequest(BaseModel):
    task: str
    nodeId: str
    title: str
    sessionId: str
    messages: List[Message]
    helpPrompt: Optional[str] = None
    stream: Optional[bool] = False


@router.post("/api/i-can/chat")
async def help_chat(request: HelpChatRequest):
    payload = request.model_dump()
    if payload.get("stream"):
        return StreamingResponse(help_chat_start_stream(payload), media_type="application/x-ndjson")
    response = await help_chat_start(payload)
    return response


@router.post("/api/i-can/chat/message")
async def help_chat_message_endpoint(request: HelpChatMessageRequest):
    payload = request.model_dump()
    if payload.get("stream"):
        return StreamingResponse(help_chat_message_stream(payload), media_type="application/x-ndjson")
    response = await help_chat_message(payload)
    return response
