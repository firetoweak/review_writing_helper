from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from services.writing_service import help_chat_message, help_chat_start


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


class HelpChatMessageRequest(BaseModel):
    task: str
    nodeId: str
    title: str
    sessionId: str
    messages: List[Message]
    helpPrompt: Optional[str] = None


@router.post("/api/i-can/chat")
async def help_chat(request: HelpChatRequest):
    return await help_chat_start(request.model_dump())


@router.post("/api/i-can/chat/message")
async def help_chat_message_endpoint(request: HelpChatMessageRequest):
    return await help_chat_message(request.model_dump())
