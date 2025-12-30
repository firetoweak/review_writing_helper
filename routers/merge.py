from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from services.writing_service import merge_texts


router = APIRouter()


class SectionText(BaseModel):
    nodeId: str
    level: int
    title: str
    text: str


class SessionMessage(BaseModel):
    messageId: str
    role: str
    content: str
    attachmentPath: Optional[List[str]] = None


class SessionItem(BaseModel):
    sesstionId: str
    messages: List[SessionMessage]


class MergeRequest(BaseModel):
    nodeId: str
    title: str
    task: str
    text: List[SectionText]
    sessionList: List[SessionItem]
    mergePrompt: Optional[str] = None
    historyText: Optional[List[dict]] = None


@router.post("/api/merge")
async def merge_endpoint(request: MergeRequest):
    return await merge_texts(request.model_dump())
