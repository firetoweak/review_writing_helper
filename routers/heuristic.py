from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from services.writing_service import continue_heuristic, start_heuristic


router = APIRouter()


class HistoryChild(BaseModel):
    nodeId: str
    title: str
    text: Optional[str] = None


class HistorySection(BaseModel):
    nodeId: str
    title: str
    level: int
    children: List[HistoryChild]


class Message(BaseModel):
    messageId: str
    role: str
    type: Optional[str] = None
    content: str
    attachmentPath: Optional[str] = None


class HeuristicRequest(BaseModel):
    nodeId: str
    title: str
    text: Optional[str] = None
    task: str
    historyText: Optional[List[HistorySection]] = None
    heuristicPrompt: Optional[str] = None
    messages: List[Message] = []


class HeuristicMessageRequest(BaseModel):
    nodeId: str
    title: str
    task: str
    messages: List[Message]
    heuristicPrompt: Optional[str] = None


@router.post("/api/heuristic-writing")
async def heuristic_start(request: HeuristicRequest):
    return await start_heuristic(request.model_dump())


@router.post("/api/heuristic-writing/message")
async def heuristic_message(request: HeuristicMessageRequest):
    return await continue_heuristic(request.model_dump())
