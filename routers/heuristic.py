from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from services.streaming import stream_json
from services.writing_service import (
    continue_heuristic,
    heuristic_message_stream,
    heuristic_start_stream,
    start_heuristic,
)


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
    stream: Optional[bool] = False


class HeuristicMessageRequest(BaseModel):
    nodeId: str
    title: str
    task: str
    messages: List[Message]
    heuristicPrompt: Optional[str] = None
    stream: Optional[bool] = False


@router.post("/api/heuristic-writing")
async def heuristic_start(request: HeuristicRequest):
    payload = request.model_dump()
    if payload.get("stream"):
        return StreamingResponse(heuristic_start_stream(payload), media_type="application/x-ndjson")
    response = await start_heuristic(payload)
    return response


@router.post("/api/heuristic-writing/message")
async def heuristic_message(request: HeuristicMessageRequest):
    payload = request.model_dump()
    if payload.get("stream"):
        return StreamingResponse(heuristic_message_stream(payload), media_type="application/x-ndjson")
    response = await continue_heuristic(payload)
    return response
