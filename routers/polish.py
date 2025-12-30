from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from services.streaming import stream_json
from services.writing_service import full_polish


router = APIRouter()


class SectionText(BaseModel):
    nodeId: str
    level: int
    title: str
    text: str


class FullTextSection(BaseModel):
    nodeId: str
    title: str
    level: int
    children: List[SectionText]


class FullPolishRequest(BaseModel):
    task: str
    fullText: List[FullTextSection]
    polishPrompt: Optional[str] = None
    stream: Optional[bool] = False


@router.post("/api/full-polish")
async def full_polish_endpoint(request: FullPolishRequest):
    payload = request.model_dump()
    response = await full_polish(payload)
    if payload.get("stream"):
        return StreamingResponse(stream_json(response), media_type="application/x-ndjson")
    return response
