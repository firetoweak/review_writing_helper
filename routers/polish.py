from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

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


@router.post("/api/full-polish")
async def full_polish_endpoint(request: FullPolishRequest):
    return await full_polish(request.model_dump())
