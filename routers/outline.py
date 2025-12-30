from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from services.writing_service import generate_outline


router = APIRouter()


class Attachment(BaseModel):
    name: str
    mimeType: Optional[str] = None
    size: Optional[int] = None


class Project(BaseModel):
    title: str
    idea: Optional[str] = None
    attachments: List[Attachment] = []


class OutlineRequest(BaseModel):
    task: str
    project: Project
    outlinePrompt: Optional[str] = None


@router.post("/api/project-outline")
async def project_outline(request: OutlineRequest):
    return await generate_outline(request.model_dump())
