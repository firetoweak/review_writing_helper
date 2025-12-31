from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from services.writing_service import text_restruct


router = APIRouter()


class TextRestructRequest(BaseModel):
    task: str
    file_path: str
    restructPrompt: Optional[str] = None
    outlinePrompt: Optional[str] = None


@router.post("/api/text-restruct")
async def text_restruct_endpoint(request: TextRestructRequest):
    return await text_restruct(request.model_dump())
