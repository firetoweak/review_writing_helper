from fastapi import APIRouter

from ai_writer_agent.models.schemas import TextRestructRequest
from services.writing_service import text_restruct


router = APIRouter()


@router.post("/api/text-restruct")
async def text_restruct_endpoint(request: TextRestructRequest):
    return await text_restruct(request.model_dump())
