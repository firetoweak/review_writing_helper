from fastapi import APIRouter

from ai_writer_agent.models.schemas import ProjectOutlineRequest
from services.writing_service import generate_outline


router = APIRouter()


@router.post("/api/project-outline")
async def project_outline(request: ProjectOutlineRequest):
    return await generate_outline(request.model_dump())
