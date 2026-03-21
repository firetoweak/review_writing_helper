from fastapi import APIRouter

from ai_writer_agent.models.schemas import OutlineMappingRequest
from services.writing_service import generate_outline_mapping


router = APIRouter()


@router.post("/api/outline-mapping")
async def outline_mapping(request: OutlineMappingRequest):
    return await generate_outline_mapping(request.model_dump())
