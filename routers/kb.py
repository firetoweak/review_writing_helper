from fastapi import APIRouter

from ai_writer_agent.models.schemas import KBDocumentActionRequest
from services.writing_service import kb_action


router = APIRouter()


@router.post("/api/kb/documents")
async def kb_documents(request: KBDocumentActionRequest):
    return await kb_action(request.model_dump())
