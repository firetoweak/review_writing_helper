from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from services.writing_service import kb_action


router = APIRouter()


class KBRequest(BaseModel):
    action: str
    document_id: str
    file_url: Optional[str] = None
    filename: Optional[str] = None


@router.post("/api/kb/documents")
async def kb_documents(request: KBRequest):
    return await kb_action(request.model_dump())
