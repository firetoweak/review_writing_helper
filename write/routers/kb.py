from fastapi import APIRouter, Depends
from models.schemas import *
from routers.deps import get_ctx
from app.container import AppContext

router = APIRouter()

@router.post("/api/kb/documents")
async def kb_documents(request: KBDocumentActionRequest, ctx: AppContext = Depends(get_ctx)):
    return await ctx.kb_admin_service.action(request.model_dump())
