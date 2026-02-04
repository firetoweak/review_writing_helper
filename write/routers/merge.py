from fastapi import APIRouter, Depends
from models.schemas import *
from routers.deps import get_ctx
from app.container import AppContext

router = APIRouter()

@router.post("/api/merge")
async def merge_endpoint(request: MergeRequest, ctx: AppContext = Depends(get_ctx)):
    return await ctx.others_service.merge_texts(request.model_dump())
