from fastapi import APIRouter, Depends
from models.schemas import *
from routers.deps import get_ctx
from app.container import AppContext

router = APIRouter()


@router.post("/api/full-polish")
async def full_polish_endpoint(request: FullPolishRequest, ctx: AppContext = Depends(get_ctx)):
    response = await ctx.others_service.full_polish(request.model_dump())
    return response