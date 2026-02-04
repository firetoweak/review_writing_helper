from fastapi import APIRouter, Depends
from models.schemas import *
from routers.deps import get_ctx
from app.container import AppContext

router = APIRouter()

@router.post("/api/industry")
async def industry(request: IndustryRequest, ctx: AppContext = Depends(get_ctx)):
    return await ctx.others_service.industry_response(request.model_dump())
