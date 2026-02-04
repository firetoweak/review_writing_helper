from fastapi import APIRouter, Depends
from models.schemas import *
from routers.deps import get_ctx
from app.container import AppContext

router = APIRouter()

@router.post("/api/float")
async def float_ans(request: FloatRequest, ctx: AppContext = Depends(get_ctx)):
    return await ctx.chat_service.float_response(request.model_dump())
