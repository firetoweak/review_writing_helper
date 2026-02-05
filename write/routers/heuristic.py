from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from models.schemas import HeuristicCreateRequest, HeuristicMessageRequest
from routers.deps import get_ctx
from app.container import AppContext

router = APIRouter()

@router.post("/api/heuristic-writing")
async def heuristic_start(request: HeuristicCreateRequest, ctx: AppContext = Depends(get_ctx)):
    payload = request.model_dump(exclude_unset=True, exclude_none=True)
    if payload.get("stream", True):
        return StreamingResponse(ctx.chat_service.heuristic_stream(payload), media_type="application/x-ndjson")
    return await ctx.chat_service.heuristic_run(payload)


@router.post("/api/heuristic-writing/message")
async def heuristic_message(request: HeuristicMessageRequest, ctx: AppContext = Depends(get_ctx)):
    payload = request.model_dump(exclude_unset=True, exclude_none=True)
    if payload.get("stream", True):
        return StreamingResponse(ctx.chat_service.heuristic_stream(payload), media_type="application/x-ndjson")
    return await ctx.chat_service.heuristic_run(payload)
