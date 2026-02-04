from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from models.schemas import ICanCreateRequest
from routers.deps import get_ctx
from app.container import AppContext


router = APIRouter()


@router.post("/api/i-can/chat")
async def help_chat(request: ICanCreateRequest, ctx: AppContext = Depends(get_ctx)):
    payload = request.model_dump(exclude_unset=True)
    if payload.get("stream", True):
        return StreamingResponse(ctx.chat_service.help_stream(payload), media_type="application/x-ndjson")
    

@router.post("/api/i-can/chat/message")
async def help_chat_message_endpoint(request: ICanCreateRequest, ctx: AppContext = Depends(get_ctx)):
    payload = request.model_dump(exclude_unset=True)
    if payload.get("stream", True):
        return StreamingResponse(ctx.chat_service.help_stream(payload), media_type="application/x-ndjson")
