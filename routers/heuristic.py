from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ai_writer_agent.models.schemas import HeuristicCreateRequest, HeuristicMessageRequest
from services.writing_service import (
    continue_heuristic,
    heuristic_message_stream,
    heuristic_start_stream,
    start_heuristic,
)


router = APIRouter()


@router.post("/api/heuristic-writing")
async def heuristic_start(request: HeuristicCreateRequest):
    payload = request.model_dump()
    if payload.get("stream") is not False:
        return StreamingResponse(heuristic_start_stream(payload), media_type="application/x-ndjson")
    response = await start_heuristic(payload)
    return response


@router.post("/api/heuristic-writing/message")
async def heuristic_message(request: HeuristicMessageRequest):
    payload = request.model_dump()
    if payload.get("stream") is not False:
        return StreamingResponse(heuristic_message_stream(payload), media_type="application/x-ndjson")
    response = await continue_heuristic(payload)
    return response
