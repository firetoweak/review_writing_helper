from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ai_writer_agent.models.schemas import FullPolishRequest
from services.review_service import full_polish_stream
from services.writing_service import full_polish


router = APIRouter()


@router.post("/api/full-polish")
async def full_polish_endpoint(request: FullPolishRequest):
    payload = request.model_dump()
    if payload.get("stream"):
        return StreamingResponse(full_polish_stream(payload), media_type="application/x-ndjson")
    response = await full_polish(payload)
    return response
