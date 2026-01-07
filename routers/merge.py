from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ai_writer_agent.models.schemas import MergeRequest
from services.writing_service import merge_texts, merge_texts_stream


router = APIRouter()


@router.post("/api/merge")
async def merge_endpoint(request: MergeRequest):
    payload = request.model_dump()
    if payload.get("stream"):
        return StreamingResponse(merge_texts_stream(payload), media_type="application/x-ndjson")
    response = await merge_texts(payload)
    return response
