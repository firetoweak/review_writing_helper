from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ai_writer_agent.models.schemas import FullReviewRequest, SectionReviewRequest
from services.review_service import full_review, full_review_stream, review_section

router = APIRouter()


@router.post("/api/section-review")
async def section_review(request: SectionReviewRequest):
    return await review_section(request.model_dump())


@router.post("/api/full-review")
async def full_review_endpoint(request: FullReviewRequest):
    payload = request.model_dump()
    if payload.get("stream"):
        return StreamingResponse(full_review_stream(payload), media_type="application/x-ndjson")
    response = await full_review(payload)
    return response
