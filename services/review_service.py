from ai_writer_agent.models.schemas import FullReviewRequest, SectionReviewRequest, FullReviewResponse
from services.factory import get_service_bundle


async def review_section(payload: dict) -> dict:
    service = get_service_bundle().review
    response = await service.section_review(SectionReviewRequest(**payload))
    return response.model_dump()

async def full_review(payload: dict) -> dict:
    service = get_service_bundle().review
    response = await service.full_review(FullReviewRequest(**payload))
    return response.model_dump()
