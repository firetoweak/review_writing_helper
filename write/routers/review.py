from fastapi import APIRouter, Depends
from models.schemas import *
from routers.deps import get_ctx
from app.container import AppContext

router = APIRouter()


@router.post("/api/section-review")
async def section_review(request: SectionReviewRequest, ctx: AppContext = Depends(get_ctx)):
    return await ctx.others_service.review(request.model_dump())

@router.post("/api/chapter-review")
async def chapter_review(request: ChapterReviewRequest, ctx: AppContext = Depends(get_ctx)):
    return await ctx.others_service.review(request.model_dump())

@router.post("/api/full-review")
async def full_review_endpoint(request: FullReviewRequest, ctx: AppContext = Depends(get_ctx)):
    return await ctx.others_service.full_review(request.model_dump())