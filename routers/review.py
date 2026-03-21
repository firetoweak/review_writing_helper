from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from services.review_service import full_review, review_section


router = APIRouter()


class SectionText(BaseModel):
    nodeId: str
    level: int
    title: str
    text: str


class ReviewRequest(BaseModel):
    task: str
    nodeId: str
    title: str
    text: List[SectionText]
    historyText: List[dict] | None = None
    reviewpPrompt: str | None = None


class FullReviewSection(BaseModel):
    nodeId: str
    title: str
    level: int
    children: List[SectionText]


class FullReviewRequest(BaseModel):
    task: str
    fullText: List[FullReviewSection]
    reviews: List[dict] | None = None
    fullReviewPrompt: str | None = None


@router.post("/api/section-review")
async def section_review(request: ReviewRequest):
    return await review_section(request.model_dump())


@router.post("/api/full-review")
async def full_review_endpoint(request: FullReviewRequest):
    return await full_review(request.model_dump())
