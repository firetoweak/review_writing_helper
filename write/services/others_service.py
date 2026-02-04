from __future__ import annotations

from typing import Any
from models.schemas import *

class OthersService:
    def __init__(self, *,
                 floatAns_agent: Any,
                 industry_agent: Any,
                 merge_agent: Any,
                 outline_agent: Any,
                 polish_agent: Any,
                 review_agent: Any,
                 ):
        self.floatAns_agent = floatAns_agent
        self.industry_agent = industry_agent
        self.merge_agent = merge_agent
        self.outline_agent = outline_agent
        self.polish_agent = polish_agent
        self.review_agent = review_agent

    async def float_response(self, req: dict) -> FloatResponse:
        data = await self.floatAns_agent.float_response(req)
        return FloatResponse(**data)

    async def industry_response(self, req: dict) -> IndustryResponse:
        data = await self.industry_agent.industry(req)
        return IndustryResponse(**data)
    
    async def merge_texts(self, req: dict) -> MergeResponse:
        data = await self.merge_agent.amerge_texts(req)
        return MergeResponse(**data)

    async def generate_outline(self, req: dict) -> ProjectOutlineResponse:
        data = await self.outline_agent.agenerate_outline(req)
        return ProjectOutlineResponse(**data)
    
    async def generate_chapter_key_point(self, req: dict) -> ChapterKeyPointResponse:
        data = await self.outline_agent.chapter_key_point(req)
        return ChapterKeyPointResponse(**data)
    
    async def generate_section_key_point(self, req: dict) -> SectionKeyPointResponse:
        data = await self.outline_agent.section_key_point(req)
        return SectionKeyPointResponse(**data)
    
    async def full_polish(self, req: dict) -> FullPolishResponse:
        data = await self.polish_agent.full_polish(req)
        return FullPolishResponse(**data)
    
    async def review(self, req: dict) :
        data = await self.review_agent.review(req)
        return ReviewResponse(**data)
    
    async def full_review(self, req: dict) -> FullReviewResponse:
        data = await self.review_agent.full_review(req)
        return FullReviewResponse(**data)
