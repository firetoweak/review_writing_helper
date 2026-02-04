from fastapi import APIRouter, Depends
from models.schemas import *
from routers.deps import get_ctx
from app.container import AppContext

router = APIRouter()


@router.post("/api/project-outline")
async def project_outline(request: ProjectOutlineRequest, ctx: AppContext = Depends(get_ctx)):
    return await ctx.others_service.generate_outline(request.model_dump())

@router.post("/api/project-outline/chapterKeyPoint")
async def chapter_key_point(request: ChapterKeyPointRequest, ctx: AppContext = Depends(get_ctx)):
    return await ctx.others_service.generate_chapter_key_point(request.model_dump())

@router.post("/api/project-outline/sectionKeyPoint")
async def section_key_point(request: SectionKeyPointRequest, ctx: AppContext = Depends(get_ctx)):
    return await ctx.others_service.generate_section_key_point(request.model_dump())
