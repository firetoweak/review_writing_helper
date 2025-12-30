from __future__ import annotations

from typing import Protocol

from ai_writer_agent.models.schemas import (
    FullPolishRequest,
    FullPolishResponse,
    FullReviewRequest,
    FullReviewResponse,
    HeuristicCreateRequest,
    HeuristicMessageRequest,
    HeuristicResponse,
    ICanCreateRequest,
    ICanMessageRequest,
    ICanResponse,
    KBDocumentActionRequest,
    KBDocumentActionResponse,
    MergeRequest,
    MergeResponse,
    ProjectOutlineRequest,
    ProjectOutlineResponse,
    SectionReviewRequest,
    SectionReviewResponse,
    HelpListResponse,
    TextRestructRequest,
    TextRestructResponse,
)


class WritingService(Protocol):
    async def project_outline(self, req: ProjectOutlineRequest) -> ProjectOutlineResponse: ...

    async def heuristic_create(self, req: HeuristicCreateRequest) -> HeuristicResponse: ...

    async def heuristic_message(self, req: HeuristicMessageRequest) -> HeuristicResponse: ...

    async def text_restruct(self, req: TextRestructRequest) -> TextRestructResponse: ...


class ReviewService(Protocol):
    async def section_review(self, req: SectionReviewRequest) -> SectionReviewResponse | HelpListResponse: ...

    async def full_review(self, req: FullReviewRequest) -> FullReviewResponse: ...

    async def full_polish(self, req: FullPolishRequest) -> FullPolishResponse: ...


class HelpService(Protocol):
    async def i_can_create(self, req: ICanCreateRequest) -> ICanResponse: ...

    async def i_can_message(self, req: ICanMessageRequest) -> ICanResponse: ...


class MergeService(Protocol):
    async def merge(self, req: MergeRequest) -> MergeResponse: ...


class KBService(Protocol):
    async def documents(self, req: KBDocumentActionRequest) -> KBDocumentActionResponse: ...


class ServiceBundle(Protocol):
    writing: WritingService
    review: ReviewService
    help: HelpService
    merge: MergeService
    kb: KBService
