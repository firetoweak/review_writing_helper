from __future__ import annotations

from dataclasses import dataclass

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
from services.agents.heuristic import HeuristicAgent
from services.agents.help import HelpAgent
from services.agents.kb import KBStore
from services.agents.merge import MergeAgent
from services.agents.outline import OutlineGenerator
from services.agents.polish import PolishAgent
from services.agents.review import ReviewAgent
from services.agents.restruct import RestructAgent
from services.contracts import HelpService, KBService, MergeService, ReviewService, ServiceBundle, WritingService


@dataclass
class WritingServiceImpl(WritingService):
    outline_agent: OutlineGenerator
    heuristic_agent: HeuristicAgent
    restruct_agent: RestructAgent

    async def project_outline(self, req: ProjectOutlineRequest) -> ProjectOutlineResponse:
        return ProjectOutlineResponse(**self.outline_agent.generate_outline(req.model_dump()))

    async def heuristic_create(self, req: HeuristicCreateRequest) -> HeuristicResponse:
        return HeuristicResponse(**self.heuristic_agent.start(req.model_dump()))

    async def heuristic_message(self, req: HeuristicMessageRequest) -> HeuristicResponse:
        return HeuristicResponse(**self.heuristic_agent.message(req.model_dump()))

    async def text_restruct(self, req: TextRestructRequest) -> TextRestructResponse:
        return TextRestructResponse(**self.restruct_agent.text_restruct(req.model_dump()))


@dataclass
class ReviewServiceImpl(ReviewService):
    review_agent: ReviewAgent
    polish_agent: PolishAgent

    async def section_review(
        self, req: SectionReviewRequest
    ) -> SectionReviewResponse | HelpListResponse:
        data = self.review_agent.review_section(req.model_dump())
        return SectionReviewResponse(**data)

    async def full_review(self, req: FullReviewRequest) -> FullReviewResponse:
        return FullReviewResponse(**self.review_agent.full_review(req.model_dump()))

    async def full_polish(self, req: FullPolishRequest) -> FullPolishResponse:
        return FullPolishResponse(**self.polish_agent.full_polish(req.model_dump()))


@dataclass
class HelpServiceImpl(HelpService):
    help_agent: HelpAgent

    async def i_can_create(self, req: ICanCreateRequest) -> ICanResponse:
        return ICanResponse(**self.help_agent.start(req.model_dump()))

    async def i_can_message(self, req: ICanMessageRequest) -> ICanResponse:
        return ICanResponse(**self.help_agent.message(req.model_dump()))


@dataclass
class MergeServiceImpl(MergeService):
    merge_agent: MergeAgent

    async def merge(self, req: MergeRequest) -> MergeResponse:
        return MergeResponse(**self.merge_agent.merge_texts(req.model_dump()))


@dataclass
class KBServiceImpl(KBService):
    kb_store: KBStore

    async def documents(self, req: KBDocumentActionRequest) -> KBDocumentActionResponse:
        return KBDocumentActionResponse(**self.kb_store.kb_action(req.model_dump()))


@dataclass
class ServiceBundleImpl(ServiceBundle):
    writing: WritingService
    review: ReviewService
    help: HelpService
    merge: MergeService
    kb: KBService


def build_service_bundle() -> ServiceBundle:
    outline_agent = OutlineGenerator()
    heuristic_agent = HeuristicAgent()
    help_agent = HelpAgent()
    review_agent = ReviewAgent()
    merge_agent = MergeAgent()
    polish_agent = PolishAgent()
    restruct_agent = RestructAgent(outline_agent)
    kb_store = KBStore()
    return ServiceBundleImpl(
        writing=WritingServiceImpl(outline_agent, heuristic_agent, restruct_agent),
        review=ReviewServiceImpl(review_agent, polish_agent),
        help=HelpServiceImpl(help_agent),
        merge=MergeServiceImpl(merge_agent),
        kb=KBServiceImpl(kb_store),
    )
