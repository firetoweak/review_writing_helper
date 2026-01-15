from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field, model_validator


class Attachment(BaseModel):
    name: str
    mimeType: Optional[str] = None
    size: Optional[int] = None


class Prompts(BaseModel):
    industryPrompt: Optional[str] = None
    outlinePrompt: Optional[str] = None
    chapterKeypointPrompt: Optional[str] = None
    sectionKeypointPrompt: Optional[str] = None
    heuristicWritingPrompt: Optional[str] = None
    heuristicCorrectPrompt: Optional[str] = None
    sectionReviewPrompt: Optional[str] = None
    chapterReviewPrompt: Optional[str] = None
    helpPrompt: Optional[str] = None
    mergePrompt: Optional[str] = None
    mergeCorrectPrompt: Optional[str] = None
    fullReviewPrompt: Optional[str] = None
    floatPrompt: Optional[str] = None


class TextList(BaseModel):
    sectionId: str
    sectionTitle: str
    text: str
    attachments_url: Optional[Dict[str, str]] = None


class HistoryTextList(BaseModel):
    chapterId: str
    chapterTitle: str
    children: List[TextList] = Field(default_factory=list)


class IndustryRequest(BaseModel):
    title: str
    idea: Optional[str] = None
    industryNameList: Optional[List[str]] = None
    prompt: Optional[Prompts] = None


class IndustryResponse(BaseModel):
    industryName: Optional[str] = None


class OutlineNode(BaseModel):
    nodeId: Optional[str] = None
    level: Optional[int] = None
    title: Optional[str] = None
    keyPoint: Optional[str] = None
    children: Optional[List["OutlineNode"]] = None


class ProjectOutlineRequest(BaseModel):
    title: str
    idea: Optional[str] = None
    fullWriteRule: str
    industry: Optional[str] = None
    prompt: Optional[Prompts] = None


class ProjectOutlineResponse(BaseModel):
    docGuide: List[dict]
    outline: List[OutlineNode]


class OutlineMappingRequest(BaseModel):
    sessionId: str
    title: str
    idea: Optional[str] = None
    outline: List[OutlineNode]


class OutlineMappingResponse(BaseModel):
    sessionId: str
    neighbors: Dict[str, Any]


class ChapterKeyPointRequest(BaseModel):
    title: str
    idea: Optional[str] = None
    fullWriteRule: Optional[str] = None
    chapterId: str
    industry: Optional[str] = None
    prompt: Optional[Prompts] = None


class ChapterKeyPointResponse(BaseModel):
    chapterId: str
    keyPoint: str


class SectionKeyPointRequest(BaseModel):
    title: str
    idea: Optional[str] = None
    sectionWriteRule: str
    chapterId: str
    sectionId: str
    industry: Optional[str] = None
    outline: Optional[List[OutlineNode]] = None
    prompt: Optional[Prompts] = None


class SectionKeyPointResponse(BaseModel):
    sectionId: str
    keyPoint: str


class Message(BaseModel):
    messageId: str
    role: str
    type: Optional[str] = None
    content: str
    attachmentPath: Optional[str] = None


class HeuristicCreateRequest(BaseModel):
    sectionWriteRule: str
    textList: List[TextList] = Field(default_factory=list)
    historyTextList: Optional[List[HistoryTextList]] = None
    sectionReviewRule: Optional[str] = None
    industry: Optional[str] = None
    title: str
    idea: Optional[str] = None
    sessionId: str
    sectionTitle: str
    prompt: Optional[Prompts] = None


class HeuristicMessageRequest(BaseModel):
    Messages: List[Message]
    sessionId: str
    stream: str


class HeuristicResponse(BaseModel):
    sessionId: str
    status: Literal["ask", "draft"]
    assistantMessage: Message


class ReviewDetail(BaseModel):
    score: Union[int, str]
    evaluate: Optional[str] = None
    suggestion: Optional[str] = None
    to_do_list: Optional[List[str]] = None


class SectionReviewRequest(BaseModel):
    textList: List[TextList] = Field(default_factory=list)
    sectionWriteRule: str
    sectionReviewRule: str
    industry: Optional[str] = None
    historyTextList: Optional[List[HistoryTextList]] = None
    title: str
    sectionTitle: str
    prompt: Optional[Prompts] = None


class SectionReviewResponse(BaseModel):
    review: ReviewDetail


class ChapterReviewRequest(BaseModel):
    textList: List[TextList] = Field(default_factory=list)
    chapterWriteRule: str
    chapterReviewRule: str
    industry: Optional[str] = None
    historyTextList: Optional[List[HistoryTextList]] = None
    title: str
    chapterTitle: str
    prompt: Optional[Prompts] = None


class ChapterReviewResponse(BaseModel):
    review: ReviewDetail


class WriteRuleIn(BaseModel):
    chapterWriteRule: Optional[str] = None
    sectionWriteRule: Optional[str] = None

    @model_validator(mode="after")
    def exactly_one(cls, m):
        has_ch = m.chapterWriteRule is not None and m.chapterWriteRule != ""
        has_se = m.sectionWriteRule is not None and m.sectionWriteRule != ""
        if has_ch == has_se:
            raise ValueError("必须且只能传入 chapterWriteRule 或 sectionWriteRule 其中一个")
        return m


class ICanCreateRequest(BaseModel):
    sessionId: str
    textList: List[TextList] = Field(default_factory=list)
    review: ReviewDetail
    writeRule: WriteRuleIn
    helpText: str
    prompt: Optional[Prompts] = None


class ICanMessageRequest(BaseModel):
    sessionId: str
    messages: List[Message] = Field(default_factory=list)
    stream: Optional[bool] = True


class ICanResponse(BaseModel):
    sessionId: str
    assistantMessage: Message


class SessionItem(BaseModel):
    sessionId: str
    messages: List[Message] = Field(default_factory=list)


class MergeRequest(BaseModel):
    writeRule: WriteRuleIn
    textList: List[TextList] = Field(default_factory=list)
    sessionList: List[SessionItem] = Field(default_factory=list)
    historyTextList: Optional[List[HistoryTextList]] = None
    review: ReviewDetail
    industry: Optional[str] = None
    prompt: Optional[Prompts] = None


class MergeResponse(BaseModel):
    text: str


class ChapterReview(BaseModel):
    chapterId: str
    chapterTitle: str
    review: ReviewDetail


class FullReviewRequest(BaseModel):
    title: str
    reviews: List[ChapterReview] = Field(default_factory=list)
    fullReviewText: str
    prompt: Optional[Prompts] = None


class FullReviewResponse(BaseModel):
    fullReviewAns: str


class FullPolishRequest(BaseModel):
    task: str
    fullText: List[HistoryTextList]
    polishPrompt: Optional[str] = None
    stream: Optional[bool] = False


class FullPolishResponse(BaseModel):
    task: str
    newFullText: List[HistoryTextList]


class TextRestructRequest(BaseModel):
    task: str
    file_path: str
    restructPrompt: Optional[str] = None
    outlinePrompt: Optional[str] = None
    industry: Optional[str] = None


class TextRestructResponse(BaseModel):
    docGuide: str
    outline: List[OutlineNode]
    fullText: List[HistoryTextList]


class KBDocumentActionRequest(BaseModel):
    action: str
    projectId: str = Field(..., description="写作任务Id")
    document_id: str
    file_url: Optional[str] = None
    filename: Optional[str] = None


class KBDocumentActionResponse(BaseModel):
    document_id: str
    status: Literal["indexed", "deleted"]


class FloatRequest(BaseModel):
    sectionTitle: str
    TextList: List[TextList] = Field(default_factory=list)
    targetText: Optional[str] = None
    userInput: Optional[str] = None
    prompt: Optional[Prompts] = None


class FloatResponse(BaseModel):
    floatText: str
