from __future__ import annotations

from typing import List, Optional, Union, Literal, Dict, Any

from pydantic import BaseModel, Field, model_validator, field_validator

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
    polishPrompt: Optional[str] = None


class TextList(BaseModel):
    sectionId: str
    sectionTitle: str
    text: str
    image_url: Dict[str, str] = Field(default_factory=dict)

class HistoryTextList(BaseModel):
    chapterId: str        
    chapterTitle: str
    children: List[TextList] = Field(default_factory=list)


class IndustryRequest(BaseModel):
    projectId: Optional[str] = None
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
    children: Optional[List["OutlineNode"]] = None


class ProjectOutlineRequest(BaseModel):
    projectId: Optional[str] = None
    title: str
    idea: Optional[str] = None
    fullWriteRule: str
    industry: Optional[str] = None
    prompt: Optional[Prompts] = None

class ProjectOutlineResponse(BaseModel):
    docGuide: str
    outline: List[OutlineNode]


class ChapterKeyPointRequest(BaseModel):
    projectId: Optional[str] = None
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
    projectId: Optional[str] = None
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

class FullOutline(BaseModel):
    nodeId: str
    level: int
    title: str
    keyPoint: Optional[str] = None
    chapterWriteRule: Optional[str] = None
    sectionWriteRule: Optional[str] = None
    children: Optional[List[FullOutline]] = None

class OutlineMappingRequest(BaseModel):
    projectId: str
    title: str
    idea: str
    outline: List[FullOutline] = Field(default_factory=list)


class OutlineMappingResponse(BaseModel):
    sessionId: str
    neighbors: Dict[str, Any]

class Attachment(BaseModel):
    text: str
    image_url: Dict[str, str] = Field(default_factory=dict)

class Message(BaseModel):
    messageId: str
    role: str
    type: Optional[str] = None
    content: str
    attachments: Optional[List[Attachment]] = None


class HeuristicCreateRequest(BaseModel):
    projectId: Optional[str] = None
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
    projectId: Optional[str] = None
    messages: List[Message]
    sessionId: str
    stream: Optional[bool] = True

class HeuristicResponse(BaseModel):
    sessionId: str
    status: Literal["ask", "draft"]
    assistantMessage: Message


class ReviewDetail(BaseModel):
    score: Optional[Any] = None
    evaluate: Optional[str] = None
    suggestion: List[str] = Field(default_factory=list)
    to_do_list: Optional[List[str]] = None


class SectionReviewRequest(BaseModel):
    projectId: Optional[str] = None
    textList: List[TextList] = Field(default_factory=list)
    sectionWriteRule: str
    sectionReviewRule: str
    industry: Optional[str] = None
    historyTextList: Optional[List[HistoryTextList]] = None
    title: str
    sectionTitle: str
    prompt: Optional[Prompts] = None


class ReviewResponse(BaseModel):
    review: ReviewDetail


class ChapterReviewRequest(BaseModel):
    projectId: Optional[str] = None
    textList: List[TextList] = Field(default_factory=list)
    chapterWriteRule: str
    chapterReviewRule: str
    industry: Optional[str] = None
    historyTextList: Optional[List[HistoryTextList]] = None
    title: str
    chapterTitle: str
    prompt: Optional[Prompts] = None


class ReviewResponse(BaseModel):
    review: ReviewDetail

class ICanCreateRequest(BaseModel):
    projectId: Optional[str] = None
    sessionId: str
    textList: List[TextList] = Field(default_factory=list)
    review: ReviewDetail
    chapterWriteRule: Optional[str] = None
    sectionWriteRule: Optional[str] = None
    helpText: str
    prompt: Optional[Prompts] = None
    stream: Optional[bool] = True


class ICanMessageRequest(BaseModel):
    projectId: Optional[str] = None
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
    projectId: Optional[str] = None
    chapterWriteRule: Optional[str] = None
    sectionWriteRule: Optional[str] = None
    textList: List[TextList] = Field(default_factory=list)
    sessionList: List[SessionItem] = Field(default_factory=list)
    historyTextList: Optional[List[HistoryTextList]] = None
    review: ReviewDetail
    industry: Optional[str] = None
    prompt: Optional[Prompts] = None


class MergeResponse(BaseModel):
    textList: List[TextList] = Field(default_factory=list)


class ChapterReview(BaseModel):
    chapterId: str
    chapterTitle: str   
    review: ReviewDetail

class FullReviewRequest(BaseModel):
    projectId: Optional[str] = None
    title: str
    reviews: List[ChapterReview] = Field(default_factory=list)
    fullReviewText: str
    prompt: Optional[Prompts] = None


class FullReviewResponse(BaseModel):
    fullReviewAns: str


class FullPolishRequest(BaseModel):
    projectId: Optional[str] = None
    fullText: List[HistoryTextList]
    prompt: Optional[Prompts] = None
    stream: Optional[bool] = False
    

class FullPolishResponse(BaseModel):
    task: str
    newFullText: List[HistoryTextList]


class TextRestructRequest(BaseModel):
    projectId: Optional[str] = None
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
    image_url: Dict[str, str] = Field(default_factory=dict)
    text: Optional[str] = None


class KBDocumentActionResponse(BaseModel):
    document_id: str
    status: Literal["indexed", "deleted"]



class FloatRequest(BaseModel):
    projectId: Optional[str] = None
    sectionTitle: str
    textList: List[TextList] = Field(default_factory=list)
    targetText: Optional[str] = None
    userInput: Optional[str] = None
    prompt: Optional[Prompts] = None

class FloatResponse(BaseModel):
    floatText: str
