from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Attachment(BaseModel):
    name: str
    mimeType: Optional[str] = None
    size: Optional[int] = None


class OutlineNode(BaseModel):
    nodeId: str
    level: int
    title: str
    keyPoint: str
    children: Optional[List["OutlineNode"]] = None


class Project(BaseModel):
    title: str
    idea: Optional[str] = None
    attachments: List[Attachment] = []


class ProjectOutlineRequest(BaseModel):
    task: str
    project: Project
    outlinePrompt: Optional[str] = None


class ProjectOutlineResponse(BaseModel):
    docGuide: str
    outline: List[OutlineNode]


class HistoryChild(BaseModel):
    nodeId: str
    title: str
    text: Optional[str] = None


class HistorySection(BaseModel):
    nodeId: str
    title: str
    level: int
    children: List[HistoryChild]


class Message(BaseModel):
    messageId: str
    role: str
    type: Optional[str] = None
    content: str
    attachmentPath: Optional[str] = None


class HeuristicCreateRequest(BaseModel):
    nodeId: str
    title: str
    text: Optional[str] = None
    task: str
    sessionId: Optional[str] = None
    historyText: Optional[List[HistorySection]] = None
    heuristicPrompt: Optional[str] = None
    messages: List[Message] = []
    stream: Optional[bool] = True


class HeuristicMessageRequest(BaseModel):
    nodeId: str
    title: str
    task: str
    sessionId: Optional[str] = None
    messages: List[Message]
    heuristicPrompt: Optional[str] = None
    stream: Optional[bool] = True


class HeuristicResponse(BaseModel):
    nodeId: str
    title: str
    task: str
    status: str
    assistantMessage: Message


class SectionText(BaseModel):
    nodeId: str
    level: int
    title: str
    text: str


class ReviewDetail(BaseModel):
    score: int
    summaryList: List[str]
    detail: str
    helpList: Optional[List[str]] = None


class SectionReviewRequest(BaseModel):
    task: str
    nodeId: str
    title: str
    text: List[SectionText]
    historyText: Optional[List[dict]] = None
    reviewpPrompt: Optional[str] = None
    industry: Optional[str] = None


class SectionReviewResponse(BaseModel):
    task: str
    title: str
    nodeId: str
    review: ReviewDetail


class HelpListResponse(BaseModel):
    task: str
    title: str
    nodeId: str
    helpList: List[str]


class ICanCreateRequest(BaseModel):
    task: str
    nodeId: str
    title: str
    text: List[SectionText]
    sessionId: str
    sessionText: str
    helpPrompt: Optional[str] = None
    messages: List[Message] = []
    stream: Optional[bool] = False


class ICanMessageRequest(BaseModel):
    task: str
    nodeId: str
    title: str
    sessionId: str
    messages: List[Message]
    helpPrompt: Optional[str] = None
    stream: Optional[bool] = False


class ICanResponse(BaseModel):
    task: str
    nodeId: str
    title: str
    sessionId: str
    assistantMessage: Message


class SessionMessage(BaseModel):
    messageId: str
    role: str
    content: str
    attachmentPath: Optional[List[str]] = None


class SessionItem(BaseModel):
    sesstionId: str
    messages: List[SessionMessage]


class MergeRequest(BaseModel):
    nodeId: str
    title: str
    task: str
    text: List[SectionText]
    sessionList: List[SessionItem]
    mergePrompt: Optional[str] = None
    historyText: Optional[List[dict]] = None
    stream: Optional[bool] = False


class MergeResponse(BaseModel):
    nodeId: str
    title: str
    task: str
    texts: List[SectionText]


class FullReviewSection(BaseModel):
    nodeId: str
    title: str
    level: int
    children: List[SectionText]


class FullReviewRequest(BaseModel):
    task: str
    fullText: List[FullReviewSection]
    reviews: Optional[List[dict]] = None
    fullReviewPrompt: Optional[str] = None
    stream: Optional[bool] = False


class FullReviewResponse(BaseModel):
    task: str
    fullReviewAns: str


class FullPolishRequest(BaseModel):
    task: str
    fullText: List[FullReviewSection]
    polishPrompt: Optional[str] = None
    stream: Optional[bool] = False


class FullPolishResponse(BaseModel):
    task: str
    newFullText: List[FullReviewSection]


class TextRestructRequest(BaseModel):
    task: str
    file_path: str
    restructPrompt: Optional[str] = None
    outlinePrompt: Optional[str] = None


class TextRestructResponse(BaseModel):
    docGuide: str
    outline: List[OutlineNode]
    fullText: List[FullReviewSection]


class KBDocumentActionRequest(BaseModel):
    action: str
    document_id: str
    file_url: Optional[str] = None
    filename: Optional[str] = None


class KBDocumentActionResponse(BaseModel):
    document_id: str
    status: str
