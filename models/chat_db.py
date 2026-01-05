from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Message:
    message_id: str
    role: str
    content: str
    type: Optional[str] = None
    attachment_path: Optional[str] = None


@dataclass
class HeuristicSession:
    node_id: str
    title: str
    messages: List[Message] = field(default_factory=list)


@dataclass
class HelpSession:
    session_id: str
    node_id: str
    title: str
    session_text: str
    messages: List[Message] = field(default_factory=list)


class ChatDB:
    def __init__(self) -> None:
        self.heuristic_sessions: Dict[str, HeuristicSession] = {}
        self.help_sessions: Dict[str, HelpSession] = {}

    def get_heuristic(self, node_id: str) -> Optional[HeuristicSession]:
        return self.heuristic_sessions.get(node_id)

    def save_heuristic(self, session: HeuristicSession) -> None:
        self.heuristic_sessions[session.node_id] = session

    def get_help(self, session_id: str) -> Optional[HelpSession]:
        return self.help_sessions.get(session_id)

    def save_help(self, session: HelpSession) -> None:
        self.help_sessions[session.session_id] = session
