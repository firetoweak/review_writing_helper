from __future__ import annotations

from typing import Any, AsyncIterator

from models.schemas import *

class ChatService:
    def __init__(self, *, heuristic_agent: Any, help_agent: Any):
        self.heuristic_agent = heuristic_agent
        self.help_agent = help_agent


    async def heuristic_stream(self, payload: dict) -> AsyncIterator[str]:
        async for line in self.heuristic_agent.stream(payload):
            yield line

    async def help_stream(self, payload: dict):
        async for line in self.help_agent.stream(payload):
            yield line