from __future__ import annotations

from typing import Any
from starlette.concurrency import run_in_threadpool

from models.schemas import *

class KBService:
    def __init__(self, *, store: Any):
        self.store = store

    async def action(self, req: dict) -> FloatResponse:
        data = await run_in_threadpool(self.store.kb_action, req)
        return FloatResponse(**data)