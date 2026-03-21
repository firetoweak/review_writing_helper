from __future__ import annotations

import json
from typing import AsyncGenerator, AsyncIterator, Dict


async def stream_json(payload: Dict) -> AsyncGenerator[str, None]:
    yield json.dumps(payload, ensure_ascii=False) + "\n"


async def stream_tokens(tokens: AsyncIterator[str]) -> AsyncGenerator[str, None]:
    async for token in tokens:
        yield json.dumps({"type": "token", "text": token}, ensure_ascii=False) + "\n"
    yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
