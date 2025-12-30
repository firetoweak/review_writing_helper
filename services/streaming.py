from __future__ import annotations

import json
from typing import AsyncGenerator, Dict


async def stream_json(payload: Dict) -> AsyncGenerator[str, None]:
    yield json.dumps(payload, ensure_ascii=False) + "\n"
