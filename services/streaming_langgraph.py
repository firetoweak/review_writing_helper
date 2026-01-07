from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional


def _chunk_text(chunk: Any) -> Optional[str]:
    if chunk is None:
        return None
    if isinstance(chunk, dict):
        return chunk.get("content") or chunk.get("text")
    return getattr(chunk, "content", None) or getattr(chunk, "text", None)


async def graph_to_ndjson_tokens(graph, graph_input: Dict[str, Any]) -> AsyncIterator[str]:
    async for event in graph.astream_events(graph_input, version="v2"):
        name = event.get("event", "")
        data = event.get("data", {}) or {}

        if name == "on_chat_model_stream":
            token = _chunk_text(data.get("chunk"))
            if token:
                yield json.dumps({"type": "token", "text": token}, ensure_ascii=False) + "\n"
        elif name == "on_chain_stream":
            token = _chunk_text(data.get("chunk"))
            if token:
                yield json.dumps({"type": "token", "text": token}, ensure_ascii=False) + "\n"

    yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"
