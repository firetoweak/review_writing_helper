from __future__ import annotations

from typing import Dict


class KBStore:
    def __init__(self) -> None:
        self._docs: Dict[str, Dict[str, str]] = {}

    def kb_action(self, payload: Dict) -> Dict:
        action = payload.get("action")
        doc_id = payload.get("document_id")
        if action == "index":
            self._docs[doc_id] = {
                "document_id": doc_id,
                "status": "indexed",
                "file_url": payload.get("file_url", ""),
                "filename": payload.get("filename", ""),
            }
            return {"document_id": doc_id, "status": "indexed"}
        if action == "delete":
            if doc_id in self._docs:
                self._docs.pop(doc_id, None)
            return {"document_id": doc_id, "status": "deleted"}
        return {"document_id": doc_id, "status": "failed"}
