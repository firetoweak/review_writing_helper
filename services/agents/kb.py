from __future__ import annotations

from typing import Dict


class KBStore:
    def __init__(self) -> None:
        self._docs: Dict[str, Dict[str, str]] = {}

    def kb_action(self, payload: Dict) -> Dict:
        action = payload.get("action")
        doc_id = payload.get("document_id")
        project_id = payload.get("projectId")
        key = f"{project_id}:{doc_id}"
        if action == "index":
            self._docs[key] = {
                "document_id": doc_id,
                "status": "indexed",
                "file_url": payload.get("file_url", ""),
                "filename": payload.get("filename", ""),
                "projectId": project_id,
            }
            return {"document_id": doc_id, "status": "indexed"}
        if action == "delete":
            if key in self._docs:
                self._docs.pop(key, None)
            return {"document_id": doc_id, "status": "deleted"}
        return {"document_id": doc_id, "status": "failed"}
