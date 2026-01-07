from typing import Dict


class KBClient:
    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, str]] = {}

    def index_document(self, document_id: str, file_url: str, filename: str) -> Dict[str, str]:
        self.documents[document_id] = {
            "document_id": document_id,
            "status": "indexed",
            "file_url": file_url,
            "filename": filename,
        }
        return {"document_id": document_id, "status": "indexed"}

    def delete_document(self, document_id: str) -> Dict[str, str]:
        self.documents.pop(document_id, None)
        return {"document_id": document_id, "status": "deleted"}
