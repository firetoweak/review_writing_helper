from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from services.agents.kb import KBStore  # type: ignore
from services.kb_client import KBClient  # type: ignore



_IMAGE_TAG_RE = re.compile(r"\[IMAGE_(\d+)\]")


def strip_image_tags(text: str) -> str:
    """Remove [IMAGE_x] tags from text (used when VLM is not enabled)."""
    return _IMAGE_TAG_RE.sub("", text or "")


def _sorted_image_tags(img_map: Dict[str, str]) -> List[str]:
    def keyfn(t: str) -> Tuple[int, str]:
        m = _IMAGE_TAG_RE.fullmatch(t.strip())
        if not m:
            return (10**12, t)
        return (int(m.group(1)), t)

    return sorted([k.strip() for k in (img_map or {}).keys() if k and k.strip()], key=keyfn)


def _flatten_doc_image_maps(image_maps: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """doc_id -> {tag:url}  =>  {tag:url} (tag expected globally unique)."""
    out: Dict[str, str] = {}
    for _, m in (image_maps or {}).items():
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip() and k.strip() not in out:
                out[k.strip()] = v.strip()
    return out


class SessionKB:
    """Session-scoped KB:

    - projectId = session:{sessionId}
    - Index incoming attachments into KB (NOT into prompt)
    - Retrieve topK chunks per turn and inject only those into prompt
    """

    TOP_K = 8
    MAX_HIT_CHARS = 900
    MAX_ATTACHMENT_CHARS = 200_000  # safety: avoid pathological huge attachment text

    def __init__(self) -> None:
        self.store = KBStore()
        self.client = KBClient(self.store)

    @staticmethod
    def project_id(session_id: str) -> str:
        sid = (session_id or "").strip() or "default"
        return f"session:{sid}"

    def index_incoming_attachments(
        self, *, session_id: str, incoming_messages: List[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Index attachments from this turn; return (indexed_doc_ids, current_image_map)."""
        project_id = self.project_id(session_id)
        indexed: List[str] = []
        current_image_map: Dict[str, str] = {}

        for m in incoming_messages or []:
            role = (m.get("role") or "").strip().lower()
            if role not in ("user", "human"):
                continue

            mid = str(m.get("messageId") or "m_u_0").strip()
            atts = m.get("attachments")
            if not isinstance(atts, list) or not atts:
                continue

            for i, att in enumerate(atts, start=1):
                if not isinstance(att, dict):
                    continue

                text = (att.get("text") or "").strip()
                if len(text) > self.MAX_ATTACHMENT_CHARS:
                    text = text[: self.MAX_ATTACHMENT_CHARS].rstrip() + "…"

                iu = att.get("image_url") or {}
                image_url: Dict[str, str] = {}
                if isinstance(iu, dict):
                    for k, v in iu.items():
                        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                            image_url[k.strip()] = v.strip()

                # If only images are provided, keep at least tags so "this image" can be retrieved.
                if (not text) and image_url:
                    text = " ".join(_sorted_image_tags(image_url))

                if not text and not image_url:
                    continue

                doc_id = f"att:{mid}:{i}"
                try:
                    self.client.index(project_id=project_id, document_id=doc_id, text=text, image_url=image_url)
                    indexed.append(doc_id)
                except Exception:
                    # Don't fail the whole request if KB indexing fails.
                    continue

                for k, v in image_url.items():
                    if k and v and k not in current_image_map:
                        current_image_map[k] = v

        return indexed, current_image_map

    def retrieve(
        self, *, session_id: str, query_text: str, top_k: int = TOP_K, allow_images: bool = True
    ) -> Tuple[str, Dict[str, str]]:
        project_id = self.project_id(session_id)
        query_text = (query_text or "").strip()
        if not query_text:
            return "", {}

        try:
            hits, image_maps = self.client.search(
                project_id=project_id,
                query_text=query_text,
                top_k=int(top_k),
                where=None,
                long_query_strategy="split",
                return_image_maps=bool(allow_images),
            )
        except Exception:
            return "", {}

        flat_img = _flatten_doc_image_maps(image_maps) if allow_images else {}

        blocks: List[str] = []
        for h in hits or []:
            doc = (getattr(h, "document", "") or "").strip()
            if not doc:
                continue
            if len(doc) > self.MAX_HIT_CHARS:
                doc = doc[: self.MAX_HIT_CHARS].rstrip() + "…"

            meta = getattr(h, "metadata", {}) or {}
            doc_id = str(meta.get("doc_id") or "?")
            chunk_id = str(meta.get("chunk_id") or "?")
            dist = getattr(h, "distance", None)
            head = f"[doc={doc_id} chunk={chunk_id}]" if dist is None else f"[doc={doc_id} chunk={chunk_id} dist={dist:.4f}]"
            blocks.append(head + "\n" + doc)

        return "\n\n".join(blocks).strip(), flat_img
