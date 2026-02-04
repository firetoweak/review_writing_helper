from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class KBHit:
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


class KBClient:
    """In-process KB client for KBStore + Chroma (no FastAPI).

    Works with the updated KBStore design:
    - Chunks store only text + [IMAGE_x] tags
    - image_url map stored in docmeta json on disk via store.get_image_url_map()
    """

    def __init__(
        self,
        store: Any,  # KBStore
        *,
        # Server max-total-tokens=8192 -> keep a safety margin.
        query_max_tokens: int = 7800,
        # Over-limit queries will be split into token chunks (multi-query).
        query_chunk_tokens: int = 1536,
        query_chunk_overlap: int = 128,
        # safety buffer for per-piece truncation
        per_piece_safety_margin: int = 128,
    ) -> None:
        self.store = store
        self.query_max_tokens = int(query_max_tokens)
        self.query_chunk_tokens = int(query_chunk_tokens)
        self.query_chunk_overlap = int(query_chunk_overlap)
        self.per_piece_safety_margin = int(per_piece_safety_margin)

        self._tokenizer = None  # lazy

    # -------------------------
    # Write APIs (sugar)
    # -------------------------
    def index(
        self,
        *,
        project_id: str,
        document_id: str,
        text: str,
        image_url: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "action": "index",
            "projectId": project_id,
            "document_id": document_id,
            "image_url": image_url or {},
            "text": text,
        }
        return self.store.kb_action(payload)

    def delete(self, *, project_id: str, document_id: str) -> Dict[str, Any]:
        payload = {
            "action": "delete",
            "projectId": project_id,
            "document_id": document_id,
        }
        return self.store.kb_action(payload)

    # -------------------------
    # Image URL map
    # -------------------------
    def get_image_url_map(self, *, project_id: str, document_id: str) -> Dict[str, str]:
        """Read doc-level image_url map (docmeta)."""
        # new KBStore provides get_image_url_map(); fall back to _read_docmeta if needed
        if hasattr(self.store, "get_image_url_map"):
            return self.store.get_image_url_map(project_id=project_id, document_id=document_id)
        if hasattr(self.store, "_read_docmeta"):
            return self.store._read_docmeta(project_id, document_id)
        return {}

    def collect_image_url_maps(self, *, project_id: str, hits: List[KBHit]) -> Dict[str, Dict[str, str]]:
        """Collect doc_id -> image_url_map for docs that appear in hits."""
        doc_ids = {str(h.metadata.get("doc_id")) for h in hits if h.metadata and h.metadata.get("doc_id")}
        out: Dict[str, Dict[str, str]] = {}
        for doc_id in doc_ids:
            out[doc_id] = self.get_image_url_map(project_id=project_id, document_id=doc_id)
        return out

    # -------------------------
    # Tokenizer helpers (optional but recommended)
    # -------------------------
    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception:
            self._tokenizer = None
            return None

        try:
            tok = AutoTokenizer.from_pretrained(self.store.embedding_model, use_fast=True)
            self._tokenizer = tok
            return tok
        except Exception:
            self._tokenizer = None
            return None

    def _count_tokens(self, text: str) -> Optional[int]:
        tok = self._get_tokenizer()
        if tok is None:
            return None
        ids = tok.encode(text, add_special_tokens=False)
        return len(ids)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tok = self._get_tokenizer()
        if tok is None:
            # conservative char truncation for Chinese (~1-2 chars/token)
            return (text or "")[: max(0, int(max_tokens * 2))]
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        ids = ids[:max_tokens]
        return tok.decode(ids, skip_special_tokens=True)

    def _split_by_tokens(self, text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        tok = self._get_tokenizer()
        if tok is None:
            # fallback: conservative char slicing
            chunk_chars = max(200, int(chunk_tokens * 2))
            overlap_chars = max(0, int(overlap_tokens * 2))
            step = max(1, chunk_chars - overlap_chars)
            out: List[str] = []
            i, n = 0, len(text)
            while i < n:
                out.append(text[i : i + chunk_chars].strip())
                i += step
            return [x for x in out if x]

        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= chunk_tokens:
            return [text]

        step = max(1, chunk_tokens - max(0, overlap_tokens))
        out: List[str] = []
        i, n = 0, len(ids)
        while i < n:
            piece_ids = ids[i : i + chunk_tokens]
            out.append(tok.decode(piece_ids, skip_special_tokens=True).strip())
            i += step
        return [x for x in out if x]

    # -------------------------
    # Text cleanup (compat)
    # -------------------------
    @staticmethod
    def _strip_image_url_footer(text: str) -> str:
        """Remove legacy footer:
            ---
            [IMAGE_URL_MAP]
            ...
        (Your new KBStore no longer injects it, but keep backward compatibility.)
        """
        if not text:
            return text
        marker = "\n[IMAGE_URL_MAP]\n"
        idx = text.rfind(marker)
        if idx < 0:
            return text.strip()
        cut = text.rfind("\n---\n", 0, idx)
        if cut >= 0:
            return text[:cut].strip()
        return text[:idx].strip()

    # -------------------------
    # Core Query API (token-safe + returns image maps)
    # -------------------------
    def search(
        self,
        *,
        project_id: str,
        query_text: str,
        top_k: int = 8,
        where: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        include: Optional[List[str]] = None,
        long_query_strategy: str = "split",  # "split" or "truncate"
        strip_legacy_footer: bool = True,
        return_image_maps: bool = True,
    ) -> Tuple[List[KBHit], Dict[str, Dict[str, str]]]:
        """Vector search with token overflow protection.

        Returns:
            hits: List[KBHit]
            image_maps: Dict[doc_id, Dict[tag,url]] (empty if return_image_maps=False)
        """
        query_text = (query_text or "").strip()
        if not query_text:
            return [], {}
        col = self.store._get_collection(project_id)

        w = dict(where or {})
        if document_id:
            w["doc_id"] = document_id

        inc = include or ["documents", "metadatas", "distances"]

        # 1) over-limit detection
        tok_n = self._count_tokens(query_text)
        if tok_n is not None:
            over_limit = tok_n > self.query_max_tokens
        else:
            over_limit = len(query_text) > int(self.query_max_tokens * 2)

        # 2) build query pieces
        if not over_limit:
            pieces = [query_text]
        else:
            if long_query_strategy == "truncate":
                pieces = [self._truncate_to_tokens(query_text, self.query_max_tokens)]
            else:
                chunk_tokens = min(self.query_chunk_tokens, max(64, self.query_max_tokens - self.per_piece_safety_margin))
                pieces = self._split_by_tokens(query_text, chunk_tokens, self.query_chunk_overlap)
                safe_max = max(64, self.query_max_tokens - self.per_piece_safety_margin)
                pieces = [self._truncate_to_tokens(p, safe_max) for p in pieces if p.strip()]

        if not pieces:
            return [], {}

        # 3) multi-query and merge (best distance wins)
        per_piece_k = max(3, int(math.ceil(top_k / max(1, len(pieces))) * 3))
        best: Dict[str, KBHit] = {}

        for p in pieces:
            # embed via KBStore pipeline
            try:
                qvec = self.store._embed_texts([p])[0]
            except Exception:
                # last resort: shrink and retry once
                p2 = self._truncate_to_tokens(p, max(64, self.query_max_tokens // 2))
                qvec = self.store._embed_texts([p2])[0]

            res = col.query(
                query_embeddings=[qvec],
                n_results=max(1, per_piece_k),
                where=w if w else None,
                include=inc,
            )

            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0] if "distances" in res else [None] * len(ids)

            for i, _id in enumerate(ids):
                dist = dists[i] if i < len(dists) else None
                doc = docs[i] if i < len(docs) else ""
                meta = metas[i] if i < len(metas) else {}
                if strip_legacy_footer:
                    doc = self._strip_image_url_footer(doc)

                hit = KBHit(id=_id, document=doc, metadata=meta, distance=dist)

                if _id not in best:
                    best[_id] = hit
                else:
                    old = best[_id].distance
                    if old is None or (dist is not None and dist < old):
                        best[_id] = hit

        hits = sorted(best.values(), key=lambda h: (h.distance is None, h.distance))
        hits = hits[: max(1, int(top_k))]

        image_maps: Dict[str, Dict[str, str]] = {}
        if return_image_maps:
            image_maps = self.collect_image_url_maps(project_id=project_id, hits=hits)

        return hits, image_maps

    # -------------------------
    # Document fetch helpers
    # -------------------------
    def get_document_chunks(
        self,
        *,
        project_id: str,
        document_id: str,
        include: Optional[List[str]] = None,
        strip_legacy_footer: bool = True,
    ) -> List[KBHit]:
        """Fetch all chunks of a document, sorted by chunk_id if present."""
        col = self.store._get_collection(project_id)
        inc = include or ["documents", "metadatas"]

        res = col.get(where={"doc_id": document_id}, include=inc)

        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []

        hits: List[KBHit] = []
        for i in range(len(ids)):
            doc = docs[i] if i < len(docs) else ""
            if strip_legacy_footer:
                doc = self._strip_image_url_footer(doc)
            hits.append(KBHit(id=ids[i], document=doc, metadata=metas[i] if i < len(metas) else {}, distance=None))

        def _chunk_key(h: KBHit) -> Tuple[int, str]:
            cid = str(h.metadata.get("chunk_id") or "")
            try:
                return (int(cid), cid)
            except Exception:
                return (10**12, cid)

        hits.sort(key=_chunk_key)
        return hits

    def list_document_ids(self, *, project_id: str) -> List[str]:
        """List doc_ids in a project. May be heavy for huge collections."""
        col = self.store._get_collection(project_id)
        res = col.get(include=["metadatas"])
        metas = res.get("metadatas") or []
        doc_ids = sorted({str(m.get("doc_id")) for m in metas if m and m.get("doc_id")})
        return doc_ids
