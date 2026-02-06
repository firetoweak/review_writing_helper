from __future__ import annotations

import os
import time
import shutil
import json
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
import threading

import fcntl  # Linux/Unix
import requests

import chromadb
from chromadb.config import Settings
import re
from bisect import bisect_right


class KBStore:
    """Project-scoped ChromaKB store.

    ✅ Request schema for indexing:
    {
        "action": "index",
        "projectId": "xxx-xxxx",
        "document_id": "doc_456",
        "image_url": {"[IMAGE_1]": "https://...png", ...},
        "text": "[IMAGE_1]aaaa[IMAGE_2]..."
    }

    Storage design:
    - `text` is the canonical document content (ONLY text + [IMAGE_x] tags).
    - `image_url` is stored as doc-level metadata on disk:
        {base_dir}/{projectId}/docmeta/{document_id}.json
      (so it won't pollute embeddings / retrieval chunks).
    - Chroma chunks store minimal image info in metadata: has_images + image_tags.
    """

    _IMAGE_TAG_RE = re.compile(r"\[IMAGE_\d+\]")

    def __init__(
        self,
        *,
        base_dir: str = "/home/netzone22/liuhao/project/ai_writer_agent/chromaKB",
        collection_name: str = "materials",
        chunk_size: int = 800,
        overlap: int = 120,
        embedding_url: str = "http://127.0.0.1:30025/v1/embeddings",
        embedding_model: str = "/home/netzone22/data/LLM/Qwen3-Embedding-8B",
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        timeout_s: int = 120,
        embedding_batch_size: int = 64,
    ) -> None:
        self.base_dir = base_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.embedding_url = embedding_url
        self.embedding_model = embedding_model
        self.embed_fn = embed_fn
        self.timeout_s = timeout_s
        self.embedding_batch_size = max(1, int(embedding_batch_size))

        os.makedirs(self.base_dir, exist_ok=True)

        self._session = requests.Session()
        self._http_lock = threading.Lock()  # avoid requests.Session concurrency issues

    # ----------------- Public API -----------------

    def kb_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        action = payload.get("action")
        project_id = payload.get("projectId")
        doc_id = payload.get("document_id")

        if action not in ("index", "delete"):
            raise ValueError(f"Unsupported action: {action}")
        if not project_id:
            raise ValueError("projectId is required")
        if not doc_id:
            raise ValueError("document_id is required")

        with self._project_write_lock(project_id):
            if action == "delete":
                self._delete_doc(project_id, doc_id)
                self._delete_docmeta(project_id, doc_id)
                return {"document_id": doc_id, "status": "deleted"}

            # ---- action == "index" ----
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text is required (non-empty string) when action=index")

            image_map = payload.get("image_url") or {}
            if image_map is not None and not isinstance(image_map, dict):
                raise ValueError('image_url must be a dict like {"[IMAGE_1]": "https://..."}')

            # ✅ DO NOT append image_url to the end of text anymore.
            final_text = text

            # idempotent + full replace
            self._delete_doc(project_id, doc_id)
            self._delete_docmeta(project_id, doc_id)

            # ✅ store doc-level image_url map on disk (not in embeddings)
            self._write_docmeta(project_id, doc_id, image_map)

            # index chunks
            self._index_doc(project_id, doc_id, final_text, image_map=image_map)

            return {"document_id": doc_id, "status": "indexed"}

    def purge_project(self, project_id: str) -> Dict[str, Any]:
        if not project_id:
            raise ValueError("project_id is required")

        with self._project_write_lock(project_id):
            pdir = self._project_dir(project_id)
            if os.path.exists(pdir):
                shutil.rmtree(pdir)
            return {"projectId": project_id, "status": "purged"}

    def get_image_url_map(self, *, project_id: str, document_id: str) -> Dict[str, str]:
        """Programmatic read of doc-level image_url map."""
        with self._project_write_lock(project_id):
            return self._read_docmeta(project_id, document_id)

    # ----------------- Locking -----------------

    @contextmanager
    def _project_write_lock(self, project_id: str):
        pdir = self._project_dir(project_id)
        os.makedirs(pdir, exist_ok=True)
        lock_path = os.path.join(pdir, ".lock")

        fp = open(lock_path, "a+")
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
            finally:
                fp.close()

    # ----------------- Chroma -----------------

    def _get_collection(self, project_id: str):
        persist_dir = os.path.join(self._project_dir(project_id), "chroma")
        os.makedirs(persist_dir, exist_ok=True)

        try:
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        except ValueError as exc:
            if "tenant" not in str(exc):
                raise
            client = self._bootstrap_default_tenant(persist_dir, exc)
        return client.get_or_create_collection(self.collection_name)

    def _bootstrap_default_tenant(self, persist_dir: str, exc: Exception):
        if not self._is_dir_empty(persist_dir):
            raise exc

        print(
            f"[KBStore] Chroma tenant missing in empty directory. "
            f"Bootstrapping default tenant at: {persist_dir}"
        )
        try:
            legacy_settings = Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=persist_dir,
            )
        except TypeError:
            legacy_settings = Settings(anonymized_telemetry=False)
        try:
            client = chromadb.Client(settings=legacy_settings)
            client.get_or_create_collection(self.collection_name)
            return client
        except Exception:
            raise exc

    def _is_dir_empty(self, path: str) -> bool:
        try:
            return not any(os.scandir(path))
        except FileNotFoundError:
            return True

    # ----------------- CRUD internals -----------------

    def _delete_doc(self, project_id: str, doc_id: str) -> None:
        col = self._get_collection(project_id)
        col.delete(where={"doc_id": doc_id})

    def _index_doc(self, project_id: str, doc_id: str, text: str, *, image_map: Dict[str, str]) -> None:
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("Empty document after chunking")

        vectors = self._embed_texts(chunks)
        if len(vectors) != len(chunks):
            raise RuntimeError(f"Embedding size mismatch: {len(vectors)} vs {len(chunks)}")

        now = int(time.time())
        ids = [f"{doc_id}:{i:05d}" for i in range(len(chunks))]

        # store minimal image info in metadata (avoid huge url strings)
        image_tags = sorted([str(k) for k in (image_map or {}).keys()])

        metadatas = [
            {
                "project_id": project_id,
                "doc_id": doc_id,
                "chunk_id": f"{i:05d}",
                "updated_at": now,
                "has_images": bool(image_tags),
                "image_tags": ",".join(image_tags) if image_tags else "",
            }
            for i in range(len(chunks))
        ]

        col = self._get_collection(project_id)
        col.upsert(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)

    # ----------------- Helpers: project/docmeta -----------------

    def _project_dir(self, project_id: str) -> str:
        safe = project_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self.base_dir, safe)

    def _docmeta_dir(self, project_id: str) -> str:
        d = os.path.join(self._project_dir(project_id), "docmeta")
        os.makedirs(d, exist_ok=True)
        return d

    def _docmeta_path(self, project_id: str, doc_id: str) -> str:
        safe_doc = doc_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self._docmeta_dir(project_id), f"{safe_doc}.json")

    def _normalize_image_map(self, image_map: Dict[str, Any]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for k, v in (image_map or {}).items():
            if not k or v is None:
                continue
            ks = str(k).strip()
            vs = str(v).strip()
            if ks and vs:
                out[ks] = vs
        return out

    def _write_docmeta(self, project_id: str, doc_id: str, image_map: Dict[str, Any]) -> None:
        data = self._normalize_image_map(image_map)
        path = self._docmeta_path(project_id, doc_id)

        # if empty => keep storage clean
        if not data:
            self._delete_docmeta(project_id, doc_id)
            return

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # atomic replace

    def _read_docmeta(self, project_id: str, doc_id: str) -> Dict[str, str]:
        path = self._docmeta_path(project_id, doc_id)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _delete_docmeta(self, project_id: str, doc_id: str) -> None:
        path = self._docmeta_path(project_id, doc_id)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    # ----------------- Chunk / embed -----------------

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text with a simple sliding window, but **protect image tags**.

        We ensure placeholders like "[IMAGE_12]" are never split across chunk boundaries.
        - If a chunk boundary (start/end) falls inside an image tag span, we expand the
          boundary to include the full tag.
        - We keep iteration progress based on the *original* window start (base_i) so
          we never get stuck even if we move the actual chunk start backward.
        """
        out: List[str] = []
        n = len(text)
        if n <= 0:
            return []

        step = max(1, self.chunk_size - self.overlap)

        spans = [(m.start(), m.end()) for m in self._IMAGE_TAG_RE.finditer(text)]
        starts = [s for s, _ in spans]

        def _span_covering(pos: int) -> Optional[tuple[int, int]]:
            if not spans:
                return None
            idx = bisect_right(starts, pos) - 1
            if idx >= 0:
                s, e = spans[idx]
                if s < pos < e:
                    return (s, e)
            return None

        i = 0
        while i < n:
            base_i = i
            start = base_i
            end = min(base_i + self.chunk_size, n)

            sp = _span_covering(start)
            if sp is not None:
                start = sp[0]
            sp = _span_covering(end)
            if sp is not None:
                end = sp[1]

            if end <= start:
                end = min(start + self.chunk_size, n)

            out.append(text[start:end])
            i = base_i + step

        return [c.strip() for c in out if c.strip()]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.embed_fn is not None:
            return self.embed_fn(texts)

        batch_size = self.embedding_batch_size
        all_vecs: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"model": self.embedding_model, "input": batch}

            with self._http_lock:
                resp = self._session.post(
                    self.embedding_url,
                    json=payload,
                    timeout=self.timeout_s,
                )

            resp.raise_for_status()
            data = resp.json()

            items = data.get("data")
            if not isinstance(items, list):
                raise RuntimeError(f"Unexpected embedding response: {data}")

            # keep stable order if server returns {index, embedding}
            if items and isinstance(items[0], dict) and "index" in items[0]:
                items = sorted(items, key=lambda x: x.get("index", 0))

            vecs = [it["embedding"] for it in items]
            if len(vecs) != len(batch):
                raise RuntimeError(f"Embedding size mismatch: {len(vecs)} vs {len(batch)}")

            all_vecs.extend(vecs)

        return all_vecs
