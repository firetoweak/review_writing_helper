from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from typing import Optional

# 直接复用你项目里的实现（按你的工程结构调整 import 路径）
# - KBStore: 负责读写 / chunk / embeddings / chroma PersistentClient
# - KBClient: 负责 search + token-safe + 读取 docmeta(image_url map)
from services.agents.kb import KBStore
from infra.kb.kb_client import KBClient


# === 按你的部署改这里（默认值来自你当前实现）===
BASE_DIR = "/home/netzone22/liuhao/project/ai_writer_agent/chromaKB"
COLLECTION_NAME = "materials"
EMBEDDING_URL = "http://127.0.0.1:30025/v1/embeddings"
EMBEDDING_MODEL = "/home/netzone22/data/LLM/Qwen3-Embedding-8B"


def build_kb() -> KBClient:
    store = KBStore(
        base_dir=BASE_DIR,
        collection_name=COLLECTION_NAME,
        embedding_url=EMBEDDING_URL,
        embedding_model=EMBEDDING_MODEL,
    )
    return KBClient(store)


def cmd_list(kb: KBClient, project_id: str) -> None:
    doc_ids = kb.list_document_ids(project_id=project_id)
    print(f"[LIST] project_id={project_id} doc_ids={len(doc_ids)}")
    for d in doc_ids[:200]:
        print(" -", d)
    if len(doc_ids) > 200:
        print(f"... (only showing first 200, total={len(doc_ids)})")


def cmd_get(kb: KBClient, project_id: str, document_id: str) -> None:
    hits = kb.get_document_chunks(project_id=project_id, document_id=document_id)
    print(f"[GET] project_id={project_id} document_id={document_id} chunks={len(hits)}")
    for h in hits[:50]:
        meta = h.metadata or {}
        chunk_id = meta.get("chunk_id")
        has_images = meta.get("has_images")
        image_tags = meta.get("image_tags")
        text_head = (h.document or "").replace("\n", " ")[:160]
        print(f"- {h.id} chunk_id={chunk_id} has_images={has_images} image_tags={image_tags} text_head={text_head!r}")

    if len(hits) > 50:
        print(f"... (only showing first 50 chunks, total={len(hits)})")

    # doc-level image_url map（如果你上传时传了 image_url）
    img_map = kb.get_image_url_map(project_id=project_id, document_id=document_id)
    print(f"[DOCMETA] image_url_map keys={len(img_map)}")
    for k in list(img_map.keys())[:50]:
        print(" -", k, "=>", img_map[k])
    if len(img_map) > 50:
        print(f"... (only showing first 50 image tags, total={len(img_map)})")


def cmd_search(kb: KBClient, project_id: str, query: str, top_k: int, document_id: Optional[str]) -> None:
    hits, image_maps = kb.search(
        project_id=project_id,
        query_text=query,
        top_k=top_k,
        document_id=document_id,
        return_image_maps=True,
    )
    print(f"[SEARCH] project_id={project_id} top_k={top_k} doc_filter={document_id or '-'} hits={len(hits)}")
    for i, h in enumerate(hits, start=1):
        meta = h.metadata or {}
        doc_id = meta.get("doc_id")
        chunk_id = meta.get("chunk_id")
        dist = h.distance
        head = (h.document or "").replace("\n", " ")[:220]
        print(f"{i:02d}. doc_id={doc_id} chunk_id={chunk_id} dist={dist} head={head!r}")

    # 命中 doc 的 docmeta（doc_id -> {tag:url}）
    print(f"[SEARCH] image_maps docs={len(image_maps)}")
    for doc_id, m in list(image_maps.items())[:20]:
        keys = list(m.keys())[:10]
        print(f"- doc_id={doc_id} tags={len(m)} sample={keys}")
    if len(image_maps) > 20:
        print(f"... (only showing first 20 docs, total={len(image_maps)})")


def main():
    ap = argparse.ArgumentParser(description="Smoke test for KBStore/KBClient (ChromaKB)")
    ap.add_argument("--project", required=True, help="projectId used when indexing")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_list = sub.add_parser("list", help="list all document_id in the project")

    s_get = sub.add_parser("get", help="get all chunks of a document_id + docmeta image_url map")
    s_get.add_argument("--doc", required=True, help="document_id")

    s_search = sub.add_parser("search", help="vector search")
    s_search.add_argument("--q", required=True, help="query text")
    s_search.add_argument("--topk", type=int, default=8, help="top_k")
    s_search.add_argument("--doc", default=None, help="optional filter by document_id")

    args = ap.parse_args()
    kb = build_kb()

    if args.cmd == "list":
        cmd_list(kb, args.project)
    elif args.cmd == "get":
        cmd_get(kb, args.project, args.doc)
    elif args.cmd == "search":
        cmd_search(kb, args.project, args.q, args.topk, args.doc)
    else:
        raise SystemExit(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
