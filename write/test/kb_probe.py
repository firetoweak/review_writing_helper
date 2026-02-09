import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.agents.kb import KBStore

BASE_DIR = "/home/netzone22/liuhao/project/ai_writer_agent/chromaKB"
COLLECTION_NAME = "materials"
EMBEDDING_URL = "http://127.0.0.1:30025/v1/embeddings"
EMBEDDING_MODEL = "/home/netzone22/data/LLM/Qwen3-Embedding-8B"

store = KBStore(
    base_dir=BASE_DIR,
    collection_name=COLLECTION_NAME,
    embedding_url=EMBEDDING_URL,
    embedding_model=EMBEDDING_MODEL,
)

col = store._collection  # 你的 KBStore 里一般会持有 chroma collection

print("[COUNT]", col.count())

# 随便取几条看看 meta 字段（不加 where）
res = col.get(limit=5, include=["metadatas", "documents", "ids"])
for i in range(len(res["ids"])):
    print("----")
    print("id:", res["ids"][i])
    print("meta:", res["metadatas"][i])
    doc = (res["documents"][i] or "").replace("\n", " ")[:160]
    print("doc_head:", doc)
