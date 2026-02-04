from __future__ import annotations
from contextlib import asynccontextmanager
import os
from fastapi import FastAPI

from infra.persistence.checkpointer_pg import (
    close_checkpointer,
    ensure_checkpoint_tables,
)
from config import load_config
from services.agents.heuristic import HeuristicAgent
from services.agents.help import HelpAgent

from services.agents.merge import MergeAgent
from services.agents.outline import OutlineAgent

from services.agents.polish import PolishAgent
from services.agents.review import ReviewAgent
from services.agents.floatAns import FloatAgent
from services.agents.industry import IndustryAgent

from services.chat_service import ChatService
from services.others_service import OthersService
from services.kb_service import KBService

from services.agents.kb import KBStore
from infra.kb.kb_client import KBClient
from app.container import AppContext


_KB_BASE_DIR = "/home/netzone22/liuhao/project/ai_writer_agent/chromaKB"
_KB_COLLECTION = "materials"
_EMBEDDING_URL = "http://127.0.0.1:30025/v1/embeddings"
_EMBEDDING_MODEL = "/home/netzone22/data/LLM/Qwen3-Embedding-8B"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    dsn = os.getenv("CHECKPOINT_DSN") or os.getenv("DATABASE_URL") or cfg.checkpoint_dsn
    if not dsn:
        db_name = os.getenv("DB_NAME") or cfg.db_name or "writing_checkpoint_db"
        if cfg.db_host and cfg.db_port and cfg.db_user and cfg.db_password:
            dsn = f"postgresql://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{db_name}"
    checkpointer = await ensure_checkpoint_tables(dsn) if dsn else None
    
    store = KBStore(
        base_dir=_KB_BASE_DIR,
        collection_name=_KB_COLLECTION,
        embedding_url=_EMBEDDING_URL,
        embedding_model=_EMBEDDING_MODEL,
    )
    kb_client = KBClient(store)    
    
    # agents：注入 checkpointer, kb_client
    heuristic_agent = HeuristicAgent(checkpointer=checkpointer, kb=kb_client)
    help_agent = HelpAgent(checkpointer=checkpointer, kb=kb_client)

    float_agent = FloatAgent()
    industry_agent = IndustryAgent()
    review_agent = ReviewAgent()
    merge_agent = MergeAgent(kb=kb_client)
    outline_agent = OutlineAgent(kb=kb_client)
    polish_agent = PolishAgent()


    # services：注入 agents
    kb_service = KBService(store=store)
    chat_service = ChatService(heuristic_agent=heuristic_agent, help_agent=help_agent)
    others_service = OthersService(float_agent=float_agent, 
                                   industry_agent=industry_agent, 
                                   review_agent=review_agent,
                                   merge_agent=merge_agent,
                                   outline_agent=outline_agent,
                                   polish_agent=polish_agent,
                                   )
    
    app.state.ctx = AppContext(
        checkpointer=checkpointer,
        kb_client=kb_client,
        chat_service=chat_service,
        others_service=others_service,
        kb_admin_service=kb_service,
        review_service=None,  # 逐步迁移，先留空也行
    )
    try:
        yield
    finally:
        if checkpointer is not None:
            await close_checkpointer()
