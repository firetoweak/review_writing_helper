from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI

from infra.persistence.checkpointer_pg import init_checkpointer, PgCheckpointSettings
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
    checkpointer = await init_checkpointer(PgCheckpointSettings(dsn="postgresql://..."))
    
    store = KBStore(
        base_dir=_KB_BASE_DIR,
        collection_name=_KB_COLLECTION,
        embedding_url=_EMBEDDING_URL,
        embedding_model=_EMBEDDING_MODEL,
    )
    kb_client = KBClient(store)    
    
    # agents：注入 checkpointer, kb_client
    heuristic_agent = HeuristicAgent(checkpointer=checkpointer)
    help_agent = HelpAgent(checkpointer=checkpointer)

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
    yield
