from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class AppContext:

    # 运行时资源（每 worker 1 份）
    checkpointer: Any          # AsyncPostgresSaver
    kb_client: Any             # KBClient / KBStore


    # 用例层 service（对 router 暴露）
    chat_service: Any       # 多轮对话
    others_service: Any     # 其他功能（如大纲、关键点等）
    kb_admin_service: Any   # 导入/删除/清理
    kb_query_service: Optional[Any]   # 检索
