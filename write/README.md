项目结构：
ai_writer_agent/
  main.py
  routers/
    outline.py         # 大纲、结构化
    heuristic.py       # 启发式写作
    review.py          # 本节/章节/全文评审
    help.py            # “我能帮你”会话
    merge.py           # 一键合入
    polish.py          # 全文润色
    kb.py              # 文档 & 知识库
  services/
    ai_client.py       # 调用模型/Agent
    kb_client.py       # 知识库相关
    writing_service.py
    review_service.py
  models/
    llm_interface_async.py         # 流式模型接口
    chat_db.py              # 数据库连接
  config.py
  config.yaml
  .env